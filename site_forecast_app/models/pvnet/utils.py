"""Useful functions for setting up PVNet model."""

import logging
import os

import fsspec
import numpy as np
import torch
import yaml
from ocf_data_sampler.config.model import Configuration
from ocf_data_sampler.config.save import save_yaml_configuration

from .consts import (
    generation_path,
    nwp_ecmwf_path,
    nwp_gencast_path,
    nwp_mo_global_path,
    satellite_path,
)

log = logging.getLogger(__name__)


def populate_data_config_sources(input_path: str, output_path: str) -> dict:
    """Re-save the data config and replace the source filepaths.

    Args:
        input_path: Path to input datapipes configuration file
        output_path: Location to save the output configuration file
    """
    with open(input_path) as infile:
        config = yaml.load(infile, Loader=yaml.FullLoader)  # noqa S506

    production_paths = {
        "pv": {"filename": generation_path},
        "nwp": {
            "ecmwf": nwp_ecmwf_path,
            "mo_global": nwp_mo_global_path,
            "gencast": nwp_gencast_path,
        },
        "satellite": {"filepath": satellite_path},
    }

    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config:
            if nwp_config[nwp_source]["zarr_path"] != "":
                if nwp_source not in production_paths["nwp"]:
                    Exception(f"Missing NWP path: {nwp_source} in production_paths")
                nwp_config[nwp_source]["zarr_path"] = production_paths["nwp"][nwp_source]

            if "forecast_minutes" in nwp_config[nwp_source]:
                nwp_config[nwp_source].pop("forecast_minutes")

            if "dropout_timedeltas_minutes" in nwp_config[nwp_source]:
                nwp_config[nwp_source]["dropout_timedeltas_minutes"] = []
                nwp_config[nwp_source]["dropout_fraction"] = 0

            nwp_config[nwp_source]["interval_end_minutes"] = nwp_config[nwp_source][
                "interval_end_minutes"
            ]

            if os.getenv("CLIENT_NAME", "nl") == "nl" and nwp_source == "mo_global":
                # this is only for NL
                if "wind_u_10m" in nwp_config[nwp_source]["channels"]:
                    nwp_config[nwp_source]["channels"].remove("wind_u_10m")
                    nwp_config[nwp_source]["channels"].append("wind_u_component_10m")
                    nwp_config[nwp_source]["normalisation_constants"]["wind_u_component_10m"] \
                        = nwp_config[nwp_source]["normalisation_constants"]["wind_u_10m"]

                if "wind_v_10m" in nwp_config[nwp_source]["channels"]:
                    nwp_config[nwp_source]["channels"].remove("wind_v_10m")
                    nwp_config[nwp_source]["channels"].append("wind_v_component_10m")
                    nwp_config[nwp_source]["normalisation_constants"]["wind_v_component_10m"] \
                        = nwp_config[nwp_source]["normalisation_constants"]["wind_v_10m"]

    if "satellite" in config["input_data"]:
        satellite_config = config["input_data"]["satellite"]
        satellite_config["zarr_path"] = production_paths["satellite"]["filepath"]
        if "satellite_image_size_pixels_height" in satellite_config:
            satellite_config["image_size_pixels_height"] = satellite_config.pop(
                "satellite_image_size_pixels_height",
            )
        if "satellite_image_size_pixels_width" in satellite_config:
            satellite_config["image_size_pixels_width"] = satellite_config.pop(
                "satellite_image_size_pixels_width",
            )

        # remove any dropout timedeltas
        if "dropout_timedeltas_minutes" in satellite_config:
            satellite_config["dropout_timedeltas_minutes"] = []
            satellite_config["dropout_fraction"] = 0

    if "generation" in config["input_data"]:
        generation_config = config["input_data"]["generation"]
        generation_config["zarr_path"] = generation_path
        # generation_config["metadata_file_path"] = site_metadata_path

        # drop site capacity mode for the moment,
        # this will come in a later release of ocf-data-sampler
        if "capacity_mode" in generation_config:
            generation_config.pop("capacity_mode")

        # remove any dropout timedeltas
        if "dropout_timedeltas_minutes" in generation_config:
            generation_config["dropout_timedeltas_minutes"] = []
            generation_config["dropout_fraction"] = 0

    # add solar position
    config["input_data"]["solar_position"] = {}
    for k in ["time_resolution_minutes", "interval_end_minutes", "interval_start_minutes"]:
        config["input_data"]["solar_position"][k] = config["input_data"]["generation"][k]

    configuration = Configuration(**config)
    save_yaml_configuration(configuration, output_path)

    return config


def set_night_time_zeros(
    batch: dict,
    preds: np.ndarray,
    t0_idx: int,
    sun_elevation_limit: float = 0.0,
) -> np.ndarray:
    """Set all predictions to zero for night time values."""
    log.debug("Setting night time values to zero")
    # get sun elevation values and if less 0, set to 0
    key = "solar_elevation"

    sun_elevation = batch[key]
    if not isinstance(sun_elevation, np.ndarray):
        sun_elevation = sun_elevation.detach().cpu().numpy()

    # The dataloader normalises solar elevation data to the range [0, 1]
    sun_elevation = (sun_elevation - 0.5) * 180

    # expand dimension from (1,197) to (1,197,7), 7 is due to the number plevels
    n_plevels = preds.shape[2]
    sun_elevation = np.repeat(sun_elevation[:, :, np.newaxis], n_plevels, axis=2)
    # only take future time steps
    sun_elevation = sun_elevation[:, t0_idx + 1 :, :]
    preds[sun_elevation < sun_elevation_limit] = 0

    return preds


def save_batch(
    batch: dict,
    model_name: str,
    site_uuid: str,
    save_batches_dir: str | None = None,
) -> None:
    """Save batch to SAVE_BATCHES_DIR if set.

    Args:
        batch: The batch to save
        model_name: The name of the model
        site_uuid: The site_uuid of the site
        save_batches_dir: The directory to save the batch to,
            defaults to environment variable SAVE_BATCHES_DIR
    """
    if save_batches_dir is None:
        save_batches_dir = os.getenv("SAVE_BATCHES_DIR", None)

    if save_batches_dir is not None:
        log.info(f"Saving batch to {save_batches_dir}")

        local_filename = f"batch_{model_name}_{site_uuid}.pt"
        torch.save(batch, local_filename)

        fs = fsspec.open(save_batches_dir).fs
        fs.put(local_filename, f"{save_batches_dir}/{local_filename}")
