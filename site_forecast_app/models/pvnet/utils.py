"""Useful functions for setting up PVNet model."""
import logging
import os
import tempfile
import zipfile
from uuid import UUID

import fsspec
import numpy as np
import torch
import xarray as xr
import yaml
from ocf_data_sampler.config.model import NWP, Configuration
from ocf_data_sampler.config.save import save_yaml_configuration
from pydantic import BaseModel

from .consts import (
    nwp_ecmwf_path,
    nwp_mo_global_path,
    satellite_path,
    site_metadata_path,
    site_netcdf_path,
)

log = logging.getLogger(__name__)


class NWPProcessAndCacheConfig(BaseModel):
    """Configuration for processing and caching NWP data."""

    source_nwp_path: str
    dest_nwp_path: str
    source: str
    config: NWP | None = None


def populate_data_config_sources(input_path:str, output_path:str) -> dict:
    """Re-save the data config and replace the source filepaths.

    Args:
        input_path: Path to input datapipes configuration file
        output_path: Location to save the output configuration file
    """
    with open(input_path) as infile:
        config = yaml.load(infile, Loader=yaml.FullLoader)  # noqa S506

    production_paths = {
        "pv": {"filename": site_netcdf_path, "metadata_filename": site_metadata_path},
        "nwp": {"ecmwf": nwp_ecmwf_path, "mo_global": nwp_mo_global_path},
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

        # Remove any hard coding about satellite delay
        if "live_delay_minutes" in satellite_config:
            satellite_config.pop("live_delay_minutes")

    if "site" in config["input_data"]:
        site_config = config["input_data"]["site"]
        site_config["file_path"] = site_netcdf_path
        site_config["metadata_file_path"] = site_metadata_path

        # drop site capacity mode for the moment,
        # this will come in a later release of ocf-data-sampler
        if "capacity_mode" in site_config:
            site_config.pop("capacity_mode")

    # add solar position
    config["input_data"]["solar_position"] = {}
    for k in ["time_resolution_minutes", "interval_end_minutes", "interval_start_minutes"]:
        config["input_data"]["solar_position"][k] = config["input_data"]["site"][k]

    configuration = Configuration(**config)
    save_yaml_configuration(configuration, output_path)

    return config


def process_and_cache_nwp(nwp_config: NWPProcessAndCacheConfig) -> None:
    """Reads zarr file, renames t variable to t2m and saves zarr to new destination."""
    source_nwp_path = nwp_config.source_nwp_path
    dest_nwp_path = nwp_config.dest_nwp_path

    log.info(
        f"Processing and caching NWP data for {source_nwp_path} "
        f"and saving to {dest_nwp_path} for {nwp_config.source}",
    )

    if os.path.exists(dest_nwp_path):
        log.info(f"File already exists at {dest_nwp_path}")
        return

    # Load dataset from source
    ds = xr.open_zarr(source_nwp_path)

    # This is important to avoid saving errors
    for v in list(ds.coords.keys()):
        if ds.coords[v].dtype == object:
            ds[v].encoding.clear()

    for v in list(ds.variables.keys()):
        if ds[v].dtype == object:
            ds[v].encoding.clear()

    # Save destination path
    log.info(f"Saving NWP data to {dest_nwp_path}")
    ds.to_zarr(dest_nwp_path, mode="a")


def download_satellite_data(satellite_source_file_path: str) -> None:
    """Download the sat data."""
    if os.path.exists(satellite_path):
        log.info(f"File already exists at {satellite_path}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:

        temporary_satellite_data = f"{tmpdir.name}/temporary_satellite_data.zarr"

        # download satellite data
        fs = fsspec.open(satellite_source_file_path).fs
        if fs.exists(satellite_source_file_path):
            log.info(
                f"Downloading satellite data from {satellite_source_file_path} "
                "to sat_min.zarr.zip",
            )
            fs.get(satellite_source_file_path, "sat_min.zarr.zip")
            log.info(f"Unzipping sat_min.zarr.zip to {satellite_path}")

            with zipfile.ZipFile("sat_min.zarr.zip", "r") as zip_ref:
                zip_ref.extractall(temporary_satellite_data)
        else:
            log.error(f"Could not find satellite data at {satellite_source_file_path}")

        # log the timestamps for satellite data
        ds = xr.open_zarr(temporary_satellite_data)
        log.info(f"Satellite data timestamps: {ds.time.values}, now scaling to 0-1")

        # scale
        ds = ds / 1023

        # save the dataset
        ds.to_zarr(satellite_path, mode="a")


def set_night_time_zeros(
    batch: dict, preds: torch.Tensor, t0_idx: int, sun_elevation_limit: float = 0.0,
) -> torch.Tensor:
    """Set all predictions to zero for night time values."""
    log.debug("Setting night time values to zero")
    # get sun elevation values and if less 0, set to 0
    key = "solar_elevation"

    sun_elevation = batch[key]
    if not isinstance(sun_elevation, np.ndarray):
        sun_elevation = sun_elevation.detach().cpu().numpy()

    # expand dimension from (1,197) to (1,197,7), 7 is due to the number plevels
    n_plevels = preds.shape[2]
    sun_elevation = np.repeat(sun_elevation[:, :, np.newaxis], n_plevels, axis=2)
    # only take future time steps
    sun_elevation = sun_elevation[:, t0_idx + 1 :, :]
    preds[sun_elevation < sun_elevation_limit] = 0

    return preds


def save_batch(
    batch: dict,
    i: int,
    model_name: str,
    site_uuid: UUID,
    save_batches_dir: str | None = None,
) -> None:
    """Save batch to SAVE_BATCHES_DIR if set.

    Args:
        batch: The batch to save
        i: The index of the batch
        model_name: The name of the
        site_uuid: The site_uuid of the site
        save_batches_dir: The directory to save the batch to,
            defaults to environment variable SAVE_BATCHES_DIR
    """
    if save_batches_dir is None:
        save_batches_dir = os.getenv("SAVE_BATCHES_DIR", None)

    if save_batches_dir is not None:
        log.info(f"Saving batch {i} to {save_batches_dir}")

        local_filename = f"batch_{i}_{model_name}_{site_uuid}.pt"
        torch.save(batch, local_filename)

        fs = fsspec.open(save_batches_dir).fs
        fs.put(local_filename, f"{save_batches_dir}/{local_filename}")
