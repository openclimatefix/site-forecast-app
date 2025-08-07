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


def populate_data_config_sources(input_path: str, output_path: str) -> dict:
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

        # remove any dropout timedeltas
        if "dropout_timedeltas_minutes" in satellite_config:
            satellite_config["dropout_timedeltas_minutes"] = []
            satellite_config["dropout_fraction"] = 0

    if "site" in config["input_data"]:
        site_config = config["input_data"]["site"]
        site_config["file_path"] = site_netcdf_path
        site_config["metadata_file_path"] = site_metadata_path

        # drop site capacity mode for the moment,
        # this will come in a later release of ocf-data-sampler
        if "capacity_mode" in site_config:
            site_config.pop("capacity_mode")

        # remove any dropout timedeltas
        if "dropout_timedeltas_minutes" in site_config:
            site_config["dropout_timedeltas_minutes"] = []
            site_config["dropout_fraction"] = 0

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

    # make the dtype of variables is strings
    ds["variable"] = ds.variable.astype(str)

    name = next(iter(ds.data_vars))
    scale_mo_global_clouds = os.getenv("MO_GLOBAL_SCALE_CLOUDS", "1") == "1"
    if nwp_config.source == "mo_global" and scale_mo_global_clouds:
        log.warning("Scaling MO Global cloud variables by from 0-100 to 0-1")

        cloud_vars = [
            "cloud_cover_high",
            "cloud_cover_low",
            "cloud_cover_medium",
        ]
        for cloud_var in cloud_vars:
            idx = list(ds.variable.values).index(cloud_var)
            ds[name][:, :, idx] = ds[name][:, :, idx] / 100.0

    # TODO this is temporary, NL data is too small
    expand_ecmwf = os.getenv("EXPAND_ECMWF", "1") == "1"
    if nwp_config.source == "ecmwf" and expand_ecmwf:

        log.info(
            "Expanding ECMWF data by 6 hours into the future and adding an extra 0.5 degree down",
        )
        chunks = {"init_time": 1, "step": 1, "latitude": -1, "longitude": -1, "variable": 1}
        # expand time by 6 hours
        for _ in range(6):
            new_step_value = ds.step.values[-1] + 3600000000000
            new_step = ds.interp(
                step=new_step_value,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            ds = xr.concat([ds, new_step], dim="step")
            ds = ds.chunk(chunks)

        log.info("Done extending by time, now extending by latitude")
        # expand by 0.5 degree down
        # note lat is descending
        for _ in range(5):
            new_lower_lat = ds.sel(latitude=ds.latitude.values[-1])
            new_lower_lat.__setitem__("latitude", ds.latitude.values[-1] - 0.1)
            ds = xr.concat([ds, new_lower_lat], dim="latitude")
            ds = ds.chunk(chunks)

        log.info("Done Expanding ECMWF")

        for var in ds:
            del ds[var].encoding["chunks"]

    # Save destination path
    log.info(f"Saving NWP data to {dest_nwp_path}")
    ds.to_zarr(dest_nwp_path, mode="a")


def download_satellite_data(satellite_source_file_path: str,
                            scaling_method: str = "constant") -> None:
    """Download the sat data."""
    if os.path.exists(satellite_path):
        log.info(f"File already exists at {satellite_path}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:

        temporary_satellite_data = f"{tmpdir}/temporary_satellite_data.zarr"

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

        if scaling_method == "constant":
            log.info("Scaling satellite data to [0,1] range via constant scaling")
            # scale the dataset to 0-1

            scale_factor = int(os.environ.get("SATELLITE_SCALE_FACTOR", 1023))
            log.info(f"Scaling satellite data by {scale_factor} to be between 0 and 1")

            ds = ds / scale_factor
        elif scaling_method == "minmax":
            log.info("Scaling satellite data to [0,1] range via min-max scaling")
            # scale the dataset to min-max
            ds = satellite_scale_minmax(ds)
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")

        # save the dataset
        ds.to_zarr(satellite_path, mode="a")

def satellite_scale_minmax(ds: xr.Dataset) -> xr.Dataset:
    """Scale the satellite dataset via min-max to [0,1] range."""
    log.info("Scaling satellite data to 0,1] range via min-max")

    channels = ds.variable.values
    # min and max values for each variable (same length as `variable`
    # and in the same order)
    min_vals = np.array(
            [
                -2.5118103,
                -64.83977,
                63.404694,
                2.844452,
                199.10002,
                -17.254883,
                -26.29155,
                -1.1009827,
                -2.4184198,
                199.57048,
                198.95093,
            ])
    max_vals = np.array(
            [
                69.60857,
                339.15588,
                340.26526,
                317.86752,
                313.2767,
                315.99194,
                274.82297,
                93.786545,
                101.34922,
                249.91806,
                286.96323,
            ])

    # Create DataArrays for min and max with the 'variable' dimension
    min_da = xr.DataArray(min_vals, coords={"variable": channels}, dims=["variable"])
    max_da = xr.DataArray(max_vals, coords={"variable": channels}, dims=["variable"])

    # Apply scaling
    scaled_ds = (ds - min_da) / (max_da - min_da)
    scaled_ds = scaled_ds.clip(min=0, max=1)  # Ensure values are within [0, 1]
    return scaled_ds

def set_night_time_zeros(
    batch: dict,
    preds: torch.Tensor,
    t0_idx: int,
    sun_elevation_limit: float = 0.0,
) -> torch.Tensor:
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
