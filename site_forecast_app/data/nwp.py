"""Functions to load and process NWP data."""

import logging
import os
from importlib.resources import files

import numpy as np
import xarray as xr
import xesmf as xe
from ocf_data_sampler.config.model import NWP
from pydantic import BaseModel

log = logging.getLogger(__name__)


class NWPProcessAndCacheConfig(BaseModel):
    """Configuration for processing and caching NWP data."""

    source_nwp_path: str
    dest_nwp_path: str
    source: str
    config: NWP | None = None


def process_and_cache_nwp(nwp_config: NWPProcessAndCacheConfig) -> None:
    """Reads zarr file, perfroms checks and transformations, and saves zarr to new destination."""
    source_nwp_path = nwp_config.source_nwp_path
    dest_nwp_path = nwp_config.dest_nwp_path

    log.info(
        f"Processing and caching NWP data from {source_nwp_path} "
        f"and saving to {dest_nwp_path} for {nwp_config.source}",
    )

    if os.path.exists(dest_nwp_path):
        log.info(f"File already exists at {dest_nwp_path}")
        return

    # Load dataset from source
    ds = xr.open_zarr(source_nwp_path, consolidated=False).load()

    # Check there are no NaNs in the data
    varname = next(iter(ds.data_vars))
    if np.isnan(ds[varname].values).any():
        raise ValueError(f"Found NaNs in {source_nwp_path}")

    # Drop encodings. This is important to avoid saving errors
    for v in list(ds.coords.keys()):
        ds[v].encoding.clear()

    for v in list(ds.variables.keys()):
        ds[v].encoding.clear()

    if "variable" in ds.coords:
        # make the dtype of variables is strings
        ds["variable"] = ds.variable.astype(str)

    if "channel" in ds.coords:
        # make the dtype of variables is strings
        ds["channel"] = ds.channel.astype(str)


    # lets make sure its in the right order
    if "init_time" in ds.coords:
        ds = ds.transpose("init_time", "step", "variable", "latitude", "longitude")

    # MetOffice Global requires extra processing
    if nwp_config.source == "mo_global":
        ds = maybe_scale_mo_cloud_variables(ds)
        ds = maybe_regrid_mo_global(ds)

    # Save destination path
    log.info(f"Saving NWP data to {dest_nwp_path}")
    ds.to_zarr(dest_nwp_path, mode="a")


def maybe_scale_mo_cloud_variables(ds: xr.Dataset) -> xr.Dataset:
    """Scale cloud variables in MetOffice Global data if requested."""
    scale_mo_global_clouds = os.getenv("MO_GLOBAL_SCALE_CLOUDS", "1") == "1"
    
    if scale_mo_global_clouds:
        log.warning("Scaling MO Global cloud variables from 0-100 to 0-1")

        varname = next(iter(ds.data_vars))

        cloud_vars = [
            "cloud_cover_high",
            "cloud_cover_low",
            "cloud_cover_medium",
            "cloud_cover_total",
        ]
        for cloud_var in cloud_vars:
            if cloud_var in ds.variable.values:
                idx = list(ds.variable.values).index(cloud_var)
                ds[varname][:, :, idx] = ds[varname][:, :, idx] / 100.0

    return ds


def maybe_regrid_mo_global(ds: xr.Dataset) -> xr.Dataset:
    """Regrid MetOffice Global data for Netherlands.

    Netherlands models were trained on data using a different grid
    than what we get in production. This function uses a look-up file
    with training coordinates to transform live data into expected format.
    """
    if os.getenv("CLIENT_NAME", "nl") == "nl":

        target_coords_path  = files("site_forecast_app.data").joinpath("nl_mo_target_coords.nc")

        ds_target_coords = xr.load_dataset(target_coords_path)

        log.info(f"Regridding mo_global to expected grid from {target_coords_path}")

        regridder = xe.Regridder(ds, ds_target_coords, method="bilinear")

        # Iterate over steps to save RAM
        ds_list = []
        for step in ds.step:
            # Copy to make sure the data is C-contiguous for efficient regridding
            ds_step = ds.sel(step=step).copy(deep=True)
            ds_list.append(regridder(ds_step))

        ds = xr.concat(ds_list, dim="step")

    return ds
