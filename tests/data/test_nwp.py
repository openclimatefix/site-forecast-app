import os
from importlib.resources import files

import numpy as np
import pytest
import xarray as xr

from site_forecast_app.data.nwp import (
    NWPProcessAndCacheConfig,
    regrid_mo_global,
    scale_mo_cloud_variables,
    process_and_cache_nwp,
)


def test_scale_mo_cloud_variables_true(nwp_mo_global_data_nl): # noqa: ARG001
    """Test cloud variables are scaled when requested."""
    nwp_ds = xr.open_zarr(os.environ["NWP_MO_GLOBAL_ZARR_PATH"])
    nwp_ds = nwp_ds.transpose("init_time", "step", "variable", "latitude", "longitude")

    # Set environment variable to request scaling
    os.environ["MO_GLOBAL_SCALE_CLOUDS"] = "1"

    output_ds = scale_mo_cloud_variables(nwp_ds)

    # Check cloud variables are on the scale of [0,1]
    cloud_vars = [
            "cloud_cover_high",
            "cloud_cover_low",
            "cloud_cover_medium",
            "cloud_cover_total",
    ]
    for var in cloud_vars:
        assert output_ds["mo_global"].sel(variable=var).data.max() == 1


def test_scale_mo_cloud_variables_false(nwp_mo_global_data_nl): # noqa: ARG001
    """Test cloud variables are NOT scaled if not requested."""
    nwp_ds = xr.open_zarr(os.environ["NWP_MO_GLOBAL_ZARR_PATH"])

    # Set environment variable to request scaling
    os.environ["MO_GLOBAL_SCALE_CLOUDS"] = "0"

    output_ds = scale_mo_cloud_variables(nwp_ds)

    # Check cloud variables are on the scale of [0,1]
    cloud_vars = [
            "cloud_cover_high",
            "cloud_cover_low",
            "cloud_cover_medium",
            "cloud_cover_total",
    ]
    for var in cloud_vars:
        assert output_ds["mo_global"].sel(variable=var).data.max() == 100


def test_regrid_mo_global_nl(nwp_mo_global_data_nl): # noqa: ARG001
    """Test MetOffice Global is regridded for NL."""
    nwp_ds = xr.open_zarr(os.environ["NWP_MO_GLOBAL_ZARR_PATH"])

    os.environ["CLIENT_NAME"] = "nl"

    target_coords_path  = files("site_forecast_app.data").joinpath("nl_mo_target_coords.nc")
    ds_target_coords = xr.load_dataset(target_coords_path)

    output_ds = regrid_mo_global(nwp_ds)

    # Check regridding returned numerical data
    assert not np.isnan(output_ds["mo_global"].values).any()
    # Check latitude and longitude are as expected
    assert all(output_ds.latitude.data == ds_target_coords.latitude.data)
    assert all(output_ds.longitude.data == ds_target_coords.longitude.data)


def test_regrid_mo_global_india(nwp_mo_global_data_india): # noqa: ARG001
    """Test MetOffice Global is NOT regridded for India."""
    nwp_ds = xr.open_zarr(os.environ["NWP_MO_GLOBAL_ZARR_PATH"])

    os.environ["CLIENT_NAME"] = "ad"

    output_ds = regrid_mo_global(nwp_ds)

    # Check latitude and longitude have not changed
    assert all(output_ds.latitude.data == nwp_ds.latitude.data)
    assert all(output_ds.longitude.data == nwp_ds.longitude.data)


def test_process_and_cash_nwp_raise_nan_error(tmp_path_factory, nwp_data_with_nans): # noqa: ARG001
    """Test an error is raised when there are NaNs in the NWP data."""
    config = NWPProcessAndCacheConfig(
                    source_nwp_path=os.environ["NWP_ECMWF_NANS_ZARR_PATH"],
                    dest_nwp_path=f"{tmp_path_factory.mktemp('data')}/nwp_ecmwf_nans_save.zarr",
                    source="ecmwf",
                )
    with pytest.raises(ValueError):
        process_and_cache_nwp(config)
