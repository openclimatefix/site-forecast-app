""" Tests for utils for pvnet"""
import os
import tempfile

import xarray as xr
import zarr

from site_forecast_app.data.satellite import download_satellite_data, satellite_scale_minmax


def test_satellite_scale_minmax(small_satellite_data, # noqa: ARG001
                                ) -> None:
    """Test for scaling satellite data using min-max scaling."""

    with zarr.storage.ZipStore(os.getenv("SATELLITE_ZARR_PATH"), mode="r") as store:
        ds = xr.open_zarr(store)

    ds_scaled = satellite_scale_minmax(ds)

    assert ds_scaled.data.min() >= 0
    assert ds_scaled.data.max() <= 1
    assert ds_scaled.data.shape == ds.data.shape


def test_satellite_download(small_satellite_data, # noqa: ARG001
                                ) -> None:
    """Test for scaling satellite data using min-max scaling."""

    with tempfile.TemporaryDirectory() as tmpdir:
        local_satellite_path = f"{tmpdir}/satellite_data.zarr"
        download_satellite_data(satellite_source_file_path=os.getenv("SATELLITE_ZARR_PATH"),
                                local_satellite_path=local_satellite_path,
                                scaling_method="minmax")


def test_satellite_download_backup(small_satellite_data, # noqa: ARG001
                                ) -> None:
    """Test for scaling satellite data using min-max scaling."""

    with zarr.storage.ZipStore(os.getenv("SATELLITE_ZARR_PATH"), mode="r") as store:
        ds = xr.open_zarr(store)

    # only select the first 2 timestamps
    ds = ds.isel(time=slice(0, 2))

    ds_15 = ds.resample(time="15T").mean().copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_satellite_path = f"{tmpdir}/satellite_data.zarr"
        local_satellite_backup_path = f"{tmpdir}/satellite_15_data.zarr"

        with zarr.storage.ZipStore(local_satellite_backup_path, mode="x") as store:
            ds_15.to_zarr(store)

        download_satellite_data(satellite_source_file_path=os.getenv("SATELLITE_ZARR_PATH"),
                                local_satellite_path=local_satellite_path,
                                scaling_method="minmax",
                                satellite_backup_source_file_path=local_satellite_backup_path)


