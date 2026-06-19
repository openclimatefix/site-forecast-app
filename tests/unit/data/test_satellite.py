""" Tests for utils for pvnet"""
import tempfile

import pandas as pd
import xarray as xr
import zarr

from site_forecast_app.data.satellite import (
    check_model_satellite_inputs_available,
    download_satellite_data,
    satellite_scale_minmax,
)


def test_satellite_scale_minmax(small_satellite_data,
                                ) -> None:
    """Test for scaling satellite data using min-max scaling."""

    with zarr.storage.ZipStore(small_satellite_data, mode="r") as store:
        ds = xr.open_zarr(store)

    ds_scaled = satellite_scale_minmax(ds)

    assert ds_scaled.data.min() >= 0
    assert ds_scaled.data.max() <= 1
    assert ds_scaled.data.shape == ds.data.shape


def test_satellite_download(small_satellite_data,
                                ) -> None:
    """Test for scaling satellite data using min-max scaling."""

    with tempfile.TemporaryDirectory() as tmpdir:
        local_satellite_path = f"{tmpdir}/satellite_data.zarr"
        download_satellite_data(satellite_source_file_path=small_satellite_data,
                                local_satellite_path=local_satellite_path,
                                scaling_method="minmax")


def test_satellite_download_backup(small_satellite_data,
                                ) -> None:
    """Test for scaling satellite data using min-max scaling."""

    with zarr.storage.ZipStore(small_satellite_data, mode="r") as store:
        ds = xr.open_zarr(store)

    # only select the first 2 timestamps
    ds = ds.isel(time=slice(0, 2))

    ds_15 = ds.resample(time="15min").mean().copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_satellite_path = f"{tmpdir}/satellite_data.zarr"
        local_satellite_backup_path = f"{tmpdir}/satellite_15_data.zarr"

        with zarr.storage.ZipStore(local_satellite_backup_path, mode="x") as store:
            ds_15.to_zarr(store)

        download_satellite_data(satellite_source_file_path=small_satellite_data,
                                local_satellite_path=local_satellite_path,
                                scaling_method="minmax",
                                satellite_backup_source_file_path=local_satellite_backup_path)


def test_check_model_satellite_inputs_available(config_filename) -> None:
    """Check satellite availability across full coverage, delay, and gap scenarios."""
    t0 = pd.Timestamp("2023-01-01 00:00")
    sat_datetime_1 = pd.date_range(
        t0 - pd.Timedelta("120min"),
        t0 - pd.Timedelta("5min"),
        freq="5min",
    )
    sat_datetime_2 = pd.date_range(
        t0 - pd.Timedelta("120min"),
        t0 - pd.Timedelta("15min"),
        freq="5min",
    )
    sat_datetime_3 = pd.date_range(
        t0 - pd.Timedelta("120min"),
        t0 - pd.Timedelta("35min"),
        freq="5min",
    )
    sat_datetime_4 = pd.to_datetime([t for t in sat_datetime_1 if t != t0 - pd.Timedelta("30min")])
    sat_datetime_5 = pd.to_datetime([t for t in sat_datetime_1 if t != t0 - pd.Timedelta("60min")])

    assert check_model_satellite_inputs_available(config_filename, t0, sat_datetime_1, country="nl")
    assert check_model_satellite_inputs_available(config_filename, t0, sat_datetime_2, country="nl")
    assert not check_model_satellite_inputs_available(
        config_filename, t0, sat_datetime_3, country="nl",
    )
    assert not check_model_satellite_inputs_available(
        config_filename, t0, sat_datetime_4, country="nl",
    )
    assert not check_model_satellite_inputs_available(
        config_filename, t0, sat_datetime_5, country="nl",
    )
