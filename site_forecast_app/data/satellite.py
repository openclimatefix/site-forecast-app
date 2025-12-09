"""Download and unzip satellite data."""

import logging
import os
import tempfile
from datetime import UTC, timedelta

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import zarr

log = logging.getLogger(__name__)


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
        ],
    )
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
        ],
    )

    # Create DataArrays for min and max with the 'variable' dimension
    min_da = xr.DataArray(min_vals, coords={"variable": channels}, dims=["variable"])
    max_da = xr.DataArray(max_vals, coords={"variable": channels}, dims=["variable"])

    # Apply scaling
    scaled_ds = (ds - min_da) / (max_da - min_da)
    scaled_ds = scaled_ds.clip(min=0, max=1)  # Ensure values are within [0, 1]
    return scaled_ds


def download_satellite_data(
    satellite_source_file_path: str,
    local_satellite_path: str,
    scaling_method: str = "constant",
    satellite_backup_source_file_path: None | str = None,
) -> None:
    """Download the sat data."""
    if os.path.exists(local_satellite_path):
        log.info(f"File already exists at {local_satellite_path}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        temporary_satellite_data = f"{tmpdir}/temporary_satellite_data.zarr"

        ds = download_and_unzip(file_zip=satellite_source_file_path, file=temporary_satellite_data)

        # log the timestamps for satellite data
        times = ds.time.values
        log.info(f"Satellite data timestamps: {times}")

        # possibily download backup satellite
        # if there are not enough time in the current satellite data
        latest_satellite_time = times.max()
        now = pd.Timestamp.now(tz=UTC).replace(tzinfo=None)
        satellite_delay = now - latest_satellite_time

        log.info(f"Latest satellite time: {latest_satellite_time}")

        if satellite_backup_source_file_path and satellite_delay > timedelta(minutes=30):
            log.info(
                "Not enough satellite data available"
                f"downloading backup from {satellite_backup_source_file_path}",
            )

            temporary_satellite_data = f"{tmpdir}/temporary_satellite_backup_data.zarr"
            ds = download_and_unzip(
                file_zip=satellite_backup_source_file_path,
                file=temporary_satellite_data,
                temp_zarr_zip="sat_backup.zarr.zip",
            )

            times = ds.time.values
            log.info(f"Satellite data timestamps: {times}, before resampling to 5 min")

            # resample satellite data to 5 minutely data
            dense_times = pd.date_range(ds.time.values.min(), ds.time.values.max(), freq="5min")
            ds = ds.interp(time=dense_times, method="linear", assume_sorted=True)
            times = ds.time.values
            log.info(f"Satellite data timestamps: {times}")

        elif timedelta(minutes=0) < satellite_delay <= timedelta(minutes=30):
            log.info(
                "Satellite delay is 5 minuted or less. "
                f"Appending a NaN timestamp at {latest_satellite_time + pd.Timedelta('5min')}",
            )

            # Extend the data with NaNs
            ds = ds.reindex(
                time=np.concatenate(
                    [
                        ds.time,
                        [np.datetime64(latest_satellite_time + pd.Timedelta("5min"))],
                    ],
                ),
                fill_value=np.nan,
            )

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

        # This is important to avoid saving errors
        for v in list(ds.coords.keys()):
            ds[v].encoding.clear()

        for v in list(ds.variables.keys()):
            ds[v].encoding.clear()

        # make the dtype of variables is strings
        ds["variable"] = ds.variable.astype(str)

        # make sure d.data.attrs['area'] is a string
        if "area" in ds.data.attrs and isinstance(ds.data.attrs["area"], dict):
            log.warning("Converting area attrs from dict to string")
            ds.data.attrs["area"] = yaml.dump(ds.data.attrs["area"])

        # save the dataset
        ds = ds.chunk(
            chunks={
                "time": len(ds.time),
                "y_geostationary": len(ds.y_geostationary) // 4,
                "x_geostationary": len(ds.x_geostationary) // 4,
                "variable": len(ds.variable),
            },
        )
        ds.to_zarr(local_satellite_path, mode="a")


def download_and_unzip(file_zip: str, file: str, temp_zarr_zip: str = "sat_min.zarr.zip") -> None:
    """Download and unzip the satellite data.

    :param file_zip: The path to the zip file containing the satellite data.
    :param file: The path to the directory where the data should be extracted.
    """
    # download satellite data
    fs = fsspec.open(file_zip).fs
    if fs.exists(file_zip):
        log.info(
            f"Downloading satellite data from {file_zip} to {temp_zarr_zip}",
        )
        fs.get(file_zip, temp_zarr_zip)
        log.info(f"Unzipping {temp_zarr_zip} to {file}")

        store = zarr.storage.ZipStore(path=temp_zarr_zip, mode="r")
        return xr.open_zarr(store)
    else:
        log.error(f"Could not find satellite data at {file_zip}")
