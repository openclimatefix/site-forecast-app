"""Functions to get/transform GenCast data."""

import datetime as dt
import json
import logging
import os

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

# Variable list and spatial/time selection
WEATHER_VARS = [
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
]

SEL_KWARGS = {
    "lat": slice(6, 35),
    "lon": slice(67, 97),
    "time": slice(None, np.timedelta64(96, "h")),
}


def select_relevant_data(ds: xr.Dataset) -> xr.Dataset:
    """Select required variables and spatial/time domain."""
    return ds[WEATHER_VARS].sel(**SEL_KWARGS)


def get_latest_6hr_init_time(now: dt.datetime | None = None) -> str:
    """Returns the latest 6-hourly init time string in a specified format.

    string format: YYYYMMDD_XXhr
    where XX is one of 00, 06, 12, 18
    """
    if now is None:
        now = dt.datetime.now(tz=dt.UTC)  # use UTC by default

    # Account for data availability delay
    expected_delay_hours = 8
    t = now - dt.timedelta(hours=expected_delay_hours)

    # Find the most recent 6-hour boundary
    cycle_hour = (t.hour // 6) * 6

    # Construct the init datetime
    cycle_time = t.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)

    # Format output to required string format
    return cycle_time.strftime("%Y%m%d_%Hhr")


def compute_ensemble_statistics(ds: xr.Dataset) -> xr.Dataset:
    """Compute statistics e.g. mean, std etc over 'sample' (ensemble member) for all variables."""
    mean_ds = ds.mean("sample")
    std_ds = ds.std("sample")

    # Compute all percentiles at once
    quantiles_ds = ds.quantile([0.5, 0.10, 0.25, 0.75, 0.90], dim="sample")

    # Rename percentile dimension to match ens_stat labels
    quantiles_ds = quantiles_ds.rename({"quantile": "ens_stat"})
    quantiles_ds = quantiles_ds.assign_coords(ens_stat=["median", "P10", "P25", "P75", "P90"])

    # Stack all stats into a single Dataset along new 'ens_stat' dimension
    combined = xr.concat(
        [
            mean_ds.assign_coords(ens_stat="mean"),
            std_ds.assign_coords(ens_stat="std"),
            quantiles_ds,
        ],
        dim="ens_stat",
        coords="minimal",
    )

    return combined


def combine_to_single_init_time(ds: xr.Dataset) -> xr.Dataset:
    """Combine two 12-hourly forecasts into one 6-hourly forecast."""
    # Shift dataset so each init_time aligns with previous one
    ds_prev = ds.shift(init_time=1, fill_value=np.nan)

    # Adjust the previous step values:
    # original steps: 12h, 24h, 36h, ...
    # want:           6h,  18h, 30h, ...
    ds_prev = ds_prev.assign_coords(time=ds_prev.time - np.timedelta64(6, "h"))

    # Merge previous+current forecasts to get all 6-hourly steps
    ds_merged = xr.concat([ds_prev, ds], dim="time").sortby("time")
    del ds, ds_prev

    # Drop the first init_time (no previous init time)
    ds_merged = ds_merged.isel(init_time=slice(1, None))
    return ds_merged


def stack_ensemble_stats_into_channels(da: xr.DataArray) -> xr.DataArray:
    """Stack ensemble statistics and variables into single channel dimension."""
    data_combined = da.stack(channel_combined=("ens_stat", "variable"))
    data_combined = data_combined.rename({"channel_combined": "channel"})

    # Create new coordinate names in the correct order
    new_channel_coords = [
        f"{stat}_{chan}"
        for stat in da.coords["ens_stat"].to_index()
        for chan in da.coords["variable"].to_index()
    ]
    data_combined = data_combined.assign_coords(channel=new_channel_coords)

    return data_combined


def pull_gencast_data(gcs_bucket_path: str, output_path: str) -> None:
    """Get GenCast data sliced to region of interest for two most recent init times.

    Combines the last two init times to get 6 hourly forecast target times and reshapes
    into the required format for ocf-data-sampler.

    Args:
        gcs_bucket_path: Path to where GenCast data is stored.
        output_path: Path to save the processed GenCast data.

    Returns:
        An xarray Dataset containing the GenCast data in the format required for ocf-data-sampler.
    """
    # Get latest initialised forecasts
    last_expected_init_time = get_latest_6hr_init_time()
    previous_expected_init_time = get_latest_6hr_init_time(
        now=dt.datetime.now(tz=dt.UTC) - dt.timedelta(hours=6),
    )

    try:
        zarr_path1 = f"{gcs_bucket_path}/{last_expected_init_time}_01_preds/predictions.zarr"
        zarr_path2 = f"{gcs_bucket_path}/{previous_expected_init_time}_01_preds/predictions.zarr"

        # Grab GCS token path and only use it if it exists
        token_path = os.getenv("GCS_TOKEN_PATH", None)

        if token_path is None:
            storage_option = {}
        else:
            with open(token_path) as f:
                token_dict = json.load(f)
            storage_option = (
                {
                    "token": token_dict,
                },
            )

        latest_init_time_nwp_ds = xr.open_zarr(
            zarr_path1,
            decode_timedelta=True,
            storage_options=storage_option,
        )
        previous_init_time_nwp_ds = xr.open_zarr(
            zarr_path2,
            decode_timedelta=True,
            storage_options=storage_option,
        )

        log.info("Successfully opened GenCast data from GCS (lazy) with token.")

    except Exception as e:
        raise RuntimeError(
            f"Error loading GenCast data from {gcs_bucket_path}",
        ) from e

    # Combine both datasets along init_time
    ds_raw = xr.concat(
        [latest_init_time_nwp_ds, previous_init_time_nwp_ds],
        dim="init_time",
    ).sortby("init_time")
    # Apply relevant slicing
    ds_sliced = select_relevant_data(ds_raw)

    # Compute ensemble statistics
    ds_ens_stats = compute_ensemble_statistics(ds_sliced)

    # Compute ensemble statistics
    ds_ens_stats = ds_ens_stats.load()
    log.info("Loaded GenCast data into memory.")

    # Transform to single init_time with 6-hourly steps
    ds_merged = combine_to_single_init_time(ds_ens_stats)

    da_merged = ds_merged.to_array(name="gencast_data")

    # Stack ensemble statistics and variables into single channel dimension
    data_combined = stack_ensemble_stats_into_channels(da_merged)

    # Rename dimensions
    data_combined = data_combined.rename(
        {
            "lat": "latitude",
            "lon": "longitude",
            "time": "step",
            "init_time": "init_time_utc",
        },
    )
    # Save to zarr path
    data_combined = data_combined.drop_encoding()
    data_combined = data_combined.astype("float32")
    data_combined = data_combined.chunk(data_combined.shape)

    data_combined.to_zarr(output_path, mode="w")
    log.info("Successfully saved processed GenCast data to zarr path.")
