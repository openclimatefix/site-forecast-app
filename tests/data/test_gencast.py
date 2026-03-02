import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from freezegun import freeze_time

from site_forecast_app.data.gencast import (
    combine_to_single_init_time,
    compute_ensemble_statistics,
    get_latest_6hr_init_time,
)


class TestGetLatestInitTime:
    @pytest.mark.parametrize(
        "current_time, expected_string",
        [
            # --- Basic Intraday Tests ---
            # 15:00 - 8h = 07:00 -> Floor to 06
            (dt.datetime(2023, 10, 10, 15, 0, tzinfo=dt.UTC), "20231010_06hr"),
            # --- Exact Boundary Tests (The Edge Cases) ---
            # 14:00 - 8h = 06:00 -> Exactly on 06 boundary
            (dt.datetime(2023, 10, 10, 14, 0, tzinfo=dt.UTC), "20231010_06hr"),
            # 13:59 - 8h = 05:59 -> Should drop back to previous block (00)
            (dt.datetime(2023, 10, 10, 13, 59, tzinfo=dt.UTC), "20231010_00hr"),
            # --- Date Rollover Tests (Midnight) ---
            # 05:00 - 8h = 21:00 (previous day) -> Floor to 18
            (dt.datetime(2023, 10, 10, 5, 0, tzinfo=dt.UTC), "20231009_18hr"),
            # 08:00 - 8h = 00:00 (today) -> Exactly on 00 boundary
            (dt.datetime(2023, 10, 10, 8, 0, tzinfo=dt.UTC), "20231010_00hr"),
            # --- Year/Month Rollover Tests ---
            # Jan 1st, 02:00 - 8h = Dec 31st, 18:00 (previous year)
            (dt.datetime(2024, 1, 1, 2, 0, tzinfo=dt.UTC), "20231231_18hr"),
        ],
    )
    def test_explicit_input_times(self, current_time, expected_string):
        """Test the logic with explicit datetime inputs covering various boundaries."""
        result = get_latest_6hr_init_time(now=current_time)
        assert result == expected_string

    @freeze_time("2023-10-10 15:00:00")
    def test_default_argument_none(self):
        """Test that passing None uses the current UTC time (mocked)."""
        # freeze_time makes datetime.now() return 15:00 UTC
        # 15:00 - 8h = 07:00 -> 06:00
        expected = "20231010_06hr"
        assert get_latest_6hr_init_time() == expected


def test_compute_ensemble_statistics():
    # 1. Setup: Create the xarray Dataset with specified dimensions and types
    n_sample = 11  # Using 11 samples (0 to 10) makes median/quantile math clean
    n_lat = 3
    n_lon = 4
    n_time = 2

    # Define coordinates
    sample_coords = np.arange(n_sample, dtype="int64")
    lat_coords = np.linspace(-10, 10, n_lat, dtype="float32")
    lon_coords = np.linspace(-20, 20, n_lon, dtype="float32")
    time_coords = pd.to_timedelta(np.arange(n_time), unit="h").to_numpy()  # timedelta64[ns]
    init_time_val = np.datetime64("2023-01-01 00:00:00")

    # Create Data Variables
    # var_predictable: Values equal the sample index.
    # This allows easy verification: Mean of [0,1,2...10] is 5.0.
    data_linear = np.zeros((n_sample, n_lat, n_lon, n_time))
    for i in range(n_sample):
        data_linear[i, :, :, :] = i

    # var_random: Just to ensure multiple variables are processed
    data_random = np.random.randn(n_sample, n_lat, n_lon, n_time)

    ds = xr.Dataset(
        data_vars={
            "temperature": (("sample", "lat", "lon", "time"), data_linear),
            "irradiance": (("sample", "lat", "lon", "time"), data_random),
        },
        coords={
            "sample": sample_coords,
            "lat": lat_coords,
            "lon": lon_coords,
            "time": time_coords,
            "init_time": init_time_val,  # Scalar coordinate as requested
        },
    )

    result = compute_ensemble_statistics(ds)

    assert "sample" not in result.dims, "The 'sample' dimension should be reduced/removed."
    assert "ens_stat" in result.dims, "The new 'ens_stat' dimension should exist."
    assert "init_time" in result.coords, "The init_time coordinate should be preserved."
    assert set(result.data_vars) == {"temperature", "irradiance"}, (
        "All data variables should be preserved."
    )

    # Check the statistics labels are correct and in expected order
    expected_stats = ["mean", "std", "median", "P10", "P25", "P75", "P90"]
    np.testing.assert_array_equal(result["ens_stat"].values, expected_stats)

    # Statistical Correctness Checks (using 'temperature')
    # Since 'temperature' data is simply 0, 1, 2, ... 10 at every grid point:

    # Check Mean: Average of 0..10 is 5.0
    calc_mean = result["temperature"].sel(ens_stat="mean")
    assert np.allclose(calc_mean, 5.0), "Mean calculation is incorrect."

    # Check Median: Median of 0..10 is 5.0
    calc_median = result["temperature"].sel(ens_stat="median")
    assert np.allclose(calc_median, 5.0), "Median calculation is incorrect."

    # Check Standard Deviation
    # Note: xarray/numpy defaults to ddof=0 (population std) unless specified
    expected_std = np.std(np.arange(n_sample))
    calc_std = result["temperature"].sel(ens_stat="std")
    assert np.allclose(calc_std, expected_std), "Std calculation is incorrect."

    # Check a Quantile (e.g., P90)
    # 90th percentile of 0..10 should be 9.0
    calc_p90 = result["temperature"].sel(ens_stat="P90")
    expected_p90 = np.quantile(np.arange(n_sample), 0.90)
    assert np.allclose(calc_p90, expected_p90), "90th Percentile calculation is incorrect."

    # Check init_time coordinate value
    assert result["init_time"].values == init_time_val


def test_combine_to_single_init_time():
    # Create Init Times: 06:00 and 12:00 (6 hours apart)
    init_times = pd.to_datetime(["2023-01-01T06:00:00", "2023-01-01T12:00:00"])

    # Create Steps: 12-hourly spacing (12h, 24h)
    steps = np.array([np.timedelta64(12, "h"), np.timedelta64(24, "h")])

    # Create Ensemble members
    ensembles = [0, 1, 2]

    # Create coordinate dictionary
    coords = {"init_time": init_times, "time": steps, "ensemble": ensembles}

    # Create dummy data (init_time, step, ensemble)
    data = np.random.rand(len(init_times), len(steps), len(ensembles))

    # Construct Dataset
    ds = xr.Dataset(
        data_vars={"temperature": (["init_time", "step", "ensemble"], data)},
        coords=coords,
    )

    ds_result = combine_to_single_init_time(ds)

    # Check Init Time Reduction
    # Should be reduced to 1 (dropping the first one, keeping the 12:00 one)
    assert ds_result.sizes["init_time"] == 1
    assert ds_result.init_time.values[0] == init_times[1]

    # Check Step Frequency
    # Calculate the new effective steps (time - init_time)
    # The result should be sorted by time, so we check the difference between consecutive steps.

    # Calculate the difference between consecutive steps
    step_diffs = np.diff(ds_result.time.values)

    # Check that the spacing between steps is NOT 12 hours anymore, but involves 6-hour intervals
    # (Note: Depending on the exact overlap logic, we look for 6h gaps or specific 6h values)

    # Ensure all step differences are multiples of 6 hours
    six_hours = np.timedelta64(6, "h")
    assert np.all(step_diffs % six_hours == np.timedelta64(0, "ns")), (
        "Steps are not aligned to 6-hour grid"
    )

    # Specific check: Ensure we have at least one 6-hour interval between steps
    # (e.g., verifying that the interleaving actually happened)
    assert np.any(step_diffs == six_hours), "Failed to interleave to 6-hourly steps"
