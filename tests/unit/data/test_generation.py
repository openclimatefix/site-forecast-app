import numpy as np
import pandas as pd
import xarray as xr

from site_forecast_app.data.generation import (
    format_generation_data,
)


def test_interpolation_limit_within_one_hour():
    """Test that the t0 NaN is filled when the gap is within the 1-hour (4-step) limit."""
    # Create 15-min data with a known value just before t0
    times = pd.date_range("2024-01-01 00:00", periods=6, freq="15min")
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan]}, index=times)

    result = df.interpolate(method="quadratic", fill_value="extrapolate", limit=4)

    # The single NaN at the end is within 1 step — it should be filled
    assert not np.isnan(result["value"].iloc[-1])


def test_interpolation_limit_beyond_one_hour():
    """Test that NaNs beyond the 1-hour (4-step) limit remain NaN after extrapolation."""
    # Create 15-min data where the last real value is more than 1 hour before t0
    times = pd.date_range("2024-01-01 00:00", periods=10, freq="15min")
    values = [1.0, 2.0, 3.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    df = pd.DataFrame({"value": values}, index=times)

    result = df.interpolate(method="quadratic", fill_value="extrapolate", limit=4)

    # Only 4 steps after the last real value should be filled; beyond that remains NaN
    assert not np.isnan(result["value"].iloc[3])  # 1st NaN (1 step) — filled
    assert not np.isnan(result["value"].iloc[4])  # 2nd NaN (2 steps) — filled
    assert not np.isnan(result["value"].iloc[5])  # 3rd NaN (3 steps) — filled
    assert not np.isnan(result["value"].iloc[6])  # 4th NaN (4 steps) — filled
    assert np.isnan(result["value"].iloc[7])  # 5th NaN (5 steps) — remains NaN
    assert np.isnan(result["value"].iloc[8])  # 6th NaN (6 steps) — remains NaN
    assert np.isnan(result["value"].iloc[9])  # 7th NaN (t0, >1 hr gap) — remains NaN


def test_format_generation_data():
    """Test for formatting generation data for pvnet"""
    # --- Input generation data ---
    location_ids = [101, 102]
    times = pd.date_range("2024-01-01", periods=3, freq="h")

    generation_values_kw = xr.DataArray(
        np.array([[1000, 2000, 3000], [4000, 5000, 6000]]),
        coords={"location_id": location_ids, "time_utc": times},
        dims=["location_id", "time_utc"],
    )

    generation_xr = xr.Dataset({"generation_kw": generation_values_kw})

    # --- Input metadata ---
    metadata_df = pd.DataFrame(
        {
            "location_id": [101, 102],
            "capacity_kwp": [5000, 10000],
            "latitude": [51.5, 52.0],
            "longitude": [-1.2, -1.5],
        },
    )

    # --- Run the function ---
    result = format_generation_data(generation_xr, metadata_df)

    # --- Assertions ---

    # 1. Dimensions renamed
    assert "location_id" in result.coords
    assert "time_utc" in result.coords
    assert "latitude" in result.coords
    assert "longitude" in result.coords
    assert "generation_mw" in result.data_vars
    assert "capacity_mwp" in result.data_vars

    # 2. Check coordinate values
    np.testing.assert_array_equal(result["location_id"].values, np.array([101, 102]))
    pd.testing.assert_index_equal(pd.DatetimeIndex(result["time_utc"].values), times)
    np.testing.assert_allclose(result["latitude"].values, np.array([51.5, 52.0]))
    np.testing.assert_allclose(result["longitude"].values, np.array([-1.2, -1.5]))

    # Check generation conversion (kW to MW)
    expected_mw = generation_values_kw.values / 1000.0
    np.testing.assert_allclose(result["generation_mw"].values, expected_mw)

    # 3. Capacity correctly computed and broadcast
    np.testing.assert_allclose(
        result["capacity_mwp"].sel(location_id=101).values, 5.0,
    )
    np.testing.assert_allclose(
        result["capacity_mwp"].sel(location_id=102).values, 10.0,
    )
