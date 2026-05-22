
import numpy as np
import pandas as pd
import xarray as xr

from site_forecast_app.data.generation import (
    format_generation_data,
)


def test_format_generation_data():
    """Test for formatting generation data for pvnet"""
    # --- Input generation data ---
    location_ids = [101, 102]
    times = pd.date_range("2024-01-01", periods=3, freq="H")

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
    assert "latitude" in result.coords
    assert "longitude" in result.coords
    assert "generation_mw" in result.data_vars
    assert "capacity_mwp" in result.data_vars

    # 3. Capacity correctly computed and broadcast
    expected_capacity = (metadata_df["capacity_kwp"] / 1000).values
    # broadcast expected (shape: location_id x time)
    expected_capacity_broadcast = np.vstack([expected_capacity]).T.repeat(3, axis=1)
    np.testing.assert_allclose(
        result["capacity_mwp"].values, expected_capacity_broadcast,
    )

