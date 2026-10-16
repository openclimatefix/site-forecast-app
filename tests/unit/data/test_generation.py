import datetime as dt
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pvsite_datamodel.sqlmodels import LocationAssetType

from site_forecast_app.data.generation import (
    _get_site_generation_data,
    format_generation_data,
)


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


@pytest.mark.asyncio
async def test_get_site_generation_data_clips_negative_extrapolation(monkeypatch):
    """Quadratic extrapolation to t0 on a falling ramp should be clipped at zero"""
    monkeypatch.setenv("READ_FROM_DATA_PLATFORM", "True")

    timestamp = pd.Timestamp("2024-06-01 20:00")
    site = MagicMock()
    site.asset_type = LocationAssetType.wind
    site.ml_id = 1
    site.capacity_kw = 1000

    # Generation collapsing on an evening ramp, stopping 15 minutes before t0.
    # Without clipping, this extrapolates to about -10 kW at t0
    times = pd.date_range(
        timestamp - dt.timedelta(hours=1, minutes=15), periods=5, freq="15min", tz="UTC",
    )
    values = [100, 95, 85, 60, 10]
    dp_data = [(t.to_pydatetime(), v) for t, v in zip(times, values, strict=True)]

    with patch(
        "site_forecast_app.data.generation.fetch_generation_from_dp",
        AsyncMock(return_value=dp_data),
    ):
        result = await _get_site_generation_data(None, site, timestamp)

    generation = result["data"]["generation_kw"].sel(location_id=1)
    assert generation.sel(time_utc=timestamp).item() == 0
    assert (generation.dropna(dim="time_utc") >= 0).all()
