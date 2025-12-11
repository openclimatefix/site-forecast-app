import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr
from pvsite_datamodel.sqlmodels import LocationAssetType

from site_forecast_app.data.generation import (
    filter_on_sun_elevation,
    format_generation_data,
    get_generation_data,
)


def test_filter_on_sun_elevation(sites):
    """Test for filtering generation data based on sun elevation"""

    site = sites[0]
    generation_df = pd.DataFrame(
        data=[
            ["2023-10-01", 0.0],
            ["2023-10-01 10:00", 0.0],  # this one will get removed
            ["2023-10-01 11:00", 1.0],
            ["2023-10-01 20:00", 0.0],
        ],
        columns=["time_utc", "1"],
    )
    generation_df.set_index("time_utc", inplace=True)

    filter_generation_df = filter_on_sun_elevation(generation_df=generation_df, site=site)
    assert len(filter_generation_df) == 3
    assert filter_generation_df.index[0] == "2023-10-01"
    assert filter_generation_df.index[1] == "2023-10-01 11:00"
    assert filter_generation_df.index[2] == "2023-10-01 20:00"


def test_get_generation_data_pv(db_session, sites, generation_db_values, init_timestamp):  # noqa: ARG001
    """Test for correct generation data"""

    # Test only checks for wind data as solar data not ready yet
    gen_sites = [s for s in sites if s.asset_type == LocationAssetType.pv][0:1]  # 1 site
    gen_data = get_generation_data(db_session, gen_sites, timestamp=init_timestamp)
    gen_xr, gen_meta = gen_data["data"], gen_data["metadata"]

    # Check for 5 (non-null) generation values
    assert gen_xr.generation_kw.shape == (1, 193)

    # Check first and last timestamps are correct
    assert gen_xr.time_utc[0] == init_timestamp - dt.timedelta(hours=48)
    assert gen_xr.time_utc[-1] == init_timestamp

    # Check for expected metadata
    assert len(gen_meta) == 1

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
