import datetime as dt

import pandas as pd
from pvsite_datamodel.sqlmodels import LocationAssetType

from site_forecast_app.data.generation import filter_on_sun_elevation, get_generation_data


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

    # Check for 5 (non-null) generation values
    assert gen_data.generation_mw.shape == (1, 193)

    # Check first and last timestamps are correct
    assert gen_data.time_utc[0] == init_timestamp - dt.timedelta(hours=48)
    assert gen_data.time_utc[-1] == init_timestamp

    # Check for expected metadata
    assert len(gen_data.location_id.values) == 1
    assert len(gen_data.latitude.values) == 1
    assert len(gen_data.longitude.values) == 1
