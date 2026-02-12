import datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pandas as pd
import pytest
from betterproto.lib.google.protobuf import Struct, Value
from dp_sdk.ocf import dp
from dp_sdk.ocf.dp import EnergySource

# Import the module under test - adjusting imports if necessary
from site_forecast_app.save import save_forecast_to_dataplatform


class MockClient:
    def __init__(self):
        self.create_forecaster = AsyncMock()
        self.list_forecasters = AsyncMock()
        self.update_forecaster = AsyncMock()
        self.list_locations = AsyncMock()
        self.get_location = AsyncMock()
        self.create_forecast = AsyncMock()
        self.get_week_average_deltas = AsyncMock()

@pytest.fixture
def mock_client():
    return MockClient()

@pytest.mark.asyncio
async def test_save_forecast_logic_legacy_uuid_mapping(mock_client):
    # Setup
    legacy_uuid = str(uuid4())
    dp_uuid = str(uuid4())

    # 1. Mock list_locations to return a match for legacy_uuid
    mock_location = dp.ListLocationsResponseLocationSummary(
        location_uuid=dp_uuid,
        metadata=Struct(fields={"legacy_uuid": Value(string_value=legacy_uuid)}),
    )
    mock_client.list_locations.return_value = dp.ListLocationsResponse(locations=[mock_location])

    # 2. Mock get_location to return capacity
    mock_client.get_location.return_value = dp.GetLocationResponse(
        location_uuid=dp_uuid,
        effective_capacity_watts=10000,
    )

    # 3. Mock forecaster existence
    mock_client.list_forecasters.return_value = dp.ListForecastersResponse(forecasters=[
        dp.Forecaster(forecaster_name="test_model", forecaster_version="0.0.1"),
    ])

    # 4. Prepare Data
    init_time_utc = datetime.datetime.now(datetime.UTC)
    forecast_df = pd.DataFrame([{
        "start_utc": init_time_utc + datetime.timedelta(minutes=60),
        "forecast_power_kw": 5.0, # 50%
        "horizon_minutes": 60,
    }])

    # 5. Run
    # Pass the legacy_uuid (as a string or UUID object) to the function
    await save_forecast_to_dataplatform(
        forecast_df=forecast_df,
        location_uuid=legacy_uuid, # Passed as legacy
        model_tag="test-model",
        init_time_utc=init_time_utc,
        client=mock_client,
    )

    # 6. Verify
    # Expect 2 calls: one for original, one for adjusted
    assert mock_client.create_forecast.call_count >= 1

    # Check the first call (original forecast)
    first_call_args = mock_client.create_forecast.call_args_list[0][0][0]
    assert first_call_args.location_uuid == dp_uuid

@pytest.mark.asyncio
async def test_save_forecast_logic_timezone_resilience(mock_client):
    # Setup
    location_uuid = str(uuid4())
    dp_uuid = location_uuid # Assuming direct match for this test

    # Mock list_locations (no legacy match needed)
    mock_client.list_locations.return_value = dp.ListLocationsResponse(locations=[])

    # Mock get_location
    mock_client.get_location.return_value = dp.GetLocationResponse(
        location_uuid=dp_uuid,
        effective_capacity_watts=10000,
    )

     # Mock forecaster
    mock_client.list_forecasters.return_value = dp.ListForecastersResponse(forecasters=[
        dp.Forecaster(forecaster_name="test_model", forecaster_version="0.0.1"),
    ])

    # Prepare Data with Naive Timestamps
    # init_time_utc is aware
    init_time_utc = datetime.datetime.now(datetime.UTC)

    # start_utc is naive
    naive_dt = (init_time_utc + datetime.timedelta(minutes=30)).replace(tzinfo=None)

    forecast_df = pd.DataFrame([{
        "start_utc": naive_dt,
        "forecast_power_kw": 8.0,
        # Missing horizon_minutes, forcing calculation
    }])

    # Run
    await save_forecast_to_dataplatform(
        forecast_df=forecast_df,
        location_uuid=location_uuid,
        model_tag="test-model",
        init_time_utc=init_time_utc,
        client=mock_client,
    )

    # Verify
    assert mock_client.create_forecast.call_count >= 1

    first_call_args = mock_client.create_forecast.call_args_list[0][0][0]

    # Check horizon calculation
    # 30 mins difference
    assert first_call_args.values[0].horizon_mins == 30
    assert first_call_args.values[0].p50_fraction == 0.8

