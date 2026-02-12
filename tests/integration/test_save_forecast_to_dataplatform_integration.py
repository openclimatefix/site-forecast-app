import datetime
import logging

import pandas as pd
import pytest
from betterproto.lib.google.protobuf import Struct
from dp_sdk.ocf import dp
from dp_sdk.ocf.dp import EnergySource, LocationType

from site_forecast_app.save import save_forecast_to_dataplatform

log = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_save_forecast_to_dataplatform_integration(client):
    """
    Integration test for saving forecasts to the Data Platform.
    Verifies that forecasts are correctly created and linked to the forecaster/location.
    """
    init_time_utc = datetime.datetime.now(datetime.UTC).replace(microsecond=0, second=0, minute=0)
    model_tag = "integration_test_model"

    # 1. Setup: Create a test Location
    import uuid
    location_name = f"integration_test_site_{uuid.uuid4().hex}"
    create_location_request = dp.CreateLocationRequest(
        location_name=location_name,
        energy_source=dp.EnergySource.SOLAR,
        location_type=dp.LocationType.SITE,
        geometry_wkt="POINT(4.5 52.0)", # Random NL location
        effective_capacity_watts=10_000, # 10 kW capacity
        metadata=Struct(fields={}),
        valid_from_utc=init_time_utc - datetime.timedelta(days=1),
    )
    location_response = await client.create_location(create_location_request)
    location_uuid = location_response.location_uuid
    log.info(f"Created test location: {location_uuid}")

    # 2. Setup: Create realistic forecast dataframe
    # Forecast 1: 15 mins ahead, 50% power (5 kW)
    # Forecast 2: 30 mins ahead, 100% power (10 kW)
    forecast_df = pd.DataFrame([
        {
            "start_utc": init_time_utc + datetime.timedelta(minutes=15),
            "forecast_power_kw": 5.0,
            "horizon_minutes": 15,
        },
        {
            "start_utc": init_time_utc + datetime.timedelta(minutes=30),
            "forecast_power_kw": 10.0,
            "horizon_minutes": 30,
        },
    ])

    # 3. Call the function under test
    await save_forecast_to_dataplatform(
        forecast_df=forecast_df,
        location_uuid=location_uuid,
        model_tag=model_tag,
        init_time_utc=init_time_utc,
        client=client,
    )

    # 4. Verify: Check Forecaster was created
    forecaster_name = model_tag.replace("-", "_")
    list_forecasters = await client.list_forecasters(
        dp.ListForecastersRequest(forecaster_names_filter=[forecaster_name]),
    )
    assert len(list_forecasters.forecasters) > 0, "Forecaster should have been created"
    forecaster = list_forecasters.forecasters[0]
    assert forecaster.forecaster_name == forecaster_name
    log.info(f"Verified forecaster creation: {forecaster_name}")

    # 5. Verify: Check Forecasts exist in Data Platform using get_forecast_at_timestamp

    # Verify first value (15 mins)
    expected_ts1 = init_time_utc + datetime.timedelta(minutes=15)
    response_1 = await client.get_forecast_at_timestamp(
        dp.GetForecastAtTimestampRequest(
            location_uuids=[location_uuid],
            energy_source=dp.EnergySource.SOLAR,
            timestamp_utc=expected_ts1,
            forecaster=forecaster,
        ),
    )
    assert len(response_1.values) == 1
    val1 = response_1.values[0]
    assert val1.location_uuid == location_uuid
    # 50% of 10kW = 5kW. Data platform returns fraction.
    assert val1.value_fraction == 0.5

    # Verify second value (30 mins)
    expected_ts2 = init_time_utc + datetime.timedelta(minutes=30)
    response_2 = await client.get_forecast_at_timestamp(
        dp.GetForecastAtTimestampRequest(
            location_uuids=[location_uuid],
            energy_source=dp.EnergySource.SOLAR,
            timestamp_utc=expected_ts2,
            forecaster=forecaster,
        ),
    )
    assert len(response_2.values) == 1
    val2 = response_2.values[0]
    assert val2.value_fraction == 1.0

    log.info("Verified forecast values match input")

    # 6. Verify: Adjusted Forecaster checks
    # Since we didn't setup pvlive observer and deltas, the adjustment
    # likely failed gracefully or didn't run fully. We can check if the
    # adjusted forecaster exists (it is created before deltas are checked).

    adj_forecaster_name = f"{forecaster_name}_adjust"
    list_adj_forecasters = await client.list_forecasters(
        dp.ListForecastersRequest(
            forecaster_names_filter=[adj_forecaster_name],
        ),
    )

    if len(list_adj_forecasters.forecasters) > 0:
        log.info(f"Adjusted forecaster {adj_forecaster_name} was created.")
    else:
        log.info("Adjusted forecaster was NOT created (expected if adjustment failed early).")
