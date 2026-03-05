"""Integration-style test for Data Platform save path."""
import asyncio
import datetime as dt

import pytest
from dp_sdk.ocf import dp

from site_forecast_app.save import save_forecast
from site_forecast_app.save.data_platform import (
    create_new_location,
    get_dataplatform_client,
)


async def verify_forecast_in_dp(
    location_uuid: str,
    init_time: dt.datetime,
    forecast_end_time: dt.datetime,
) -> dp.GetForecastAsTimeseriesResponse:
    """Verifies the forecast in the Data Platform."""
    async with get_dataplatform_client() as client:
        forecaster_name = "test_model"
        list_forecasters_request = dp.ListForecastersRequest(
            forecaster_names_filter=[forecaster_name],
        )
        list_forecasters_response = await client.list_forecasters(list_forecasters_request)
        assert len(list_forecasters_response.forecasters) > 0
        forecaster = list_forecasters_response.forecasters[0]

        query_start = init_time - dt.timedelta(minutes=1)
        query_end = forecast_end_time + dt.timedelta(minutes=1)

        get_forecast_request = dp.GetForecastAsTimeseriesRequest(
            location_uuid=location_uuid,
            energy_source=dp.EnergySource.SOLAR,
            time_window=dp.TimeWindow(
                start_timestamp_utc=query_start,
                end_timestamp_utc=query_end,
            ),
            forecaster=forecaster,
        )

        return await client.get_forecast_as_timeseries(get_forecast_request)


async def setup_test_location_in_dp(site_name: str, capacity_kw: float) -> str:
    """Sets up a test location in the Data Platform."""
    async with get_dataplatform_client() as client:
        return await create_new_location(
            client,
            site_name,
            capacity_kw,
            latitude=0.0,
            longitude=0.0,
            init_time_utc=dt.datetime(2020, 1, 1, tzinfo=dt.UTC),
            location_type=dp.LocationType.SITE,
        )


@pytest.mark.integration
def test_save_forecast_integration(
    monkeypatch, db_session, sites, forecast_values, dp_address,
):
    """test for end-to-end Data Platform save flow."""
    host, port = dp_address
    monkeypatch.setenv("SAVE_TO_DATA_PLATFORM", "true")
    monkeypatch.setenv("DATA_PLATFORM_HOST", host)
    monkeypatch.setenv("DATA_PLATFORM_PORT", str(port))

    # 1. Create a Location in DP
    site = sites[0]
    site_name = site.client_location_name or "test_integration_site"

    dp_location_uuid = asyncio.run(setup_test_location_in_dp(site_name, site.capacity_kw))

    # 2. Prepare forecast data
    init_time = forecast_values["start_utc"][0]
    forecast = {
        "meta": {
            "location_uuid": site.location_uuid,
            "version": "0.0.0-test",
            "timestamp": init_time,
            "client_location_name": site_name,
            "capacity_kw": site.capacity_kw,
        },
        "values": [
            {
                "start_utc": s,
                "end_utc": e,
                "forecast_power_kw": p,
            }
            for s, e, p in zip(
                forecast_values["start_utc"],
                forecast_values["end_utc"],
                forecast_values["forecast_power_kw"],
                strict=False,
            )
        ],
    }

    # 3. Save forecast (orchestrates DB + DP)
    save_forecast(
        db_session,
        forecast,
        write_to_db=False,
        ml_model_name="test-model",
        ml_model_version="0.0.0-test",
    )

    # 4. Verify in DP
    forecast_resp = asyncio.run(
        verify_forecast_in_dp(
            dp_location_uuid,
            init_time,
            forecast["values"][-1]["end_utc"],
        ),
    )
    assert len(forecast_resp.values) == len(forecast["values"])



