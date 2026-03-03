"""Integration-style test for Data Platform save path."""
import asyncio
import datetime as dt

import pytest
from dp_sdk.ocf import dp
from grpclib.client import Channel

from site_forecast_app.save import save_forecast


@pytest.mark.integration
def test_save_forecast_integration(
    monkeypatch, db_session, sites, forecast_values, dp_address,
):
    """Combined integration test for end-to-end Data Platform save flow."""
    host, port = dp_address
    monkeypatch.setenv("SAVE_TO_DATA_PLATFORM", "true")
    monkeypatch.setenv("DP_HOST", host)
    monkeypatch.setenv("DP_PORT", str(port))

    # 1. Create a Location in DP
    site = sites[0]
    site_name = site.client_location_name or "test_integration_site"

    create_location_request = dp.CreateLocationRequest(
        location_name=site_name,
        energy_source=dp.EnergySource.SOLAR,
        geometry_wkt="POINT(0 0)",
        location_type=dp.LocationType.SITE,
        effective_capacity_watts=int(site.capacity_kw * 1000),
        valid_from_utc=dt.datetime(2020, 1, 1, tzinfo=dt.UTC),
    )

    async def _create_loc():
        channel = Channel(host=host, port=port)
        try:
            client = dp.DataPlatformDataServiceStub(channel)
            return await client.create_location(create_location_request)
        finally:
            channel.close()

    resp = asyncio.run(_create_loc())
    dp_location_uuid = resp.location_uuid

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
    async def _verify():
        channel = Channel(host=host, port=port)
        try:
            client = dp.DataPlatformDataServiceStub(channel)
            forecaster_name = "test_model"
            list_forecasters_request = dp.ListForecastersRequest(
                forecaster_names_filter=[forecaster_name],
            )
            list_forecasters_response = await client.list_forecasters(list_forecasters_request)
            assert len(list_forecasters_response.forecasters) > 0
            forecaster = list_forecasters_response.forecasters[0]

            query_start = init_time - dt.timedelta(minutes=1)
            query_end = forecast["values"][-1]["end_utc"] + dt.timedelta(minutes=1)

            get_forecast_request = dp.GetForecastAsTimeseriesRequest(
                location_uuid=dp_location_uuid,
                energy_source=dp.EnergySource.SOLAR,
                time_window=dp.TimeWindow(
                    start_timestamp_utc=query_start,
                    end_timestamp_utc=query_end,
                ),
                forecaster=forecaster,
            )

            forecast_resp = await client.get_forecast_as_timeseries(get_forecast_request)
            return forecast_resp
        finally:
            channel.close()

    forecast_resp = asyncio.run(_verify())
    assert len(forecast_resp.values) == len(forecast["values"])



