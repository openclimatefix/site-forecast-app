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
    forecaster_name: str = "test_model",
) -> dp.GetForecastAsTimeseriesResponse:
    """Verifies the forecast in the Data Platform."""
    async with get_dataplatform_client() as client:
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


async def setup_test_location_in_dp(
    site_name: str,
    capacity_kw: float,
    latitude: float,
    longitude: float,
) -> str:
    """Gets or creates a test location in the Data Platform, returning its UUID.

    Looks up the location by name first so that both integration tests can share
    the same module-scoped DP container without failing on duplicate name errors.
    """
    async with get_dataplatform_client() as client:
        # Return existing location UUID if already present
        list_resp = await client.list_locations(dp.ListLocationsRequest())
        for loc in list_resp.locations:
            if loc.location_name == site_name.lower():
                return loc.location_uuid

        return await create_new_location(
            client,
            site_name,
            capacity_kw,
            latitude=latitude,
            longitude=longitude,
            init_time_utc=dt.datetime(2020, 1, 1, tzinfo=dt.UTC),
            location_type=dp.LocationType.SITE,
        )


async def setup_nednl_observer() -> str:
    """Creates the 'nednl' observer in the Data Platform if it does not already exist.

    Required by make_forecaster_adjuster which calls get_week_average_deltas with
    observer_name='nednl'. A fresh DP container has no observers pre-seeded.
    """
    async with get_dataplatform_client() as client:
        # Check if it already exists to avoid duplicates
        list_resp = await client.list_observers(
            dp.ListObserversRequest(observer_names_filter=["nednl"]),
        )
        if list_resp.observers:
            return list_resp.observers[0].observer_uuid

        create_resp = await client.create_observer(
            dp.CreateObserverRequest(name="nednl"),
        )
        return create_resp.observer_uuid


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

    dp_location_uuid = asyncio.run(
        setup_test_location_in_dp(site_name, site.capacity_kw, site.latitude, site.longitude),
    )

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


@pytest.mark.integration
def test_save_adjuster_forecast_to_dataplatform(
    monkeypatch, db_session, sites, forecast_values, dp_address,
):
    """Verify that an adjusted forecast is saved to the Data Platform alongside the base forecast.

    This test:
    1. Spins up real Docker containers (PostgreSQL + DataPlatform) via the ``dp_address`` fixture.
    2. Creates a location in the DP.
    3. Saves a base forecast with ``use_adjuster=True`` via ``save_forecast``.
    4. Queries the DP for both the base (``test-model``) and adjuster (``test-model_adjust``)
       forecasters and asserts that forecasts were stored for each.

    The adjuster call to ``make_forecaster_adjuster`` uses ``GetWeekAverageDeltasRequest``.
    Because there is no historical data in the fresh container, the deltas will be empty and
    all adjustments will be zero — but the ``create_forecast`` request must still be submitted
    and the forecaster must exist in the DP.
    """
    host, port = dp_address
    monkeypatch.setenv("SAVE_TO_DATA_PLATFORM", "true")
    monkeypatch.setenv("DATA_PLATFORM_HOST", host)
    monkeypatch.setenv("DATA_PLATFORM_PORT", str(port))

    # 1. Create a Location in DP
    site = sites[0]
    site_name = site.client_location_name or "test_adjuster_site"

    dp_location_uuid = asyncio.run(
        setup_test_location_in_dp(site_name, site.capacity_kw, site.latitude, site.longitude),
    )

    # 1b. Create the 'nednl' observer (required by the adjuster's delta query)
    asyncio.run(setup_nednl_observer())

    # 2. Prepare forecast data (single slot for simplicity)
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

    # 3. Save forecast with use_adjuster=True so both base + adjusted are pushed to DP
    save_forecast(
        db_session,
        forecast,
        write_to_db=False,
        ml_model_name="test-model",
        ml_model_version="0.0.0-test",
        use_adjuster_database=True,
    )

    # 4. Verify base forecast exists in DP
    base_resp = asyncio.run(
        verify_forecast_in_dp(
            dp_location_uuid,
            init_time,
            forecast["values"][-1]["end_utc"],
            forecaster_name="test_model",
        ),
    )
    assert len(base_resp.values) == len(forecast["values"]), (
        "Base forecast values count mismatch in Data Platform"
    )

    # 5. Verify adjuster forecast exists in DP under the _adjust forecaster
    adjuster_resp = asyncio.run(
        verify_forecast_in_dp(
            dp_location_uuid,
            init_time,
            forecast["values"][-1]["end_utc"],
            forecaster_name="test_model_adjust",
        ),
    )
    assert len(adjuster_resp.values) == len(forecast["values"]), (
        "Adjuster forecast values count mismatch in Data Platform"
    )
