"""Integration-style test for Data Platform save path."""
import datetime as dt

import pandas as pd
import pytest
from dp_sdk.ocf import dp

from site_forecast_app.save import save_forecast, save_forecast_to_dataplatform


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="module")
async def test_save_forecast_triggers_dataplatform(
    monkeypatch, db_session, sites, forecast_values, client,
):
    """Ensure save_forecast calls the Data Platform helper when enabled."""

    monkeypatch.setenv("SAVE_TO_DATA_PLATFORM", "true")
    # Point DP_HOST/PORT to the testcontainer started by the client fixture
    # Access private attributes as grpclib Channel doesn't expose them publicly
    monkeypatch.setenv("DP_HOST", client.channel._host)
    monkeypatch.setenv("DP_PORT", str(client.channel._port))

    # 1. Create a Location in DP so save_forecast can find it by name
    site = sites[0]
    # Ensure site has a name we can use
    site_name = site.client_location_name or "test_site_name_fixture"

    create_location_request = dp.CreateLocationRequest(
        location_name=site_name,
        energy_source=dp.EnergySource.SOLAR,
        geometry_wkt="POINT(0 0)",
        location_type=dp.LocationType.SITE,
        effective_capacity_watts=int(site.capacity_kw * 1000),
        valid_from_utc=dt.datetime(2020, 1, 1, tzinfo=dt.UTC),
    )
    resp = await client.create_location(create_location_request)
    dp_location_uuid = resp.location_uuid

    forecast = {
        "meta": {
            "location_uuid": site.location_uuid,
            "version": "0.0.0-test",
            "timestamp": forecast_values["start_utc"][0],
            "client_location_name": site_name,
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

    # Monkeypatch asyncio.run to capture tasks generated inside save_forecast
    # This prevents RuntimeError because asyncio.run() cannot be called when an
    # event loop is running.
    import asyncio
    captured_tasks = []

    def fake_run(coro):
        task = asyncio.create_task(coro)
        captured_tasks.append(task)
        return None

    monkeypatch.setattr("site_forecast_app.save.asyncio.run", fake_run)

    save_forecast(
        db_session,
        forecast,
        write_to_db=False,
        ml_model_name="test-model",
        ml_model_version="0.0.0-test",
        use_adjuster=False,
    )

    # Allow async tasks created by save_forecast to complete
    if captured_tasks:
        await asyncio.gather(*captured_tasks)

    # Verify that the forecast was actually created in the DP
    # We can check by listing forecasters or getting the forecast

    forecaster_name = "test_model" # save_forecast replaces '-' with '_'
    list_forecasters_request = dp.ListForecastersRequest(
        forecaster_names_filter=[forecaster_name],
    )
    list_forecasters_response = await client.list_forecasters(list_forecasters_request)

    assert len(list_forecasters_response.forecasters) > 0, "Forecaster not created in DP"
    forecaster = list_forecasters_response.forecasters[0]

    # Check for forecast values
    # We look for the first timestamp
    start_ts = forecast["values"][0]["start_utc"]
    end_ts = forecast["values"][-1]["end_utc"]

    # Widen the window slightly to avoid boundary issues
    query_start = start_ts - dt.timedelta(minutes=1)
    query_end = end_ts + dt.timedelta(minutes=1)

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
    assert len(forecast_resp.values) == len(forecast["values"]), (
        f"Forecast values count mismatch in DP. Expected {len(forecast['values'])}, "
        f"got {len(forecast_resp.values)}. Response: {forecast_resp.values}"
    )


@pytest.mark.integration
def test_save_forecast_sends_adjusted_forecast(monkeypatch, db_session, sites, forecast_values):
    """Ensure adjusted forecast also hits DP via create_forecast."""

    create_calls: list = []
    adjust_calls: list[dict] = []

    async def fake_make_forecaster_adjuster(
        client,
        location_uuid,
        init_time_utc,
        forecast_values,
        model_tag,
        forecaster,
    ):
        adjust_calls.append(
            {
                "client": client,
                "location_uuid": location_uuid,
                "init_time_utc": init_time_utc,
                "forecast_values": forecast_values,
                "model_tag": model_tag,
                "forecaster": forecaster,
            },
        )
        return "adjusted-request"

    class FakeChannel:
        def __init__(self, host: str, port: int):
            self.host = host
            self.port = port

        def close(self):
            return None

    site = sites[0]
    site_name = site.client_location_name or "test_site_unit"
    fake_dp_uuid = "fake-dp-location-uuid-1234"

    class FakeLocation:
        location_name = site_name
        location_uuid = fake_dp_uuid

    class FakeDPStub:
        def __init__(self, channel):
            self.channel = channel

        async def list_locations(self, _request):
            # Return the site already seeded so _resolve_target_uuid finds it
            return type("Resp", (), {"locations": [FakeLocation()]})

        async def list_forecasters(self, _request):
            return type("Resp", (), {"forecasters": []})

        async def create_forecaster(self, request):
            forecaster = type("Forecaster", (), {"forecaster_version": request.version})
            return type("Resp", (), {"forecaster": forecaster})

        async def update_forecaster(self, request):
            forecaster = type("Forecaster", (), {"forecaster_version": request.new_version})
            return type("Resp", (), {"forecaster": forecaster})

        async def get_location(self, _request):
            return type("Location", (), {"effective_capacity_watts": 1_000_000})

        async def create_location(self):
            return type("Resp", (), {"location_uuid": fake_dp_uuid})

        async def create_forecast(self, request):
            create_calls.append(request)
            return None

    monkeypatch.setenv("SAVE_TO_DATA_PLATFORM", "true")
    monkeypatch.setenv("DP_HOST", "localhost")
    monkeypatch.setenv("DP_PORT", "50051")
    monkeypatch.setattr("site_forecast_app.save.Channel", FakeChannel)
    monkeypatch.setattr("site_forecast_app.save.dp.DataPlatformDataServiceStub", FakeDPStub)
    monkeypatch.setattr(
        "site_forecast_app.save._make_forecaster_adjuster",
        fake_make_forecaster_adjuster,
    )

    forecast = {
        "meta": {
            "location_uuid": site.location_uuid,
            "version": "0.0.0-test",
            "timestamp": forecast_values["start_utc"][0],
            "client_location_name": site_name,
        },
        "values": [
            {
                "start_utc": forecast_values["start_utc"][0],
                "end_utc": forecast_values["end_utc"][0],
                "forecast_power_kw": forecast_values["forecast_power_kw"][0],
            },
        ],
    }

    save_forecast(
        db_session,
        forecast,
        write_to_db=False,
        ml_model_name="test-model",
        ml_model_version="0.0.0-test",
        use_adjuster=True,
    )

    # Expect base + adjusted calls
    assert len(create_calls) == 2
    assert "adjusted-request" in create_calls
    assert len(adjust_calls) == 1


@pytest.mark.asyncio(loop_scope="module")
async def test_save_forecast_to_dataplatform_integration(client):
    """
    Test saving data to the Data Platform (Integration).
    """

    # 1. Create a Location in DP
    # We need a location UUID. Let's generate one.

    # Try creating with Site type, fallback to something else if fails
    # assuming LocationType.Site exists or is valid.
    # If not, user might need to adjust. But let's assume valid based on usage in other tests.
    # Solar consumer used LocationType.GSP.
    create_location_request = dp.CreateLocationRequest(
        location_name="test_site_integration",
        energy_source=dp.EnergySource.SOLAR,
        geometry_wkt="POINT(0 0)",
        location_type=dp.LocationType.SITE,
        effective_capacity_watts=10_000,
        valid_from_utc=dt.datetime(2020, 1, 1, tzinfo=dt.UTC),
    )

    location_response = await client.create_location(create_location_request)
    dp_location_uuid = location_response.location_uuid

    # 2. Prepare fake forecast data
    init_time = dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.UTC)

    fake_data = pd.DataFrame(
        {
            "start_utc": [
                init_time + dt.timedelta(minutes=15),
                init_time + dt.timedelta(minutes=30),
            ],
            "end_utc": [
                init_time + dt.timedelta(minutes=30),
                init_time + dt.timedelta(minutes=45),
            ],
            "forecast_power_kw": [5.0, 8.0],  # 5 kW and 8 kW
            "horizon_minutes": [15, 30],
            # probabilistic_values if needed
        },
    )

    # 3. Call save_forecast_to_dataplatform
    # We pass client_location_name="test_site_integration".
    # The function should map it to dp_location_uuid.
    await save_forecast_to_dataplatform(
        forecast_df=fake_data,
        client_location_name="test_site_integration",
        model_tag="test-integration-mn",
        init_time_utc=init_time,
        client=client,
        use_adjuster=False,
    )

    # 4. Verify data in DP using GetForecast or ListForecasts
    # We need to find the forecaster first
    forecaster_name = "test_integration_mn"
    list_forecasters_request = dp.ListForecastersRequest(
        forecaster_names_filter=[forecaster_name],
    )
    list_forecasters_response = await client.list_forecasters(list_forecasters_request)

    assert len(list_forecasters_response.forecasters) > 0
    forecaster = list_forecasters_response.forecasters[0]

    # Get Forecast values
    start_ts_dp = init_time + dt.timedelta(minutes=15)
    end_ts_dp = init_time + dt.timedelta(minutes=45)

    get_forecast_request = dp.GetForecastAsTimeseriesRequest(
        location_uuid=dp_location_uuid,
        energy_source=dp.EnergySource.SOLAR,
        time_window=dp.TimeWindow(
            start_timestamp_utc=start_ts_dp,
            end_timestamp_utc=end_ts_dp,
        ),
        forecaster=forecaster,
        # This filter might reduce result to 1 if we only want horizon 15.
        horizon_mins=15,
    )
    # Removing horizon_mins filter or explicitly checking for specific one
    # If I remove it, I should get both. But GetForecastAsTimeseriesRequest
    # definition showed optional horizon_mins.
    # Let's remove horizon_mins filter to get both.

    get_forecast_request = dp.GetForecastAsTimeseriesRequest(
        location_uuid=dp_location_uuid,
        energy_source=dp.EnergySource.SOLAR,
        time_window=dp.TimeWindow(
            start_timestamp_utc=start_ts_dp,
            end_timestamp_utc=end_ts_dp,
        ),
        forecaster=forecaster,
    )

    forecast_resp = await client.get_forecast_as_timeseries(get_forecast_request)
    assert len(forecast_resp.values) == 2

    # Check values
    # value is p50_fraction = (kW * 1000) / capacity_watts
    # capacity = 10,000 W.
    # 5 kW = 0.5 fraction (horizon 15)
    # 8 kW = 0.8 fraction (horizon 30)

    # Values might not be sorted by time, so we convert to list and sort or check by existence
    fractions = sorted([v.p50_value_fraction for v in forecast_resp.values])
    assert abs(fractions[0] - 0.5) < 1e-6
    assert abs(fractions[1] - 0.8) < 1e-6



