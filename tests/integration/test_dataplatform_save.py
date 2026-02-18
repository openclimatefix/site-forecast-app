"""Integration-style test for Data Platform save path."""
import datetime as dt
import uuid

import pandas as pd
import pytest
from betterproto.lib.google.protobuf import Struct, Value
from dp_sdk.ocf import dp

from site_forecast_app.save import save_forecast, save_forecast_to_dataplatform


@pytest.mark.integration
def test_save_forecast_triggers_dataplatform(monkeypatch, db_session, sites, forecast_values):
    """Ensure save_forecast calls the Data Platform helper when enabled."""

    calls: list[dict] = []

    async def fake_save_forecast_to_dataplatform(
        forecast_df,
        location_uuid,
        model_tag,
        init_time_utc,
        client,
        use_adjuster=True,
    ):
        calls.append(
            {
                "len": len(forecast_df),
                "location_uuid": str(location_uuid),
                "model_tag": model_tag,
                "init_time_utc": init_time_utc,
                "client_type": type(client).__name__,
                "use_adjuster": use_adjuster,
            },
        )

    class FakeChannel:
        def __init__(self, host: str, port: int):
            self.host = host
            self.port = port

        def close(self):
            return None

    class FakeDPStub:
        def __init__(self, channel):
            self.channel = channel

    monkeypatch.setenv("SAVE_TO_DATA_PLATFORM", "true")
    monkeypatch.setenv("DP_HOST", "localhost")
    monkeypatch.setenv("DP_PORT", "50051")
    monkeypatch.setattr(
        "site_forecast_app.save.save_forecast_to_dataplatform",
        fake_save_forecast_to_dataplatform,
    )
    monkeypatch.setattr("site_forecast_app.save.Channel", FakeChannel)
    monkeypatch.setattr(
        "site_forecast_app.save.dp.DataPlatformDataServiceStub",
        FakeDPStub,
    )

    site = sites[0]
    forecast = {
        "meta": {
            "location_uuid": site.location_uuid,
            "version": "0.0.0-test",
            "timestamp": dt.datetime.now(dt.UTC).replace(microsecond=0),
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

    save_forecast(
        db_session,
        forecast,
        write_to_db=False,
        ml_model_name="test-model",
        ml_model_version="0.0.0-test",
        use_adjuster=False,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["len"] == len(forecast_values["start_utc"])
    assert call["location_uuid"] == str(site.location_uuid)
    assert call["model_tag"] == "test-model"
    assert call["init_time_utc"] == forecast["meta"]["timestamp"]
    assert call["client_type"] == "FakeDPStub"
    assert call["use_adjuster"] is False


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

    class FakeDPStub:
        def __init__(self, channel):
            self.channel = channel

        async def list_locations(self, _request):
            return type("Resp", (), {"locations": []})

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

    site = sites[0]
    forecast = {
        "meta": {
            "location_uuid": site.location_uuid,
            "version": "0.0.0-test",
            "timestamp": dt.datetime.now(dt.UTC).replace(microsecond=0),
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
    legacy_uuid = uuid.uuid4()

    metadata = Struct(fields={"legacy_uuid": Value(string_value=str(legacy_uuid))})

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
        metadata=metadata,
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
    # We pass legacy_uuid. The function should map it to dp_location_uuid using the metadata.
    await save_forecast_to_dataplatform(
        forecast_df=fake_data,
        location_uuid=legacy_uuid,
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



