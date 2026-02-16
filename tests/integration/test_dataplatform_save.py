"""Integration-style test for Data Platform save path."""

import datetime as dt

import pytest

from site_forecast_app.save import save_forecast


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
    ):
        calls.append(
            {
                "len": len(forecast_df),
                "location_uuid": str(location_uuid),
                "model_tag": model_tag,
                "init_time_utc": init_time_utc,
                "client_type": type(client).__name__,
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
    monkeypatch.setattr("site_forecast_app.save.save_forecast_to_dataplatform", fake_save_forecast_to_dataplatform)
    monkeypatch.setattr("site_forecast_app.save.Channel", FakeChannel)
    monkeypatch.setattr("site_forecast_app.save.dp.DataPlatformDataServiceStub", FakeDPStub)

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
