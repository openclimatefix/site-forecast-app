import asyncio
import importlib
import json
import sys
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pandas as pd
import pytest


@pytest.fixture()
def save_forecast_to_dataplatform_with_mocks(monkeypatch):
    """Import `site_forecast_app.save` under dp_sdk/betterproto mocks.

    This test file used to mutate `sys.modules` at import time, which polluted
    the interpreter state for subsequent tests.
    """
    # Keep a reference to the real module (if already imported) so we can restore it.
    original_save_module = sys.modules.get("site_forecast_app.save")

    # Mock betterproto
    mock_betterproto = MagicMock()
    mock_struct = MagicMock()
    mock_betterproto.lib.google.protobuf.Struct = mock_struct
    monkeypatch.setitem(sys.modules, "betterproto", mock_betterproto)
    monkeypatch.setitem(sys.modules, "betterproto.lib", mock_betterproto.lib)
    monkeypatch.setitem(sys.modules, "betterproto.lib.google", mock_betterproto.lib.google)
    monkeypatch.setitem(
        sys.modules,
        "betterproto.lib.google.protobuf",
        mock_betterproto.lib.google.protobuf,
    )

    # Mock dp_sdk
    mock_dp = MagicMock()
    mock_ocs = MagicMock()
    mock_ocs.dp = mock_dp
    mock_sdk = MagicMock()
    mock_sdk.ocf = mock_ocs
    monkeypatch.setitem(sys.modules, "dp_sdk", mock_sdk)
    monkeypatch.setitem(sys.modules, "dp_sdk.ocf", mock_ocs)
    monkeypatch.setitem(sys.modules, "dp_sdk.ocf.dp", mock_dp)

    # Force a fresh import under mocks
    monkeypatch.delitem(sys.modules, "site_forecast_app.save", raising=False)
    save_module = importlib.import_module("site_forecast_app.save")
    save_module = importlib.reload(save_module)

    yield save_module.save_forecast_to_dataplatform, mock_dp

    # Restore the original module for the rest of the test session.
    if original_save_module is not None:
        sys.modules["site_forecast_app.save"] = original_save_module
    else:
        sys.modules.pop("site_forecast_app.save", None)


# Helper to run async tests
def async_run(coro):
    return asyncio.run(coro)

def test_save_forecast_to_dataplatform_values(save_forecast_to_dataplatform_with_mocks):
    save_forecast_to_dataplatform, mock_dp = save_forecast_to_dataplatform_with_mocks
    # Setup Inputs
    init_time = datetime.now(UTC)
    forecast_df = pd.DataFrame([
        {
            "start_utc": init_time + timedelta(minutes=15),
            "forecast_power_kw": 5.0, # 5 kW
            "probabilistic_values": json.dumps({"p10": 4.0, "p90": 6.0}),
        },
        {
            "start_utc": init_time + timedelta(minutes=30),
            "forecast_power_kw": 10.0, # 10 kW
            "horizon_minutes": 30,
        },
    ])
    location_uuid = uuid4()
    model_tag = "test_tag"
    client = MagicMock()

    # Mock Async Methods
    client.list_forecasters = AsyncMock()
    client.update_forecaster = AsyncMock()
    client.create_forecaster = AsyncMock()
    client.get_location = AsyncMock()

    # Setup Forecaster response (Existing)
    mock_forecaster = MagicMock()
    mock_forecaster.forecaster_version = version("site-forecast-app")
    client.list_forecasters.return_value = MagicMock(forecasters=[mock_forecaster])

    # Setup Location response
    mock_location = MagicMock()
    mock_location.effective_capacity_watts = 10000.0 # 10 kW capacity
    client.get_location.return_value = mock_location

    # Execution
    async_run(save_forecast_to_dataplatform(
        forecast_df=forecast_df,
        location_uuid=location_uuid,
        model_tag=model_tag,
        init_time_utc=init_time,
        client=client,
    ))

    # Verifications

    # 1. Location Loaded
    client.get_location.assert_called_once()

    # 2. Forecaster Checked
    client.list_forecasters.assert_called_once()

    # 3. Validation of Forecast Values Construction
    # We can't easily inspect the 'forecast_values' variable inside the function unless we return it
    # or if we mock CreateForecastRequestForecastValue to capture calls.
    # Since we mocked dp.CreateForecastRequestForecastValue, we can check calls to it.

    assert mock_dp.CreateForecastRequestForecastValue.call_count == 2

    calls = mock_dp.CreateForecastRequestForecastValue.call_args_list

    # First row check
    # 5 kW / 10 kW capacity = 0.5 fraction
    call1_kwargs = calls[0].kwargs
    assert call1_kwargs["horizon_mins"] == 15
    assert call1_kwargs["p50_fraction"] == 0.5
    assert call1_kwargs["other_statistics_fractions"] == {"p10": 0.4, "p90": 0.6}

    # 4. Check that CreateForecastRequest was called properly
    assert mock_dp.CreateForecastRequest.called
    create_request_kwargs = mock_dp.CreateForecastRequest.call_args.kwargs

    assert create_request_kwargs["forecaster"] == mock_forecaster
    assert create_request_kwargs["location_uuid"] == str(location_uuid)
    assert create_request_kwargs["energy_source"] == mock_dp.EnergySource.SOLAR
    assert create_request_kwargs["init_time_utc"] == init_time
    # Should have 2 values
    assert len(create_request_kwargs["values"]) == 2

    # 5. Check Client Submission
    client.create_forecast.assert_called_once()
    assert client.create_forecast.call_args[0][0] == mock_dp.CreateForecastRequest.return_value

def test_zero_capacity_handling(caplog, save_forecast_to_dataplatform_with_mocks):
    save_forecast_to_dataplatform, mock_dp = save_forecast_to_dataplatform_with_mocks
    mock_dp.reset_mock()
    forecast_df = pd.DataFrame(
        [{"start_utc": datetime.now(UTC), "forecast_power_kw": 1}],
    )
    client = MagicMock()
    client.list_forecasters = AsyncMock()
    client.list_forecasters.return_value.forecasters = []
    client.create_forecaster = AsyncMock()

    # Zero Capacity
    client.get_location = AsyncMock()
    client.get_location.return_value.effective_capacity_watts = 0.0

    async_run(save_forecast_to_dataplatform(
        forecast_df=forecast_df,
        location_uuid=uuid4(),
        model_tag="tag",
        init_time_utc=datetime.now(UTC),
        client=client,
    ))

    assert "has 0 capacity, skipping" in caplog.text
    # Should not attempt to create values
    assert mock_dp.CreateForecastRequestForecastValue.call_count == 0

