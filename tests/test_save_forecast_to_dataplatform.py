import asyncio
import logging
from datetime import UTC, datetime
from importlib.metadata import version
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pandas as pd
from dp_sdk.ocf import dp
from dp_sdk.ocf.dp import EnergySource

from site_forecast_app.save import save_forecast_to_dataplatform


def test_save_forecast_to_dataplatform(caplog):
    caplog.set_level(logging.INFO)

    # Mock inputs
    forecast_df = pd.DataFrame({"test": [1, 2, 3]})
    location_uuid = uuid4()
    model_tag = "test_model"
    init_time_utc = datetime.now(UTC)
    client = MagicMock()

    # Mock Async methods on client
    client.list_forecasters = AsyncMock()
    client.update_forecaster = AsyncMock()
    client.create_forecaster = AsyncMock()
    client.get_location = AsyncMock()

    # Setup list_forecasters return value
    mock_forecaster = MagicMock()
    mock_forecaster.forecaster_version = version("site-forecast-app")

    list_response = MagicMock()
    list_response.forecasters = [mock_forecaster]
    client.list_forecasters.return_value = list_response

    # Call the function
    asyncio.run(save_forecast_to_dataplatform(
        forecast_df=forecast_df,
        location_uuid=location_uuid,
        model_tag=model_tag,
        init_time_utc=init_time_utc,
        client=client,
    ))

    # Assert logs
    assert "Writing to data platform" in caplog.text

    # Assert client calls
    client.list_forecasters.assert_called_once()

    # Check that GetLocationRequest was constructed correctly
    # Note: mocking libraries handles class construction tracking differently
    # depending on setup. But since dp is imported from the module, and we
    # mock the CLIENT methods which receive the request object... We can
    # inspect the arguments passed to client.get_location

    client.get_location.assert_called_once()
    call_args = client.get_location.call_args
    request_arg = call_args[0][0]

    # Validating the request properties
    assert request_arg.location_uuid == str(location_uuid)
    assert request_arg.energy_source == dp.EnergySource.SOLAR
    assert request_arg.include_geometry is False

def test_save_forecast_to_dataplatform_create_new(caplog):
    caplog.set_level(logging.INFO)

    # Mock inputs
    forecast_df = pd.DataFrame({"test": [1, 2, 3]})
    location_uuid = uuid4()
    model_tag = "test_model"
    init_time_utc = datetime.now(UTC)
    client = MagicMock()

    # Mock Async methods
    client.list_forecasters = AsyncMock()
    client.create_forecaster = AsyncMock()
    client.get_location = AsyncMock()

    # Setup list_forecasters to return empty
    list_response = MagicMock()
    list_response.forecasters = []
    client.list_forecasters.return_value = list_response

    # Setup create_forecaster return
    create_response = MagicMock()
    create_response.forecaster = MagicMock()
    client.create_forecaster.return_value = create_response

    # Call the function
    asyncio.run(save_forecast_to_dataplatform(
        forecast_df=forecast_df,
        location_uuid=location_uuid,
        model_tag=model_tag,
        init_time_utc=init_time_utc,
        client=client,
    ))

    # Assert create_forecaster was called
    client.create_forecaster.assert_called_once()


def test_save_forecast_to_dataplatform_exception(caplog):
    # Test exception handling
    client = MagicMock()
    client.list_forecasters = AsyncMock(side_effect=Exception("Data Platform Error"))

    asyncio.run(save_forecast_to_dataplatform(
        forecast_df=pd.DataFrame(),
        location_uuid=uuid4(),
        model_tag="model",
        init_time_utc=datetime.now(UTC),
        client=client,
    ))

    assert "Failed to save forecast to data platform" in caplog.text
