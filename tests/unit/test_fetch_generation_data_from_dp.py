import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from site_forecast_app.save.data_platform import fetch_generation_from_dp


@pytest.mark.asyncio
async def test_fetch_generation_from_dp_success():
    """Test fetching generation data from the Data Platform."""
    mock_client = AsyncMock()

    # Mock get_observations_as_timeseries
    mock_obs_val1 = MagicMock()
    # Mock protobuf Timestamp with ToDatetime(tzinfo=UTC)
    mock_ts = MagicMock()
    mock_ts.ToDatetime.return_value = datetime(2024, 6, 1, 10, 0, tzinfo=UTC)
    mock_obs_val1.timestamp_utc = mock_ts
    mock_obs_val1.value_fraction = 0.5

    mock_obs_resp = MagicMock()
    mock_obs_resp.values = [mock_obs_val1]
    mock_obs_resp.effective_capacity_watts = 2000  # 2 kW
    mock_client.get_observations_as_timeseries.return_value = mock_obs_resp

    # Mock get_dataplatform_client context manager
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_client

    with (
        patch(
            "site_forecast_app.save.data_platform.get_dataplatform_client",
            return_value=mock_ctx,
        ),
        patch(
            "site_forecast_app.save.data_platform.fetch_dp_location_map",
            return_value={"test-site": "uuid-123"},
        ),
    ):
        os.environ["OBSERVER_NAME"] = "test-observer"

        data = await fetch_generation_from_dp(
            "test-site",
            datetime(2024, 6, 1, tzinfo=UTC),
            datetime(2024, 6, 2, tzinfo=UTC),
        )

        assert len(data) == 1
        # 0.5 fraction of 2000W = 1000W = 1.0 kW
        assert data[0][1] == 1.0
        assert data[0][0] == datetime(2024, 6, 1, 10, 0, tzinfo=UTC)

        # Verify the request
        mock_client.get_observations_as_timeseries.assert_called_once()
        req = mock_client.get_observations_as_timeseries.call_args[0][0]
        assert req.location_uuid == "uuid-123"
        assert req.observer_name == "test-observer"
        mock_client.get_location.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_generation_from_dp_no_site():
    """Test when site is not found in the location map."""
    mock_client = AsyncMock()
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_client

    with (
        patch(
            "site_forecast_app.save.data_platform.get_dataplatform_client",
            return_value=mock_ctx,
        ),
        patch(
            "site_forecast_app.save.data_platform.fetch_dp_location_map",
            return_value={"other-site": "uuid-123"},
        ),
    ):
        data = await fetch_generation_from_dp(
            "missing-site",
            datetime(2024, 6, 1, tzinfo=UTC),
            datetime(2024, 6, 2, tzinfo=UTC),
        )
        assert data == []
        mock_client.get_observations_as_timeseries.assert_not_called()
