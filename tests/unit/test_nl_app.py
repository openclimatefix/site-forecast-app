import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from nl_blend.app import run_blend_app

# ---------------------------------------------------------------------------
# Unit tests for the NL blend application orchestration
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_dependencies():
    """Mock out all Data Platform and heavy compute dependencies."""
    with patch("nl_blend.app.get_dataplatform_client") as mock_client_ctx, \
         patch("nl_blend.app.fetch_dp_location_map", new_callable=AsyncMock) as mock_loc_map, \
         patch("nl_blend.app.fetch_location_capacity_watts", new_callable=AsyncMock) as mock_capacity, \
         patch("nl_blend.app.load_nl_mae_scorecard") as mock_load_mae, \
         patch("nl_blend.app.get_nl_blend_weights", new_callable=AsyncMock) as mock_weights, \
         patch("nl_blend.app.get_blend_forecast_values_latest", new_callable=AsyncMock) as mock_blend, \
         patch("nl_blend.app._save_forecasts", new_callable=AsyncMock) as mock_save:
             
        mock_client = AsyncMock()
        mock_client_ctx.return_value.__aenter__.return_value = mock_client
        mock_client_ctx.return_value.__aexit__.return_value = None

        yield {
            "client_ctx": mock_client_ctx,
            "client": mock_client,
            "fetch_dp_location_map": mock_loc_map,
            "fetch_location_capacity_watts": mock_capacity,
            "load_nl_mae_scorecard": mock_load_mae,
            "get_nl_blend_weights": mock_weights,
            "get_blend_forecast_values_latest": mock_blend,
            "_save_forecasts": mock_save
        }

@pytest.mark.asyncio
async def test_run_blend_app_success(mock_dependencies):
    """Test full execution path assuming everything returns data."""
    deps = mock_dependencies

    deps["fetch_dp_location_map"].return_value = {"site_id": "test-uuid"}
    deps["fetch_location_capacity_watts"].return_value = 1000000 
    deps["load_nl_mae_scorecard"].return_value = pd.DataFrame()
    deps["get_nl_blend_weights"].return_value = pd.DataFrame({"model_A": [1.0]})
    
    # Mocking non-empty result to trigger saving
    mock_blend_df = pd.DataFrame({"target_time": [], "expected_power_generation_megawatts": []})
    mock_blend_df.loc[0] = [pd.Timestamp("2024-01-01 12:00", tz="UTC"), 10.0]
    deps["get_blend_forecast_values_latest"].return_value = mock_blend_df

    await run_blend_app()

    deps["fetch_dp_location_map"].assert_called_once()
    deps["fetch_location_capacity_watts"].assert_called_once_with(
        client=deps["client"], location_uuid="test-uuid"
    )
    deps["load_nl_mae_scorecard"].assert_called_once()
    deps["get_nl_blend_weights"].assert_called_once()
    deps["get_blend_forecast_values_latest"].assert_called_once()
    deps["_save_forecasts"].assert_called_once()

@pytest.mark.asyncio
async def test_run_blend_app_aborts_on_empty_location(caplog, mock_dependencies):
    """Test early exit if no location map is returned."""
    deps = mock_dependencies

    deps["fetch_dp_location_map"].return_value = {}

    with caplog.at_level(logging.ERROR, logger="nl_blend_app"):
        await run_blend_app()

    assert "empty location map" in caplog.text
    deps["fetch_location_capacity_watts"].assert_not_called()

@pytest.mark.asyncio
async def test_run_blend_app_aborts_on_empty_blend(caplog, mock_dependencies):
    """Test safe exit without saving if no blended forecasts are generated."""
    deps = mock_dependencies

    deps["fetch_dp_location_map"].return_value = {"site_id": "test-uuid"}
    deps["fetch_location_capacity_watts"].return_value = 1000000 
    deps["load_nl_mae_scorecard"].return_value = pd.DataFrame()
    deps["get_nl_blend_weights"].return_value = pd.DataFrame({"model_A": [1.0]})
    
    # Return empty blended DF
    deps["get_blend_forecast_values_latest"].return_value = pd.DataFrame()

    with caplog.at_level(logging.WARNING, logger="nl_blend.blend"):
        await run_blend_app()

    deps["_save_forecasts"].assert_not_called()
