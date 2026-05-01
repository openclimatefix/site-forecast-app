import logging
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from site_forecast_app.blend.app import run_blend_app

# ---------------------------------------------------------------------------
# Unit tests for the NL blend application orchestration
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_dependencies():
    """Mock out all Data Platform and heavy compute dependencies."""
    with (
        patch("site_forecast_app.blend.app.get_dataplatform_client") as mock_client_ctx,
        patch(
            "site_forecast_app.blend.app.fetch_dp_location_map", new_callable=AsyncMock,
        ) as mock_loc_map,
        patch("site_forecast_app.blend.app.load_nl_mae_scorecard") as mock_load_mae,
        patch(
            "site_forecast_app.blend.app.get_blend_weights", new_callable=AsyncMock,
        ) as mock_weights,
        patch(
            "site_forecast_app.blend.app.get_blend_forecast_values_latest", new_callable=AsyncMock,
        ) as mock_blend,
        patch("site_forecast_app.blend.app._save_forecasts", new_callable=AsyncMock) as mock_save,
    ):

        mock_client = AsyncMock()
        mock_client_ctx.return_value.__aenter__.return_value = mock_client
        mock_client_ctx.return_value.__aexit__.return_value = None

        yield {
            "client_ctx": mock_client_ctx,
            "client": mock_client,
            "fetch_dp_location_map": mock_loc_map,
            "load_nl_mae_scorecard": mock_load_mae,
            "get_blend_weights": mock_weights,
            "get_blend_forecast_values_latest": mock_blend,
            "_save_forecasts": mock_save,
        }

def _mock_scorecard() -> pd.DataFrame:
    """Returns a minimal scorecard DataFrame with a valid Timedelta index."""
    return pd.DataFrame(
        {"model_A": [0.1]},
        index=pd.to_timedelta(["24h"]),
    )

@pytest.mark.asyncio
async def test_run_blend_app_success(mock_dependencies):
    """Test full execution path: both main blend and adjuster pass run (use_adjuster=True)."""
    deps = mock_dependencies

    deps["fetch_dp_location_map"].return_value = {"site_id": "test-uuid"}
    deps["load_nl_mae_scorecard"].return_value = _mock_scorecard()
    deps["get_blend_weights"].return_value = pd.DataFrame({"model_A": [1.0]})

    # Mocking non-empty result to trigger saving
    mock_blend_df = pd.DataFrame({"target_time": [], "expected_power_generation_megawatts": []})
    mock_blend_df.loc[0] = [pd.Timestamp("2024-01-01 12:00", tz="UTC"), 10.0]
    deps["get_blend_forecast_values_latest"].return_value = mock_blend_df

    await run_blend_app()

    # These are called once per run (shared setup)
    deps["fetch_dp_location_map"].assert_called_once()
    deps["load_nl_mae_scorecard"].assert_called_once()

    # get_blend_weights / blend / save are each called twice:
    # once for the main blend pass and once for the adjuster pass.
    assert deps["get_blend_weights"].call_count == 2, (
        f"Expected get_blend_weights to be called 2 times (main + adjuster), "
        f"got {deps['get_blend_weights'].call_count}"
    )
    assert deps["get_blend_forecast_values_latest"].call_count == 2, (
        f"Expected get_blend_forecast_values_latest to be called 2 times, "
        f"got {deps['get_blend_forecast_values_latest'].call_count}"
    )
    assert deps["_save_forecasts"].call_count == 2, (
        f"Expected _save_forecasts to be called 2 times (main + adjuster), "
        f"got {deps['_save_forecasts'].call_count}"
    )

@pytest.mark.asyncio
async def test_run_blend_app_aborts_on_empty_location(caplog, mock_dependencies):
    """Test early exit if no location map is returned."""
    deps = mock_dependencies

    deps["fetch_dp_location_map"].return_value = {}

    with caplog.at_level(logging.ERROR, logger="blend_app"):
        await run_blend_app()

    assert "empty location map" in caplog.text

@pytest.mark.asyncio
async def test_run_blend_app_aborts_on_empty_blend(caplog, mock_dependencies):
    """Test safe exit without saving if no blended forecasts are generated."""
    deps = mock_dependencies

    deps["fetch_dp_location_map"].return_value = {"site_id": "test-uuid"}
    deps["load_nl_mae_scorecard"].return_value = _mock_scorecard()
    deps["get_blend_weights"].return_value = pd.DataFrame({"model_A": [1.0]})

    # Return empty blended DF
    deps["get_blend_forecast_values_latest"].return_value = pd.DataFrame()

    with caplog.at_level(logging.WARNING, logger="site_forecast_app.blend.blend"):
        await run_blend_app()

    deps["_save_forecasts"].assert_not_called()
