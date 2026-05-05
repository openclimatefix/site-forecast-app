import logging
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
from tests.integration.conftest import dp_address
from site_forecast_app.blend.app import rename_columns_with_adjuster, run_blend_app
from site_forecast_app.save.data_platform import get_dataplatform_client
from dp_sdk.ocf import dp
from datetime import datetime, UTC

# ---------------------------------------------------------------------------
# Unit tests for the NL blend application orchestration
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_dependencies(dp_address, monkeypatch):
    """Mock out heavy compute dependencies, but use actual DP container."""
    host, port = dp_address
    monkeypatch.setenv("DATA_PLATFORM_HOST", host)
    monkeypatch.setenv("DATA_PLATFORM_PORT", str(port))

    with (
        patch("site_forecast_app.blend.app.load_nl_mae_scorecard") as mock_load_mae,
        patch(
            "site_forecast_app.blend.app.get_blend_weights", new_callable=AsyncMock,
        ) as mock_weights,
        patch(
            "site_forecast_app.blend.app.get_regional_blend_weights", new_callable=AsyncMock,
        ) as mock_regional_weights,
        patch(
            "site_forecast_app.blend.app.get_blend_forecast_values_latest", new_callable=AsyncMock,
        ) as mock_blend,
        patch("site_forecast_app.blend.app._save_forecasts", new_callable=AsyncMock) as mock_save,
    ):
        yield {
            "load_nl_mae_scorecard": mock_load_mae,
            "get_blend_weights": mock_weights,
            "get_regional_blend_weights": mock_regional_weights,
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

    async with get_dataplatform_client() as client:
        await client.create_location(
            dp.CreateLocationRequest(
                location_name="site_id",
                energy_source=dp.EnergySource.SOLAR,
                geometry_wkt="POINT(0 0)",
                location_type=dp.LocationType.SITE,
                effective_capacity_watts=1_000,
                valid_from_utc=datetime(2020, 1, 1, tzinfo=UTC),
            )
        )
    deps["load_nl_mae_scorecard"].return_value = _mock_scorecard()
    deps["get_blend_weights"].return_value = pd.DataFrame({"model_A": [1.0]})

    # Mocking non-empty result to trigger saving
    mock_blend_df = pd.DataFrame({"target_time": [], "expected_power_generation_megawatts": []})
    mock_blend_df.loc[0] = [pd.Timestamp("2024-01-01 12:00", tz="UTC"), 10.0]
    deps["get_blend_forecast_values_latest"].return_value = mock_blend_df

    await run_blend_app()

    # These are called once per run (shared setup)
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

    # We do not seed any locations here, so list_locations will return empty.

    with caplog.at_level(logging.ERROR, logger="blend_app"):
        await run_blend_app()

    assert "empty location map" in caplog.text

@pytest.mark.asyncio
async def test_run_blend_app_aborts_on_empty_blend(caplog, mock_dependencies):
    """Test safe exit without saving if no blended forecasts are generated."""
    deps = mock_dependencies

    async with get_dataplatform_client() as client:
        await client.create_location(
            dp.CreateLocationRequest(
                location_name="site_id",
                energy_source=dp.EnergySource.SOLAR,
                geometry_wkt="POINT(0 0)",
                location_type=dp.LocationType.SITE,
                effective_capacity_watts=1_000,
                valid_from_utc=datetime(2020, 1, 1, tzinfo=UTC),
            )
        )
    deps["load_nl_mae_scorecard"].return_value = _mock_scorecard()
    deps["get_blend_weights"].return_value = pd.DataFrame({"model_A": [1.0]})

    # Return empty blended DF
    deps["get_blend_forecast_values_latest"].return_value = pd.DataFrame()

    with caplog.at_level(logging.WARNING, logger="site_forecast_app.blend.blend"):
        await run_blend_app()

    deps["_save_forecasts"].assert_not_called()


# ---------------------------------------------------------------------------
# Tests: rename_columns_with_adjuster
# ---------------------------------------------------------------------------


class TestRenameColumnsWithAdjuster:
    """Tests for the helper that appends '_adjust' to weight column names."""

    def test_renames_all_columns(self):
        """Every column name gets the '_adjust' suffix."""
        df = pd.DataFrame({"model_A": [0.6], "model_B": [0.4]})
        renamed_df = rename_columns_with_adjuster(df)

        assert list(renamed_df.columns) == ["model_A_adjust", "model_B_adjust"]

    def test_empty_dataframe(self):
        """Works cleanly on an empty DataFrame."""
        df = pd.DataFrame()
        renamed_df = rename_columns_with_adjuster(df)

        assert list(renamed_df.columns) == []

    def test_original_dataframe_is_unmodified(self):
        """The original DataFrame columns should not be mutated."""
        df = pd.DataFrame({"model_A": [0.6]})
        rename_columns_with_adjuster(df)

        assert list(df.columns) == ["model_A"]

@pytest.mark.asyncio
async def test_run_blend_app_filters_regional_locations(mock_dependencies):
    """Test that regional blends are only run for locations starting with 'nl_'."""
    deps = mock_dependencies

    async with get_dataplatform_client() as client:
        # Create national location
        await client.create_location(
            dp.CreateLocationRequest(
                location_name="nl_national",
                energy_source=dp.EnergySource.SOLAR,
                geometry_wkt="POINT(0 0)",
                location_type=dp.LocationType.NATION,
                effective_capacity_watts=1_000,
                valid_from_utc=datetime(2020, 1, 1, tzinfo=UTC),
            )
        )
        # Create regional location
        await client.create_location(
            dp.CreateLocationRequest(
                location_name="nl_groningen",
                energy_source=dp.EnergySource.SOLAR,
                geometry_wkt="POINT(0 0)",
                location_type=dp.LocationType.STATE,
                effective_capacity_watts=1_000,
                valid_from_utc=datetime(2020, 1, 1, tzinfo=UTC),
            )
        )
        # Create other locations (should be filtered out)
        await client.create_location(
            dp.CreateLocationRequest(
                location_name="taun1",
                energy_source=dp.EnergySource.SOLAR,
                geometry_wkt="POINT(0 0)",
                location_type=dp.LocationType.SITE,
                effective_capacity_watts=1_000,
                valid_from_utc=datetime(2020, 1, 1, tzinfo=UTC),
            )
        )
        await client.create_location(
            dp.CreateLocationRequest(
                location_name="temp_3",
                energy_source=dp.EnergySource.SOLAR,
                geometry_wkt="POINT(0 0)",
                location_type=dp.LocationType.SITE,
                effective_capacity_watts=1_000,
                valid_from_utc=datetime(2020, 1, 1, tzinfo=UTC),
            )
        )
    deps["load_nl_mae_scorecard"].return_value = _mock_scorecard()
    deps["get_blend_weights"].return_value = pd.DataFrame({"model_A": [1.0]})
    deps["get_regional_blend_weights"].return_value = pd.DataFrame({"model_A": [1.0]})

    mock_blend_df = pd.DataFrame({"target_time": [], "expected_power_generation_megawatts": []})
    mock_blend_df.loc[0] = [pd.Timestamp("2024-01-01 12:00", tz="UTC"), 10.0]
    deps["get_blend_forecast_values_latest"].return_value = mock_blend_df

    await run_blend_app()

    # Should run main blend and adjuster for:
    # 1. nl_national (National pass) -> uses get_blend_weights (2 calls)
    # 2. nl_groningen (Regional pass) -> uses get_regional_blend_weights (2 calls)
    # taun1 and temp_3 should be filtered out.
    # Total = 2 locations * 2 passes = 4 blend & save calls
    assert deps["get_blend_weights"].call_count == 2
    assert deps["get_regional_blend_weights"].call_count == 2
    assert deps["get_blend_forecast_values_latest"].call_count == 4
    assert deps["_save_forecasts"].call_count == 4
