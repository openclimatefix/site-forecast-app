"""Unit tests for the DE blend"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from dp_sdk.ocf import dp

from site_forecast_app import blend
from site_forecast_app.blend.app import run_blend_app
from site_forecast_app.blend.init_times import load_nl_mae_scorecard

BLEND_DIR = Path(blend.__file__).parent

DE_NATIONAL = "de_national"
DE_ZONES = ["de_50hertz", "de_amprion", "de_tennet", "de_transnetbw"]


@pytest.fixture
def mock_blend(monkeypatch, de_dp_locations):
    """Patch the Data Platform client and the blend's weight, fetch and save steps."""
    monkeypatch.setenv("DATA_PLATFORM_HOST", "mock_host")
    monkeypatch.setenv("DATA_PLATFORM_PORT", "50051")

    with (
        patch("site_forecast_app.blend.app.get_dataplatform_client") as get_client,
        patch(
            "site_forecast_app.blend.app.get_blend_weights", new_callable=AsyncMock,
        ) as weights,
        patch(
            "site_forecast_app.blend.app.get_regional_blend_weights", new_callable=AsyncMock,
        ) as regional_weights,
        patch(
            "site_forecast_app.blend.app.get_blend_forecast_values_latest", new_callable=AsyncMock,
        ) as blend_values,
        patch("site_forecast_app.blend.app._save_forecasts", new_callable=AsyncMock) as save,
    ):
        client = AsyncMock()
        client.list_locations.return_value = MagicMock(locations=de_dp_locations)
        get_client.return_value.__aenter__.return_value = client

        weights.return_value = pd.DataFrame({"de_ecmwf_pv": [1.0]})
        regional_weights.return_value = pd.DataFrame({"de_ecmwf_pv": [1.0]})
        blend_values.return_value = pd.DataFrame(
            {
                "target_time": [pd.Timestamp("2026-09-21 12:00", tz="UTC")],
                "expected_power_generation_megawatts": [10.0],
            },
        )

        yield {
            "client": client,
            "weights": weights,
            "regional_weights": regional_weights,
            "blend_values": blend_values,
            "save": save,
        }


def _saved(mock_blend: dict, field: str) -> list:
    """The given argument of every _save_forecasts call, in call order."""
    return [call.kwargs[field] for call in mock_blend["save"].call_args_list]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blends_the_nation_and_every_zone(mock_blend, de_blend_config):
    """DE blends the nation, then each of the four TSO zones, then the adjuster pass."""
    await run_blend_app(config=de_blend_config)

    assert _saved(mock_blend, "location_key") == [DE_NATIONAL, *DE_ZONES, DE_NATIONAL]


@pytest.mark.asyncio
async def test_adjuster_pass_is_national_only(mock_blend, de_blend_config):
    """Zones have no {model}_adjust forecasters to blend, so only the nation is readjusted."""
    await run_blend_app(config=de_blend_config)

    forecasters = _saved(mock_blend, "forecaster_name")
    assert forecasters.count("de_blend") == 1 + len(DE_ZONES)
    assert forecasters.count("de_blend_adjust") == 1


@pytest.mark.asyncio
async def test_zones_use_regional_weights(mock_blend, de_blend_config):
    """The nation uses the national candidate set, zones use the regional one."""
    await run_blend_app(config=de_blend_config)

    assert mock_blend["weights"].call_count == 2  # national blend + adjuster
    assert mock_blend["regional_weights"].call_count == len(DE_ZONES)


@pytest.mark.asyncio
async def test_nothing_saved_when_location_map_is_empty(mock_blend, de_blend_config):
    """An empty location map aborts the run rather than blending the wrong location."""
    mock_blend["client"].list_locations.return_value = MagicMock(locations=[])

    await run_blend_app(config=de_blend_config)

    mock_blend["save"].assert_not_called()


@pytest.mark.asyncio
async def test_nothing_saved_when_blend_is_empty(mock_blend, de_blend_config):
    """An empty blend result is skipped instead of written as a gap."""
    mock_blend["blend_values"].return_value = pd.DataFrame()

    await run_blend_app(config=de_blend_config)

    mock_blend["save"].assert_not_called()


# ---------------------------------------------------------------------------
# Config against the data it points at
# ---------------------------------------------------------------------------


class TestDeConfig:
    """The DE config must match the scorecard and the DE locations."""

    def test_scorecard_covers_every_configured_model(self, de_blend_config):
        """Models missing from the scorecard are dropped when the MAE curves shift."""
        df_mae = load_nl_mae_scorecard(str(BLEND_DIR / de_blend_config.scorecard_path))
        configured = {
            de_blend_config.backup_model,
            *de_blend_config.day_ahead_candidate_models,
            *de_blend_config.regional_candidate_models,
        }

        missing = sorted(configured - set(df_mae.columns))
        assert not missing, f"models in config.yaml but not in the scorecard: {missing}"

    def test_scorecard_has_no_gaps(self, de_blend_config):
        """A NaN in the scorecard propagates into the weight optimisation."""
        df_mae = load_nl_mae_scorecard(str(BLEND_DIR / de_blend_config.scorecard_path))

        assert not df_mae.isna().to_numpy().any()

    def test_scorecard_reaches_the_minimum_horizon(self, de_blend_config):
        """A scorecard starting above min_forecast_horizon would emit nothing."""
        df_mae = load_nl_mae_scorecard(str(BLEND_DIR / de_blend_config.scorecard_path))

        assert df_mae.index.min() <= de_blend_config.min_forecast_horizon

    def test_regional_location_type_matches_the_de_zones(self, de_blend_config, de_dp_locations):
        """The configured regional type must be the one the DE zones actually use."""
        zone_types = {
            location.location_type
            for location in de_dp_locations
            if location.location_name != de_blend_config.national_location_key
        }

        assert zone_types == {getattr(dp.LocationType, de_blend_config.regional_location_type)}

    def test_national_location_key_is_a_de_location(self, de_blend_config, de_dp_locations):
        """A key not in the map makes the blend silently fall back to the first location."""
        names = {location.location_name for location in de_dp_locations}

        assert de_blend_config.national_location_key in names
