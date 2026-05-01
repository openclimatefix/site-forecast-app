"""Unit tests for the NL Blend adjuster logic.

Covers:
  - NlBlendConfig.adjuster_forecaster_name: correct '_adjust' suffix.
  - NlBlendConfig.use_adjuster: flag is honoured by the config model.
  - _run_blend_pass: weight columns are renamed with '_adjust' suffix
    when use_adjuster=True.
  - run_blend_app: adjuster pass is executed iff use_adjuster=True in config.
"""
import logging
from typing import ClassVar
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from site_forecast_app.blend.app import _run_blend_pass, run_blend_app
from site_forecast_app.blend.config import NlBlendConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> NlBlendConfig:
    """Build a minimal NlBlendConfig, optionally overriding fields."""
    defaults = {
        "backup_model": "nl_regional_2h_pv_ecmwf",
        "national_candidate_models": ["nl_regional_48h_pv_ecmwf"],
        "regional_candidate_models": ["nl_regional_48h_pv_ecmwf"],
        "blend_kernel": [0.75, 0.5, 0.25],
        "forecaster_name": "nl_blend",
        "use_adjuster": False,
    }
    defaults.update(overrides)
    return NlBlendConfig(**defaults)


def _mock_scorecard() -> pd.DataFrame:
    """Minimal (horizon x model) MAE scorecard."""
    return pd.DataFrame(
        {"model_A": [0.1]},
        index=pd.to_timedelta(["24h"]),
    )


def _non_empty_blend_df() -> pd.DataFrame:
    """Minimal blended DataFrame with one row so _save_forecasts is reached."""
    df = pd.DataFrame(columns=["target_time", "expected_power_generation_megawatts"])
    df.loc[0] = [pd.Timestamp("2024-01-01 12:00", tz="UTC"), 10.0]
    return df


# ---------------------------------------------------------------------------
# Tests: NlBlendConfig — adjuster_forecaster_name
# ---------------------------------------------------------------------------


class TestAdjusterForecasterName:
    """Tests for the adjuster_forecaster_name computed property."""

    @pytest.mark.parametrize(
        ("forecaster_name", "expected"),
        [
            ("nl_blend", "nl_blend_adjust"),
            ("my_custom_forecaster", "my_custom_forecaster_adjust"),
            ("blend_v2", "blend_v2_adjust"),
            ("x", "x_adjust"),
        ],
    )
    def test_various_forecaster_names(self, forecaster_name, expected):
        """adjuster_forecaster_name always appends '_adjust'."""
        cfg = _make_config(forecaster_name=forecaster_name)
        assert cfg.adjuster_forecaster_name == expected

    def test_adjuster_name_differs_from_base(self):
        """adjuster_forecaster_name must not equal forecaster_name."""
        cfg = _make_config(forecaster_name="nl_blend")
        assert cfg.adjuster_forecaster_name != cfg.forecaster_name


# ---------------------------------------------------------------------------
# Tests: NlBlendConfig — use_adjuster flag
# ---------------------------------------------------------------------------


class TestUseAdjusterFlag:
    """Tests for the use_adjuster config flag."""

    def test_use_adjuster_defaults_to_false(self):
        """use_adjuster is False when not explicitly set."""
        cfg = _make_config()
        assert cfg.use_adjuster is False

    def test_use_adjuster_can_be_set_true(self):
        """use_adjuster=True is accepted and stored correctly."""
        cfg = _make_config(use_adjuster=True)
        assert cfg.use_adjuster is True


# ---------------------------------------------------------------------------
# Tests: _run_blend_pass — weight column renaming
# ---------------------------------------------------------------------------


class TestRunBlendPassAdjusterColumns:
    """Tests that weight columns are renamed with '_adjust' when use_adjuster=True."""

    WEIGHTS_DF: ClassVar[pd.DataFrame] = pd.DataFrame(
        {"model_A": [0.6], "model_B": [0.4]},
    )

    async def _captured_columns(self, *, use_adjuster: bool) -> list[str]:
        """Run one blend pass and return the weight column names seen by the blender."""
        captured: list[pd.DataFrame] = []

        async def capture_blend(weights_df, **_kwargs):
            captured.append(weights_df.copy())
            return pd.DataFrame()  # empty -> no save

        forecaster_name = "nl_blend_adjust" if use_adjuster else "nl_blend"

        with (
            patch(
                "site_forecast_app.blend.app.get_blend_weights",
                new_callable=AsyncMock,
                return_value=self.WEIGHTS_DF.copy(),
            ),
            patch(
                "site_forecast_app.blend.app.get_blend_forecast_values_latest",
                side_effect=capture_blend,
            ),
        ):
            await _run_blend_pass(
                client=AsyncMock(),
                t0=pd.Timestamp("2024-01-01 12:00", tz="UTC"),
                location_uuid="test-uuid",
                location_key="nl_national",
                df_mae=_mock_scorecard(),
                max_horizon=pd.Timedelta("24h"),
                forecaster_name=forecaster_name,
                use_adjuster=use_adjuster,
            )

        assert len(captured) == 1
        return list(captured[0].columns)

    @pytest.mark.asyncio
    async def test_adjuster_pass_renames_weight_columns(self):
        """Weight columns gain '_adjust' suffix when use_adjuster=True."""
        cols = await self._captured_columns(use_adjuster=True)
        assert all(col.endswith("_adjust") for col in cols), (
            f"Expected all columns to end with '_adjust', got: {cols}"
        )

    @pytest.mark.asyncio
    async def test_main_pass_does_not_rename_columns(self):
        """Weight columns are NOT renamed when use_adjuster=False."""
        cols = await self._captured_columns(use_adjuster=False)
        assert not any(col.endswith("_adjust") for col in cols), (
            f"Expected no '_adjust' columns in main pass, got: {cols}"
        )

    @pytest.mark.asyncio
    async def test_adjuster_column_names_match_originals_plus_suffix(self):
        """Each renamed column is exactly '{original}_adjust'."""
        original_cols = list(self.WEIGHTS_DF.columns)
        renamed = await self._captured_columns(use_adjuster=True)
        assert renamed == [f"{c}_adjust" for c in original_cols]


# ---------------------------------------------------------------------------
# Tests: run_blend_app — conditional adjuster pass
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_app_dependencies():
    """Mock all external I/O used by run_blend_app."""
    with (
        patch("site_forecast_app.blend.app.get_dataplatform_client") as mock_client_ctx,
        patch(
            "site_forecast_app.blend.app.fetch_dp_location_map",
            new_callable=AsyncMock,
        ) as mock_loc_map,
        patch("site_forecast_app.blend.app.load_nl_mae_scorecard") as mock_load_mae,
        patch(
            "site_forecast_app.blend.app.get_blend_weights",
            new_callable=AsyncMock,
        ) as mock_weights,
        patch(
            "site_forecast_app.blend.app.get_blend_forecast_values_latest",
            new_callable=AsyncMock,
        ) as mock_blend,
        patch(
            "site_forecast_app.blend.app._save_forecasts",
            new_callable=AsyncMock,
        ) as mock_save,
        patch("site_forecast_app.blend.app.load_blend_config") as mock_cfg,
    ):
        mock_client = AsyncMock()
        mock_client_ctx.return_value.__aenter__.return_value = mock_client
        mock_client_ctx.return_value.__aexit__.return_value = None

        mock_loc_map.return_value = {"nl_national": "uuid-123"}
        mock_load_mae.return_value = _mock_scorecard()
        mock_weights.return_value = pd.DataFrame({"model_A": [1.0]})
        mock_blend.return_value = _non_empty_blend_df()

        yield {
            "client": mock_client,
            "fetch_dp_location_map": mock_loc_map,
            "load_nl_mae_scorecard": mock_load_mae,
            "get_blend_weights": mock_weights,
            "get_blend_forecast_values_latest": mock_blend,
            "_save_forecasts": mock_save,
            "load_blend_config": mock_cfg,
        }


class TestRunBlendAppAdjusterPass:
    """Tests for the conditional adjuster pass in run_blend_app."""

    @pytest.mark.asyncio
    async def test_adjuster_pass_runs_when_use_adjuster_true(self, mock_app_dependencies):
        """Weights, blend, and save are each called twice when use_adjuster=True."""
        deps = mock_app_dependencies
        deps["load_blend_config"].return_value = _make_config(use_adjuster=True)

        await run_blend_app()

        assert deps["get_blend_weights"].call_count == 2
        assert deps["get_blend_forecast_values_latest"].call_count == 2
        assert deps["_save_forecasts"].call_count == 2

    @pytest.mark.asyncio
    async def test_adjuster_pass_skipped_when_use_adjuster_false(self, mock_app_dependencies):
        """Weights, blend, and save are each called once when use_adjuster=False."""
        deps = mock_app_dependencies
        deps["load_blend_config"].return_value = _make_config(use_adjuster=False)

        await run_blend_app()

        assert deps["get_blend_weights"].call_count == 1
        assert deps["get_blend_forecast_values_latest"].call_count == 1
        assert deps["_save_forecasts"].call_count == 1

    @pytest.mark.asyncio
    async def test_adjuster_pass_uses_adjuster_forecaster_name(self, mock_app_dependencies):
        """The adjuster save call uses '{forecaster_name}_adjust'."""
        deps = mock_app_dependencies
        cfg = _make_config(use_adjuster=True, forecaster_name="nl_blend")
        deps["load_blend_config"].return_value = cfg

        await run_blend_app()

        save_calls = deps["_save_forecasts"].call_args_list
        assert len(save_calls) == 2
        assert save_calls[1].kwargs["forecaster_name"] == cfg.adjuster_forecaster_name

    @pytest.mark.asyncio
    async def test_main_pass_uses_base_forecaster_name(self, mock_app_dependencies):
        """The first save call uses the base forecaster_name (not suffixed)."""
        deps = mock_app_dependencies
        cfg = _make_config(use_adjuster=True, forecaster_name="nl_blend")
        deps["load_blend_config"].return_value = cfg

        await run_blend_app()

        save_calls = deps["_save_forecasts"].call_args_list
        assert save_calls[0].kwargs["forecaster_name"] == cfg.forecaster_name

    @pytest.mark.asyncio
    async def test_adjuster_pass_logs_start(self, mock_app_dependencies, caplog):
        """A log message is emitted when the adjuster pass begins."""
        deps = mock_app_dependencies
        deps["load_blend_config"].return_value = _make_config(use_adjuster=True)

        with caplog.at_level(logging.INFO, logger="blend_app"):
            await run_blend_app()

        assert any("adjuster" in r.message.lower() for r in caplog.records), (
            "Expected a log message mentioning 'adjuster' when use_adjuster=True"
        )
