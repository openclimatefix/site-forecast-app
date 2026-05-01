"""Unit tests for the NL Blend adjuster logic.

Covers:
  - NlBlendConfig.adjuster_forecaster_name: correct '_adjust' suffix.
  - NlBlendConfig.use_adjuster: flag is honoured by the config model.
  - _run_blend_pass: weight columns are renamed with '_adjust' suffix
    when use_adjuster=True.
"""
from typing import ClassVar
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from site_forecast_app.blend.app import _run_blend_pass
from site_forecast_app.blend.config import NlBlendConfig, load_blend_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> NlBlendConfig:
    """Load the real config.yaml and override specific fields for a test."""
    return load_blend_config().model_copy(update=overrides)


def _mock_scorecard() -> pd.DataFrame:
    """Minimal (horizon x model) MAE scorecard."""
    return pd.DataFrame(
        {"model_A": [0.1]},
        index=pd.to_timedelta(["24h"]),
    )


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
        cfg = _cfg(forecaster_name=forecaster_name)
        assert cfg.adjuster_forecaster_name == expected

    def test_adjuster_name_differs_from_base(self):
        """adjuster_forecaster_name must not equal forecaster_name."""
        cfg = _cfg()
        assert cfg.adjuster_forecaster_name != cfg.forecaster_name


# ---------------------------------------------------------------------------
# Tests: NlBlendConfig — use_adjuster flag
# ---------------------------------------------------------------------------


class TestUseAdjusterFlag:
    """Tests for the use_adjuster config flag."""

    def test_use_adjuster_is_configurable_true(self):
        """use_adjuster=True round-trips through the model."""
        cfg = _cfg(use_adjuster=True)
        assert cfg.use_adjuster is True

    def test_use_adjuster_is_configurable_false(self):
        """use_adjuster=False can be set regardless of config.yaml default."""
        cfg = _cfg(use_adjuster=False)
        assert cfg.use_adjuster is False


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
