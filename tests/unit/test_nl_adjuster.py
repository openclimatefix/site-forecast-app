"""Unit tests for the NL Blend adjuster logic.

Covers:
  - NlBlendConfig.adjuster_forecaster_name: correct '_adjust' suffix.
  - NlBlendConfig.use_adjuster: flag is honoured by the config model.
  - _run_blend_pass: weight columns are renamed with '_adjust' suffix
    when use_adjuster=True.
"""

import pandas as pd
import pytest

from site_forecast_app.blend.app import rename_columns_with_adjuster
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
