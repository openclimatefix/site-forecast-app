"""Unit tests for the Blend configuration."""

import pytest

from site_forecast_app.blend.config import BlendConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(base_cfg: BlendConfig, **overrides) -> BlendConfig:
    """Load the real config.yaml and override specific fields for a test."""
    return base_cfg.model_copy(update=overrides)


# ---------------------------------------------------------------------------
# Tests: BlendConfig — adjuster_forecaster_name
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
    def test_various_forecaster_names(self, blend_config, forecaster_name, expected):
        """adjuster_forecaster_name always appends '_adjust'."""
        cfg = _cfg(blend_config, forecaster_name=forecaster_name)
        assert cfg.adjuster_forecaster_name == expected

    def test_adjuster_name_differs_from_base(self, blend_config):
        """adjuster_forecaster_name must not equal forecaster_name."""
        cfg = _cfg(blend_config)
        assert cfg.adjuster_forecaster_name != cfg.forecaster_name


# ---------------------------------------------------------------------------
# Tests: BlendConfig — use_adjuster flag
# ---------------------------------------------------------------------------


class TestUseAdjusterFlag:
    """Tests for the use_adjuster config flag."""

    def test_use_adjuster_is_configurable_true(self, blend_config):
        """use_adjuster=True round-trips through the model."""
        cfg = _cfg(blend_config, use_adjuster=True)
        assert cfg.use_adjuster is True

    def test_use_adjuster_is_configurable_false(self, blend_config):
        """use_adjuster=False can be set regardless of config.yaml default."""
        cfg = _cfg(blend_config, use_adjuster=False)
        assert cfg.use_adjuster is False
