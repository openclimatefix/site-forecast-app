"""Unit tests for blend.blend.blend_forecasts_together.

These tests verify the core blending algorithm in isolation, without any
Data Platform connections. They confirm that:
  - p50/p10/p90 values are correctly weighted and summed.
  - Missing model data causes the weight sum to deviate from 1.0 (triggering a warning).
  - Models whose p50 is NaN or absent at a given target time are skipped.
  - Empty inputs return an empty DataFrame with the correct columns.
"""
import math
from typing import ClassVar

import pandas as pd
import pytest

from site_forecast_app.blend.blend import blend_forecasts_together

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_df(target_times, model_name, p50_values, p10_values=None, p90_values=None):
    """Build a long-format model DataFrame for a single model."""
    rows = []
    for i, t in enumerate(target_times):
        row = {
            "target_time": t,
            "model_name": model_name,
            "expected_power_generation_megawatts": p50_values[i],
        }
        if p10_values is not None:
            row["p10_mw"] = p10_values[i]
        if p90_values is not None:
            row["p90_mw"] = p90_values[i]
        rows.append(row)
    return pd.DataFrame(rows)


def _make_weights_df(target_times, model_weights: dict):
    """Build a wide-format weights DataFrame indexed by target_time."""
    return pd.DataFrame(model_weights, index=pd.DatetimeIndex(target_times, tz="UTC"))


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------

T0 = pd.Timestamp("2024-06-01 09:00", tz="UTC")
TARGET_TIMES = [T0 + pd.Timedelta(f"{h}h") for h in range(1, 5)]


# ---------------------------------------------------------------------------
# Test: empty inputs
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    EXPECTED_COLUMNS: ClassVar[set[str]] = {
        "target_time",
        "expected_power_generation_megawatts",
        "p10_mw",
        "p90_mw",
    }

    def test_empty_model_df(self):
        """Empty model dataframe returns empty result with correct columns."""
        empty_model = pd.DataFrame(
            columns=[
                "target_time",
                "model_name",
                "expected_power_generation_megawatts",
                "p10_mw",
                "p90_mw",
            ],
        )
        weights = _make_weights_df(TARGET_TIMES, {"model_A": [1.0, 1.0, 1.0, 1.0]})
        result = blend_forecasts_together(empty_model, weights)

        assert result.empty
        assert set(result.columns) == self.EXPECTED_COLUMNS

    def test_empty_weights_df(self):
        """Empty weights dataframe returns empty result with correct columns."""
        models_df = _make_model_df(TARGET_TIMES, "model_A", [10.0, 20.0, 30.0, 40.0])
        empty_weights = pd.DataFrame()
        result = blend_forecasts_together(models_df, empty_weights)

        assert result.empty
        assert set(result.columns) == self.EXPECTED_COLUMNS

    def test_both_empty(self):
        """Both inputs empty returns empty result."""
        result = blend_forecasts_together(pd.DataFrame(), pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# Test: single model blending (100% weight)
# ---------------------------------------------------------------------------

class TestSingleModelBlend:
    def test_p50_passthrough_at_full_weight(self):
        """A single model at 100% weight should pass its p50 through unchanged."""
        p50 = [10.0, 20.0, 30.0, 40.0]
        models_df = _make_model_df(TARGET_TIMES, "model_A", p50)
        weights = _make_weights_df(TARGET_TIMES, {"model_A": [1.0, 1.0, 1.0, 1.0]})

        result = blend_forecasts_together(models_df, weights)

        assert len(result) == 4
        result_indexed = result.set_index("target_time")
        for i, t in enumerate(TARGET_TIMES):
            assert result_indexed.loc[
                t, "expected_power_generation_megawatts",
            ] == pytest.approx(p50[i])

    def test_p10_p90_passthrough_at_full_weight(self):
        """p10/p90 should also pass through unchanged when a single model has 100% weight."""
        p50 = [10.0, 20.0, 30.0, 40.0]
        p10 = [5.0, 10.0, 15.0, 20.0]
        p90 = [15.0, 30.0, 45.0, 60.0]
        models_df = _make_model_df(TARGET_TIMES, "model_A", p50, p10, p90)
        weights = _make_weights_df(TARGET_TIMES, {"model_A": [1.0, 1.0, 1.0, 1.0]})

        result = blend_forecasts_together(models_df, weights)

        result_indexed = result.set_index("target_time")
        for i, t in enumerate(TARGET_TIMES):
            assert result_indexed.loc[t, "p10_mw"] == pytest.approx(p10[i])
            assert result_indexed.loc[t, "p90_mw"] == pytest.approx(p90[i])


# ---------------------------------------------------------------------------
# Test: two-model blending
# ---------------------------------------------------------------------------

class TestTwoModelBlend:
    def test_50_50_blend_averages_p50(self):
        """50/50 weights produce the arithmetic mean of both models' p50."""
        p50_a = [10.0, 20.0, 30.0, 40.0]
        p50_b = [20.0, 40.0, 60.0, 80.0]
        expected_blend = [(a + b) / 2 for a, b in zip(p50_a, p50_b, strict=True)]

        df_a = _make_model_df(TARGET_TIMES, "model_A", p50_a)
        df_b = _make_model_df(TARGET_TIMES, "model_B", p50_b)
        models_df = pd.concat([df_a, df_b], ignore_index=True)

        weights = _make_weights_df(
            TARGET_TIMES,
            {"model_A": [0.5, 0.5, 0.5, 0.5], "model_B": [0.5, 0.5, 0.5, 0.5]},
        )

        result = blend_forecasts_together(models_df, weights)

        assert len(result) == 4
        result_indexed = result.set_index("target_time")
        for i, t in enumerate(TARGET_TIMES):
            assert result_indexed.loc[t, "expected_power_generation_megawatts"] == pytest.approx(
                expected_blend[i],
            )

    def test_75_25_blend_p50(self):
        """75/25 weights correctly weight the primary model more heavily."""
        p50_primary = [100.0] * 4
        p50_backup = [0.0] * 4

        df_p = _make_model_df(TARGET_TIMES, "primary", p50_primary)
        df_b = _make_model_df(TARGET_TIMES, "backup", p50_backup)
        models_df = pd.concat([df_p, df_b], ignore_index=True)

        weights = _make_weights_df(
            TARGET_TIMES,
            {"primary": [0.75, 0.75, 0.75, 0.75], "backup": [0.25, 0.25, 0.25, 0.25]},
        )

        result = blend_forecasts_together(models_df, weights)
        result_indexed = result.set_index("target_time")

        for t in TARGET_TIMES:
            assert result_indexed.loc[
                t, "expected_power_generation_megawatts",
            ] == pytest.approx(75.0)

    def test_50_50_blend_p10_p90(self):
        """50/50 weights should average p10 and p90 values correctly."""
        p10_a = [2.0, 4.0, 6.0, 8.0]
        p10_b = [4.0, 8.0, 12.0, 16.0]
        p90_a = [18.0, 36.0, 54.0, 72.0]
        p90_b = [10.0, 20.0, 30.0, 40.0]

        df_a = _make_model_df(TARGET_TIMES, "model_A", [10.0] * 4, p10_a, p90_a)
        df_b = _make_model_df(TARGET_TIMES, "model_B", [10.0] * 4, p10_b, p90_b)
        models_df = pd.concat([df_a, df_b], ignore_index=True)

        weights = _make_weights_df(
            TARGET_TIMES,
            {"model_A": [0.5] * 4, "model_B": [0.5] * 4},
        )

        result = blend_forecasts_together(models_df, weights)
        result_indexed = result.set_index("target_time")

        for i, t in enumerate(TARGET_TIMES):
            expected_p10 = (p10_a[i] + p10_b[i]) / 2
            expected_p90 = (p90_a[i] + p90_b[i]) / 2
            assert result_indexed.loc[t, "p10_mw"] == pytest.approx(expected_p10)
            assert result_indexed.loc[t, "p90_mw"] == pytest.approx(expected_p90)

    def test_taper_transition_between_models(self):
        """Kernel taper [0.75, 0.5, 0.25] blends correctly at the transition zone."""
        # Primary dominates early horizons, backup takes over at later horizons
        p50_primary = [50.0] * 4
        p50_backup = [100.0] * 4

        df_p = _make_model_df(TARGET_TIMES, "primary", p50_primary)
        df_b = _make_model_df(TARGET_TIMES, "backup", p50_backup)
        models_df = pd.concat([df_p, df_b], ignore_index=True)

        # Kernel [1.0, 0.75, 0.25, 0.0] simulates a taper
        primary_weights = [1.0, 0.75, 0.25, 0.0]
        backup_weights = [0.0, 0.25, 0.75, 1.0]

        weights = _make_weights_df(
            TARGET_TIMES,
            {"primary": primary_weights, "backup": backup_weights},
        )

        result = blend_forecasts_together(models_df, weights)
        result_indexed = result.set_index("target_time")

        for i, t in enumerate(TARGET_TIMES):
            expected = primary_weights[i] * 50.0 + backup_weights[i] * 100.0
            assert result_indexed.loc[t, "expected_power_generation_megawatts"] == pytest.approx(
                expected,
            )


# ---------------------------------------------------------------------------
# Test: missing model data
# ---------------------------------------------------------------------------

class TestMissingModelData:
    def test_one_model_missing_data_warns_and_uses_available(self, caplog):
        """If a model has weight but no data for a time step, a warning is logged
        and the blend only uses the model that has data. Weight sum will be < 1.0."""
        import logging

        # Only model_A has data; model_B has weight assigned but no rows
        df_a = _make_model_df(TARGET_TIMES, "model_A", [100.0] * 4)
        # model_B: assign weights but don't provide any forecast rows
        weights = _make_weights_df(
            TARGET_TIMES,
            {"model_A": [0.5] * 4, "model_B": [0.5] * 4},
        )

        with caplog.at_level(logging.WARNING, logger="blend.blend"):
            result = blend_forecasts_together(df_a, weights)

        # All 4 target times should still produce rows (using model_A's data only)
        assert len(result) == 4

        # Warning about weights not summing to 1.0 should have been emitted
        assert any("sum to" in record.message for record in caplog.records), (
            "Expected a warning about blend weights not summing to 1.0"
        )

        # p50 for model_A only: 0.5 * 100.0 = 50.0. Since model_B is missing,
        # the weight sum is 0.5. The blended value is normalised to 50.0 / 0.5 = 100.0.
        result_indexed = result.set_index("target_time")
        for t in TARGET_TIMES:
            assert result_indexed.loc[
                t, "expected_power_generation_megawatts",
            ] == pytest.approx(100.0)

    def test_all_models_missing_data_produces_no_rows(self):
        """If no model has data at a target time, that time step is skipped entirely."""
        # Weights reference model_X and model_Y but we provide data for neither
        empty_df = pd.DataFrame(
            columns=[
                "target_time",
                "model_name",
                "expected_power_generation_megawatts",
            ],
        )
        weights = _make_weights_df(
            TARGET_TIMES,
            {"model_X": [0.5] * 4, "model_Y": [0.5] * 4},
        )

        result = blend_forecasts_together(empty_df, weights)
        assert result.empty

    def test_nan_p10_p90_treated_as_absent(self):
        """NaN p10/p90 for one model should not contribute to the blended p10/p90."""
        # model_A has valid p10/p90; model_B has NaN p10/p90
        df_a = _make_model_df(TARGET_TIMES, "model_A", [50.0] * 4, [10.0] * 4, [90.0] * 4)
        nan4 = [float("nan")] * 4
        df_b = _make_model_df(TARGET_TIMES, "model_B", [50.0] * 4, nan4, nan4)
        models_df = pd.concat([df_a, df_b], ignore_index=True)

        weights = _make_weights_df(
            TARGET_TIMES,
            {"model_A": [0.5] * 4, "model_B": [0.5] * 4},
        )

        result = blend_forecasts_together(models_df, weights)
        result_indexed = result.set_index("target_time")

        for t in TARGET_TIMES:
            # Only model_A contributes p10/p90, with weight 0.5
            # Since only model_A contributes (weight_sum=0.5 for p10),
            # the blend is (0.5 * 10.0) / 0.5 = 10.0 (normalised).
            assert result_indexed.loc[t, "p10_mw"] == pytest.approx(10.0)
            assert result_indexed.loc[t, "p90_mw"] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Test: output format correctness
# ---------------------------------------------------------------------------

class TestOutputFormat:
    def test_output_columns_always_present(self):
        """Output DataFrame always has all four required columns."""
        models_df = _make_model_df(TARGET_TIMES, "model_A", [1.0] * 4)
        weights = _make_weights_df(TARGET_TIMES, {"model_A": [1.0] * 4})

        result = blend_forecasts_together(models_df, weights)

        assert "target_time" in result.columns
        assert "expected_power_generation_megawatts" in result.columns
        assert "p10_mw" in result.columns
        assert "p90_mw" in result.columns

    def test_p10_p90_nan_when_not_in_input(self):
        """p10_mw and p90_mw should be NaN when neither input model provides them."""
        models_df = _make_model_df(TARGET_TIMES, "model_A", [10.0, 20.0, 30.0, 40.0])
        weights = _make_weights_df(TARGET_TIMES, {"model_A": [1.0] * 4})

        result = blend_forecasts_together(models_df, weights)

        for _, row in result.iterrows():
            assert math.isnan(row["p10_mw"]), "p10_mw should be NaN when input has no p10"
            assert math.isnan(row["p90_mw"]), "p90_mw should be NaN when input has no p90"

    def test_weights_sum_to_one_correct_blend(self):
        """Weights that exactly sum to 1.0 produce numerically correct results."""
        # Use more than 2 models: 3-way blend
        df_a = _make_model_df(TARGET_TIMES, "model_A", [90.0] * 4)
        df_b = _make_model_df(TARGET_TIMES, "model_B", [60.0] * 4)
        df_c = _make_model_df(TARGET_TIMES, "model_C", [30.0] * 4)
        models_df = pd.concat([df_a, df_b, df_c], ignore_index=True)

        # Weights: A=0.5, B=0.3, C=0.2 — sums to 1.0
        weights = _make_weights_df(
            TARGET_TIMES,
            {
                "model_A": [0.5] * 4,
                "model_B": [0.3] * 4,
                "model_C": [0.2] * 4,
            },
        )

        result = blend_forecasts_together(models_df, weights)
        expected_p50 = 0.5 * 90.0 + 0.3 * 60.0 + 0.2 * 30.0  # = 45 + 18 + 6 = 69.0

        result_indexed = result.set_index("target_time")
        for t in TARGET_TIMES:
            assert result_indexed.loc[t, "expected_power_generation_megawatts"] == pytest.approx(
                expected_p50,
            )
