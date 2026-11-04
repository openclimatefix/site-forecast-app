from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from site_forecast_app.blend.config import BlendConfig
from site_forecast_app.blend.weights import (
    calculate_optimal_blend_weights,
    get_blend_weights,
    make_avg_mae_func,
)

# ---------------------------------------------------------------------------
# Tests for get_blend_weights
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_blend_weights_missing_init_times(blend_config: BlendConfig):
    """Verify weights fallback to penalty logic when models missing."""
    t0 = pd.Timestamp("2024-06-01 12:00", tz="UTC")
    location_uuid = "test-uuid"
    max_horizon = pd.Timedelta("48h")

    # Create MAE scorecard where nl_regional_ecmwf_only is better than nl_regional_pv_ecmwf_mo_sat
    horizons = [pd.Timedelta("30min"), pd.Timedelta("1h")]
    df_mae = pd.DataFrame(
        {
            "nl_regional_ecmwf_only": [1.0, 1.5],
            "nl_regional_pv_ecmwf_mo_sat": [2.0, 2.5],
            "backup_blend": [5.0, 5.0],
        },
        index=horizons,
    )

    # Mock extract_latest_nl_init_times to return an incomplete dict
    mock_client = AsyncMock()

    with patch(
        "site_forecast_app.blend.weights.fetch_latest_nl_init_times", new_callable=AsyncMock,
    ) as mock_fetch:
        # only nl_regional_ecmwf_only has a recent init_time
        mock_fetch.return_value = {
            "nl_regional_ecmwf_only": pd.Timestamp("2024-06-01 11:30", tz="UTC"),
        }

        weights_df = await get_blend_weights(
            t0=t0,
            location_uuid=location_uuid,
            df_mae=df_mae,
            max_horizon=max_horizon,
            client=mock_client,
            config=blend_config,
        )

    assert weights_df is not None
    assert "nl_regional_ecmwf_only" in weights_df.columns
    # nl_regional_pv_ecmwf_mo_sat has penalty and is worse, so maybe only backup shows up.
    # Check that at least the fallback is there.
    assert len(weights_df) > 0 # At least 1 valid horizon after shifting

    # The weight sum at the row should be close to 1.0
    weight_sum = weights_df.sum(axis=1)
    assert (weight_sum > 0.99).all() and (weight_sum < 1.01).all()

@pytest.mark.asyncio
async def test_get_blend_weights_all_fail(blend_config: BlendConfig):
    """Verify fallback when no initialisation times exist (everything falls back)."""
    t0 = pd.Timestamp("2024-06-01 12:00", tz="UTC")
    df_mae = pd.DataFrame({"nl_regional_ecmwf_only": [1.0]}, index=[pd.Timedelta("30min")])

    with patch(
        "site_forecast_app.blend.weights.fetch_latest_nl_init_times", new_callable=AsyncMock,
    ) as mock_fetch:
        # No init times found -> delays are 1000 days
        mock_fetch.return_value = {}

        weights_df = await get_blend_weights(
            t0=t0,
            location_uuid="u",
            df_mae=df_mae,
            max_horizon=pd.Timedelta("30min"),
            client=AsyncMock(),
            config=blend_config,
        )
        assert len(weights_df) == 1
        # It still computes weights evenly or heavily shifts since they both have huge penalties.
        # Just ensure the DF isn't empty.
        assert not weights_df.empty

@pytest.mark.asyncio
async def test_get_blend_weights_intraday_then_day_ahead(blend_config: BlendConfig):
    """Verify intraday blending is applied on top of the day-ahead blend."""
    config = blend_config.model_copy(
        update={"intraday_candidate_models": ["fake_intraday"]},
    )

    t0 = pd.Timestamp("2024-06-01 12:00", tz="UTC")
    max_horizon = pd.Timedelta("36h")

    horizons = pd.timedelta_range(
        start="15min",
        end="36h",
        freq="15min",
    )

    df_mae = pd.DataFrame(
        {
            "nl_regional_ecmwf_only": 5.0,
            "nl_regional_48h_pv_ecmwf": [
                5.0 if horizon <= pd.Timedelta(hours=8) else 2.0
                for horizon in horizons
            ],
            "fake_intraday": [
                1.0 if horizon <= pd.Timedelta(hours=8) else np.nan
                for horizon in horizons
            ],
        },
        index=horizons,
    )

    with patch(
        "site_forecast_app.blend.weights.fetch_latest_nl_init_times",
        new_callable=AsyncMock,
    ) as mock_fetch:
        mock_fetch.return_value = {
            "nl_regional_ecmwf_only": t0,
            "nl_regional_48h_pv_ecmwf": t0,
            "fake_intraday": t0,
        }
        weights_df = await get_blend_weights(
            t0=t0,
            location_uuid="test-uuid",
            df_mae=df_mae,
            max_horizon=max_horizon,
            client=AsyncMock(),
            config=config,
        )
        # Intraday model dominates before the taper.
        assert weights_df.loc["2024-06-01 19:00:00+00:00", "fake_intraday"] == pytest.approx(1.0)

        assert (
            weights_df.loc[
                "2024-06-01 19:45:00+00:00",
                "nl_regional_48h_pv_ecmwf",
            ]
            == pytest.approx(0.5)
        )

        assert (
            weights_df.loc[
                "2024-06-01 20:15:00+00:00",
                "nl_regional_48h_pv_ecmwf",
            ]
            == pytest.approx(1.0)
        )
        assert pd.isna(
            weights_df.loc[
                "2024-06-01 20:15:00+00:00",
                "fake_intraday",
            ],
        )
def test_candidate_unavailable_falls_back_to_backup():
    horizons = pd.timedelta_range(
        start="15min",
        periods=40,
        freq="15min",
    )
    df_mae = pd.DataFrame(
        {
            "candidate": [
                1.0 if horizon <= pd.Timedelta(hours=8) else np.nan
                for horizon in horizons
            ],
            "backup": [5.0] * len(horizons),
        },
        index=horizons,
    )

    weights = calculate_optimal_blend_weights(
        df_mae=df_mae,
        backup_model_name="backup",
        candidate_models=["candidate"],
        kernel=[0.75, 0.5, 0.25],
        score_func=make_avg_mae_func(
            pd.Timedelta(hours=8),
            pd.Timedelta(minutes=15),
        ),
        min_forecast_horizon=pd.Timedelta(minutes=15),
    )

    after_eight_hours = weights.index > pd.Timedelta(hours=8)

    assert weights.loc[after_eight_hours, "candidate"].isna().all()
    assert (weights.loc[after_eight_hours, "backup"] == 1.0).all()
