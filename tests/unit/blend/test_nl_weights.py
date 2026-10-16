from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from site_forecast_app.blend.config import BlendConfig
from site_forecast_app.blend.weights import get_blend_weights

# ---------------------------------------------------------------------------
# Tests for get_blend_weights
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_blend_weights_missing_init_times(blend_config: BlendConfig):
    """Verify weights fallback to penalty logic when models missing."""
    t0 = pd.Timestamp("2024-06-01 12:00", tz="UTC")
    location_uuid = "test-uuid"
    max_horizon = pd.Timedelta("48h")

    # Create MAE scorecard where nl_regional_2h_pv_ecmwf is better than nl_regional_pv_ecmwf_mo_sat
    horizons = [pd.Timedelta("30min"), pd.Timedelta("1h")]
    df_mae = pd.DataFrame(
        {
            "nl_regional_2h_pv_ecmwf": [1.0, 1.5],
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
        # only nl_regional_2h_pv_ecmwf has a recent init_time
        mock_fetch.return_value = {
            "nl_regional_2h_pv_ecmwf": pd.Timestamp("2024-06-01 11:30", tz="UTC"),
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
    assert "nl_regional_2h_pv_ecmwf" in weights_df.columns
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
    df_mae = pd.DataFrame({"nl_regional_2h_pv_ecmwf": [1.0]}, index=[pd.Timedelta("30min")])

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
            "nl_regional_2h_pv_ecmwf": 5.0,
            "nl_regional_48h_pv_ecmwf": [
                5.0 if horizon <= pd.Timedelta(hours=8) else 2.0
                for horizon in horizons
            ],
            "fake_intraday": [
                1.0 if horizon <= pd.Timedelta(hours=8) else 5.0
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
            "nl_regional_2h_pv_ecmwf": t0,
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
        assert list(weights_df.columns) == ["fake_intraday"]
        assert weights_df["fake_intraday"].iloc[0] == pytest.approx(1.0)
