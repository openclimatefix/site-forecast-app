from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from site_forecast_app.blend.config import BlendConfig
from site_forecast_app.blend.weights import get_blend_weights

# ---------------------------------------------------------------------------
# Tests for get_blend_weights
# ---------------------------------------------------------------------------

def _national_models(config: BlendConfig) -> list[str]:
    """All national models (backup first), de-duplicated, order preserved."""
    return list(dict.fromkeys([config.backup_model, *config.national_candidate_models]))


@pytest.mark.asyncio
async def test_get_blend_weights_missing_init_times(blend_config: BlendConfig):
    """Verify weights fallback to penalty logic when models missing."""
    t0 = pd.Timestamp("2024-06-01 12:00", tz="UTC")
    location_uuid = "test-uuid"
    max_horizon = pd.Timedelta("48h")

    backup = blend_config.backup_model
    # The one candidate we will give a recent init time to.
    fresh_candidate = next(m for m in blend_config.national_candidate_models if m != backup)

    # Create MAE scorecard where every candidate is better than the backup.
    horizons = pd.timedelta_range("15min", "3h", freq="15min")
    df_mae = pd.DataFrame(
        {
            model: [5.0] * len(horizons) if model == backup else [1.0] * len(horizons)
            for model in _national_models(blend_config)
        },
        index=horizons,
    )

    # Mock extract_latest_nl_init_times to return an incomplete dict
    mock_client = AsyncMock()

    with patch(
        "site_forecast_app.blend.weights.fetch_latest_nl_init_times", new_callable=AsyncMock,
    ) as mock_fetch:
        # only fresh_candidate has a recent init_time
        mock_fetch.return_value = {
            fresh_candidate: pd.Timestamp("2024-06-01 11:30", tz="UTC"),
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
    assert len(weights_df) > 0 # At least 1 valid horizon after shifting

    # The backup gets delay=0 and the fresh candidate is cheaper, so both take weight.
    # Every other candidate is penalised out of the shifted scorecard entirely.
    assert fresh_candidate in weights_df.columns
    assert set(weights_df.columns) <= {fresh_candidate, backup}

    # The weight sum at the row should be close to 1.0
    weight_sum = weights_df.sum(axis=1)
    assert (weight_sum > 0.99).all() and (weight_sum < 1.01).all()


@pytest.mark.asyncio
async def test_get_blend_weights_all_fail(blend_config: BlendConfig):
    """Verify fallback when no initialisation times exist (everything falls back)."""
    t0 = pd.Timestamp("2024-06-01 12:00", tz="UTC")
    backup = blend_config.backup_model
    df_mae = pd.DataFrame(
        {model: [1.0] for model in _national_models(blend_config)},
        index=[pd.Timedelta("30min")],
    )

    with patch(
        "site_forecast_app.blend.weights.fetch_latest_nl_init_times", new_callable=AsyncMock,
    ) as mock_fetch:
        # No init times found -> candidates get the max penalty delay, backup gets 0
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
        # Every candidate is penalised out, so the backup takes the full weight.
        assert not weights_df.empty
        assert list(weights_df.columns) == [backup]
