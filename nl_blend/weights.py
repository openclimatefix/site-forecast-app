"""Logic for calculating optimal model weights for NL site blending."""
import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from nl_blend.data_platform import fetch_latest_nl_init_times
from nl_blend.init_times import calculate_model_delays, shift_mae_curves
from site_forecast_app.save.data_platform import get_dataplatform_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants - NL model registry
# ---------------------------------------------------------------------------

# The absolute fallback: always available, longest horizon, weakest accuracy
NL_BACKUP_MODEL = "nl_regional_2h_pv_ecmwf"

# Day-ahead model(s): 48-hour horizon, NWP-driven, no satellite
NL_DAY_AHEAD_MODELS = ["nl_regional_48h_pv_ecmwf"]

# Intraday models: shorter horizon but lower MAE while fresh
NL_INTRADAY_MODELS = [
    "nl_regional_pv_ecmwf_mo_sat",
    "nl_regional_pv_ecmwf_sat",
    "nl_national_pv_ecmwf_sat_small",
]

ALL_NL_MODELS = [NL_BACKUP_MODEL, *NL_DAY_AHEAD_MODELS, *NL_INTRADAY_MODELS]

# Blend kernel: weights applied at the transition zone between two models.
# [1.0 primary, 0.75 primary, 0.5 primary, 0.25 primary] then 0.0 (backup takes over).
# Matches the UK tapering kernel to avoid abrupt model switches.
BLEND_KERNEL: list[float] = [0.75, 0.5, 0.25]

# Minimum horizon emitted in any blended forecast
MIN_FORECAST_HORIZON = pd.Timedelta("30min")

# Score-function horizons - matches the UK horizon window choices:
#   Stage 1 (day-ahead selection):  optimise over 36 h  (NL ECMWF covers 48 h)
#   Stage 2 (intraday selection):   optimise over  8 h  (satellite models ~6-8 h)
_STAGE1_SCORE_HOURS = 36
_STAGE2_SCORE_HOURS = 8


# ---------------------------------------------------------------------------
# Score function factory
# ---------------------------------------------------------------------------

def make_avg_mae_func(n_hours: int) -> Callable[[pd.Series], float]:
    """Returns a scoring function for MAE over a window.

    Computes the mean MAE over [MIN_FORECAST_HORIZON, n_hours], excluding the
    final boundary point (matching the UK half-open interval convention).
    """
    def _score(horizon_mae: pd.Series) -> float:
        window = horizon_mae.loc[MIN_FORECAST_HORIZON : f"{n_hours}h"]
        # Drop the last boundary point to keep intervals half-open, as in UK
        return float(window.iloc[:-1].mean())

    _score.__name__ = f"avg_mae_{n_hours}h"
    return _score


# ---------------------------------------------------------------------------
# Weight array helpers
# ---------------------------------------------------------------------------

def make_blend_weights_array(
    size: int,
    blend_start_index: int,
    kernel: list[float],
) -> np.ndarray:
    """Constructs a 1-D weight array for the primary (non-backup) model.

    Layout
    ------
    Indices  0 … blend_start_index-1          → 1.0  (primary dominates)
    Indices  blend_start_index … +len(kernel)  → kernel values (tapering)
    Indices  beyond kernel                     → 0.0  (backup dominates)

    The complementary backup weights are (1 - this array).
    """
    weights = np.zeros(size)
    weights[:blend_start_index] = 1.0
    weights[blend_start_index : blend_start_index + len(kernel)] = kernel
    return weights


def index_of_last_non_nan_value(x: np.ndarray) -> int:
    """Returns the index of the last non-NaN element in x.

    Returns -1 if x is entirely NaN, which causes the calling loop to produce
    an empty range and silently skip the model - the same outcome as the UK
    implementation when a model has no valid horizon coverage.
    """
    non_nan_indices = np.where(~np.isnan(x))[0]
    if len(non_nan_indices) == 0:
        return -1
    return int(non_nan_indices[-1])


# ---------------------------------------------------------------------------
# Core weight optimiser
# ---------------------------------------------------------------------------

def calculate_optimal_blend_weights(
    df_mae: pd.DataFrame,
    backup_model_name: str,
    kernel: list[float],
    score_func: Callable[[pd.Series], float],
) -> pd.DataFrame:
    """Finds the blend weight array that minimises score_func.

    Minimises score_func when mixing one candidate model with the backup model.
    ------------------------------------------
    1. Start with the backup model scoring as the baseline.
    2. For each candidate model:
       a. If the candidate's horizon coverage is >= the backup's, try using it
          at weight=1 across all horizons.
       b. Otherwise, sweep every valid blend_start_index and try the kernel
          taper at each position.
       c. Keep the (model, weights) pair that gives the lowest score.
    3. If no candidate beats the backup, return weights=1 for the backup.
    4. Otherwise return weights for the winner and its complementary backup.

    NaN positions are preserved in the output so downstream code can detect
    which horizons a given model genuinely covers.

    Args:
        df_mae:            Shifted (horizon x model) MAE DataFrame.
        backup_model_name: Column name of the fallback model.
        kernel:            Taper kernel list (e.g. [0.75, 0.5, 0.25]).
        score_func:        Callable(pd.Series) -> float, lower is better.

    Returns:
        Wide DataFrame (same index as df_mae) with one column per participating
        model, values are blend weights in [0, 1] or NaN where coverage is absent.
    """
    if df_mae.empty:
        return pd.DataFrame(columns=df_mae.columns)

    if backup_model_name not in df_mae.columns:
        raise ValueError(
            f"backup_model_name='{backup_model_name}' not in df_mae columns: "
            f"{list(df_mae.columns)}",
        )

    kernel_arr = np.array(kernel)

    # Fill NaN with a large penalty value so the score function always prefers
    # a model with real data over one with gaps - matches the UK fill strategy.
    fill_val = np.nanmax(df_mae.values) * 10
    df_filled = df_mae.fillna(fill_val)

    baseline_score = score_func(df_filled[backup_model_name])
    logger.debug(f"Baseline score ({backup_model_name}): {baseline_score}")

    best_score = baseline_score
    best_model = backup_model_name
    best_weights: np.ndarray | None = None

    backup_last_idx = index_of_last_non_nan_value(df_mae[backup_model_name].values)

    for model in [c for c in df_mae.columns if c != backup_model_name]:
        model_last_idx = index_of_last_non_nan_value(df_mae[model].values)

        if model_last_idx < 0:
            # Model has no valid data at all - skip
            logger.debug(f"Model {model} has no valid data at all - skip.")
            continue

        if model_last_idx >= backup_last_idx:
            # Model covers at least as much horizon as the backup - use it fully
            candidate_weights = np.ones(len(df_mae))
            candidate_mae = (
                candidate_weights * df_filled[model]
                + (1 - candidate_weights) * df_filled[backup_model_name]
            )
            score = score_func(candidate_mae)
            logger.debug(
                f"Model {model} full coverage score: {score:.5f} "
                f"(baseline: {baseline_score:.5f})",
            )
            if score < best_score:
                best_score = score
                best_weights = candidate_weights
                best_model = model
        else:
            # Model has shorter coverage - sweep the taper start position
            max_blend_start = model_last_idx - len(kernel_arr) + 1
            for position in range(max_blend_start + 1):
                if df_mae.index[position] < MIN_FORECAST_HORIZON:
                    continue
                candidate_weights = make_blend_weights_array(
                    len(df_mae), position, kernel_arr,
                )
                candidate_mae = (
                    candidate_weights * df_filled[model]
                    + (1 - candidate_weights) * df_filled[backup_model_name]
                )
                score = score_func(candidate_mae)
                logger.debug(f"Model {model} at taper position {position} score: {score:.5f}")
                if score < best_score:
                    best_score = score
                    best_weights = candidate_weights
                    best_model = model

    # Build the result DataFrame
    if best_model == backup_model_name:
        # Backup wins: it takes full weight everywhere it has coverage
        w_backup = np.ones(len(df_mae))
        w_backup[df_mae[backup_model_name].isna()] = np.nan
        return pd.DataFrame({backup_model_name: w_backup}, index=df_mae.index)

    # A challenger won: split weights between winner and backup
    w_best = best_weights.copy()
    w_backup = 1.0 - best_weights

    # Mask out horizons where the respective model has no data
    w_best[df_mae[best_model].isna()] = np.nan
    w_backup[df_mae[backup_model_name].isna()] = np.nan

    return pd.DataFrame(
        {best_model: w_best, backup_model_name: w_backup},
        index=df_mae.index,
    )


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------

async def get_nl_blend_weights(
    t0: pd.Timestamp,
    location_uuid: str,
    df_mae: pd.DataFrame,
    max_horizon: pd.Timedelta,
) -> pd.DataFrame:
    """Produces the final blend weight DataFrame for t0, matching the UK two-stage cascade.

    Stage 1 - Day-ahead selection
        Choose the best day-ahead model (or keep the backup) by minimising
        average MAE over a 36-hour window. Produces an intermediate blend
        called 'stage1_blend'.

    Stage 2 - Intraday injection
        Choose the best intraday model (or keep stage1_blend) by minimising
        average MAE over an 8-hour window.

    Stage 1 weights are then scaled by how much of stage1_blend survives into
    the final stage 2 output, so every row of the returned DataFrame sums to 1
    (or NaN where all models lack coverage).

    The returned index is shifted from relative horizon (pd.Timedelta) to
    absolute UTC target times (t0 + horizon), ready for direct look-up in
    blend_forecasts_together.

    Args:
        t0:            Blend reference time (UTC, floor to 30 min).
        location_uuid: Data Platform location UUID.
        df_mae:        (horizon x model) MAE scorecard from load_nl_mae_scorecard.
        max_horizon:   Maximum horizon in the scorecard; used as the max_delay
                       cutoff when fetching init times.

    Returns:
        Wide DataFrame indexed by absolute UTC target time, columns = model names.
    """
    # ------------------------------------------------------------------ #
    # 1. Fetch model initialisation times from Data Platform              #
    # ------------------------------------------------------------------ #
    async with get_dataplatform_client() as client:
        model_init_times = await fetch_latest_nl_init_times(
            client=client,
            location_uuid=location_uuid,
            model_names=ALL_NL_MODELS,
            t0=t0,
            max_delay=max_horizon,
        )

    logger.info(f"Fetched model initialisation times: {model_init_times}")

    # ------------------------------------------------------------------ #
    # 2. Assign a penalty delay to any model not found in DP              #
    #    (max_horizon → effectively excluded from the blend)              #
    #    Matches the UK behaviour of silently falling back rather than    #
    #    erroring when individual models are absent.                       #
    # ------------------------------------------------------------------ #
    missing = [m for m in ALL_NL_MODELS if m not in model_init_times]
    if missing:
        logger.info(
            f"No init time found for {missing}; assigning max_horizon delay "
            f"so they are excluded from the blend.",
        )
    for m in missing:
        model_init_times[m] = t0 - max_horizon

    # ------------------------------------------------------------------ #
    # 3. Compute numeric delays and shift the MAE scorecard               #
    # ------------------------------------------------------------------ #
    delays = calculate_model_delays(model_init_times, t0)
    logger.info(f"Computed model delays relative to t0 ({t0}): {delays}")
    df_delayed_mae = shift_mae_curves(df_mae, delays)

    if df_delayed_mae.empty:
        logger.error(
            "Shifted MAE DataFrame is empty - no model delays overlap with the "
            "scorecard. Cannot produce blend weights.",
        )
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # 4. Stage 1: blend day-ahead model into backup                       #
    # ------------------------------------------------------------------ #
    stage1_cols = [NL_BACKUP_MODEL, *NL_DAY_AHEAD_MODELS]
    # Keep only columns that survived the delay-shift
    stage1_cols = [c for c in stage1_cols if c in df_delayed_mae.columns]

    df_stage1_weights = calculate_optimal_blend_weights(
        df_mae=df_delayed_mae[stage1_cols],
        backup_model_name=NL_BACKUP_MODEL,
        kernel=BLEND_KERNEL,
        score_func=make_avg_mae_func(_STAGE1_SCORE_HOURS),
    )

    logger.info(f"Stage 1 weights head:\n{df_stage1_weights.head()}")

    # Derive the effective MAE of the stage-1 blend so stage 2 can compare
    # intraday models against it.  A row is NaN only if ALL stage-1 models
    # are NaN at that horizon.
    mask_stage1 = ~df_delayed_mae[stage1_cols].isnull().all(axis=1)
    df_delayed_mae["stage1_blend"] = (
        df_stage1_weights.reindex(columns=stage1_cols).fillna(0)
        * df_delayed_mae[stage1_cols]
    ).sum(axis=1, skipna=True).where(mask_stage1)

    # ------------------------------------------------------------------ #
    # 5. Stage 2: blend intraday model into stage-1 result                #
    # ------------------------------------------------------------------ #
    stage2_cols = [*NL_INTRADAY_MODELS, "stage1_blend"]
    stage2_cols = [c for c in stage2_cols if c in df_delayed_mae.columns]

    df_stage2_weights = calculate_optimal_blend_weights(
        df_mae=df_delayed_mae[stage2_cols],
        backup_model_name="stage1_blend",
        kernel=BLEND_KERNEL,
        score_func=make_avg_mae_func(_STAGE2_SCORE_HOURS),
    )

    logger.info(f"Stage 2 weights head:\n{df_stage2_weights.head()}")

    # ------------------------------------------------------------------ #
    # 6. Combine stage weights                                             #
    #    Stage-1 weights must be scaled by how much stage2 still uses     #
    #    'stage1_blend' so the final weights for every row sum to 1.      #
    # ------------------------------------------------------------------ #
    stage1_passthrough = df_stage2_weights["stage1_blend"]

    # Use multiply to avoid in-place mutation of df_stage1_weights
    df_stage1_scaled = df_stage1_weights.multiply(stage1_passthrough, axis=0)

    df_stage2_models_only = df_stage2_weights.drop(columns="stage1_blend")

    weights_df = pd.concat([df_stage1_scaled, df_stage2_models_only], axis=1)

    # ------------------------------------------------------------------ #
    # 7. Drop columns that contribute zero weight across all horizons     #
    # ------------------------------------------------------------------ #
    weights_df = weights_df.loc[:, weights_df.sum(axis=0, skipna=True) > 0]

    # ------------------------------------------------------------------ #
    # 8. Convert relative-horizon index → absolute UTC target times       #
    # ------------------------------------------------------------------ #
    weights_df.index = weights_df.index + t0

    logger.info(
        f"Blend weights computed for {len(weights_df)} target times, "
        f"participating models: {list(weights_df.columns)}",
    )

    return weights_df
