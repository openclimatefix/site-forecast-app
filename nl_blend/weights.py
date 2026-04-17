"""Logic for calculating optimal model weights for NL site blending.

Mirrors the two-stage hierarchical approach used in the UK blend:

  Stage 1 - Select the single best day-ahead model to blend against the backup
            (NL_BACKUP_MODEL). Scored over the first 36 forecast hours.
            Equivalent to the UK blend's National_xg / pvnet_day_ahead stage.

  Stage 2 - Select the single best intraday model to blend against the
            stage-1 day-ahead blend. Scored over the first 8 forecast hours.
            Equivalent to the UK blend's intraday stage.

At any given horizon, at most two models are active (one day-ahead + one
intraday), and their weights always sum to 1.0.

The final weight DataFrame has one column per participating model and the
index is absolute UTC target times (t0 + horizon), ready for direct look-up
in blend_forecasts_together.
"""
import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from dp_sdk.ocf import dp

from nl_blend.data_platform import fetch_latest_nl_init_times
from nl_blend.init_times import calculate_model_delays, shift_mae_curves

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants - NL model registry
# ---------------------------------------------------------------------------

# The absolute fallback: always available, longest horizon, weakest accuracy.
# Equivalent of National_xg in the UK blend.
NL_BACKUP_MODEL = "nl_regional_2h_pv_ecmwf"

# Day-ahead models: longer horizon, moderate near-term accuracy.
# Equivalent of pvnet_day_ahead in the UK blend.
NL_DAY_AHEAD_MODELS = [
    "nl_regional_48h_pv_ecmwf",
]

# Intraday models: shorter horizon, highest near-term accuracy.
# Equivalent of pvnet_v2 / pvnet_ecmwf / pvnet_cloud in the UK blend.
NL_INTRADAY_MODELS = [
    "nl_regional_pv_ecmwf_mo_sat",
    "nl_regional_pv_ecmwf_sat",
    "nl_national_pv_ecmwf_sat_small",
]

ALL_NL_MODELS = [NL_BACKUP_MODEL, *NL_DAY_AHEAD_MODELS, *NL_INTRADAY_MODELS]

# Blend kernel: weights applied at the taper zone between two models.
# [0.75, 0.5, 0.25] avoids abrupt model switches - identical to the UK blend.
BLEND_KERNEL: list[float] = [0.75, 0.5, 0.25]

# Minimum horizon emitted in any blended forecast.
MIN_FORECAST_HORIZON = pd.Timedelta("30min")

# Score windows - mirror the UK blend score windows exactly:
#   36 h  used when selecting the best day-ahead model vs backup
#    8 h  used when selecting the best intraday model vs the da-blend
DA_SCORE_HOURS = 36
INTRADAY_SCORE_HOURS = 8


# ---------------------------------------------------------------------------
# Score function factory
# ---------------------------------------------------------------------------

def make_avg_mae_func(n_hours: int) -> Callable[[pd.Series], float]:
    """Returns a scoring function that averages MAE over the first n_hours.

    Computes the mean MAE over [MIN_FORECAST_HORIZON, n_hours], excluding the
    final boundary point (half-open interval, matching the UK blend).

    Args:
        n_hours: The number of hours to average over.

    Returns:
        A callable that takes a horizon-MAE Series and returns a float score.
    """
    def _score(horizon_mae: pd.Series) -> float:
        window = horizon_mae.loc[MIN_FORECAST_HORIZON : f"{n_hours}h"]
        # Drop the last boundary point to keep intervals half-open.
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
    Indices  0 ... blend_start_index-1          -> 1.0  (primary dominates)
    Indices  blend_start_index ... +len(kernel)  -> kernel values (tapering)
    Indices  beyond kernel                       -> 0.0  (backup dominates)

    The complementary backup weights are (1 - this array).

    Examples:
    --------
    >>> make_blend_weights_array(8, 1, [0.75, 0.5, 0.25])
    array([1.  , 0.75, 0.5 , 0.25, 0.  , 0.  , 0.  , 0.  ])

    >>> make_blend_weights_array(6, 1, [])
    array([1., 0., 0., 0., 0., 0.])
    """
    weights = np.zeros(size)
    weights[:blend_start_index] = 1.0
    weights[blend_start_index : blend_start_index + len(kernel)] = kernel
    return weights


def index_of_last_non_nan_value(x: np.ndarray) -> int:
    """Returns the index of the last non-NaN element in x.

    Returns -1 if x is entirely NaN, which causes the calling loop to produce
    an empty range and silently skip the model - matching the UK blend
    behaviour when a model has no valid horizon coverage.
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
    """Selects the single best model to blend against the backup and returns weights.

    Mirrors the UK blend algorithm exactly - only the ONE best candidate is
    selected, not all candidates that beat the baseline. At any horizon, at
    most two models are active (the best candidate + the backup).

    Algorithm
    ---------
    1. Establish a baseline score using the backup model alone.
    2. For each candidate model, sweep all possible taper-start positions and
       find the weight array that minimises score_func.
    3. Keep only the single candidate with the best (lowest) score. If no
       candidate beats the baseline, return backup-only weights.
    4. Return a two-column DataFrame [best_model, backup_model] where the
       primary weights are the sweep result and backup weights = 1 - primary.
       Weights sum to 1.0 at every horizon.

    Args:
        df_mae:            Shifted (horizon x model) MAE DataFrame. NaN values
                           indicate the model has no coverage at that horizon.
        backup_model_name: Column name of the fallback/backup model. Must be
                           present in df_mae.
        kernel:            Taper kernel (e.g. [0.75, 0.5, 0.25]). Values must
                           be strictly between 0 and 1 and non-increasing.
        score_func:        Callable(pd.Series) -> float, lower is better.

    Returns:
        DataFrame (same index as df_mae) with either:
          - One column  (backup only) when no candidate beats the baseline, or
          - Two columns (best_model + backup) when a candidate is selected.
        Weights sum to 1.0 at every non-NaN horizon.
    """
    if df_mae.empty:
        return pd.DataFrame(columns=df_mae.columns)

    if backup_model_name not in df_mae.columns:
        raise ValueError(
            f"backup_model_name='{backup_model_name}' not found in df_mae columns: "
            f"{list(df_mae.columns)}",
        )

    kernel_arr = np.array(kernel)
    if not (kernel_arr > 0).all():
        raise ValueError("All kernel values must be > 0")
    if not (kernel_arr < 1).all():
        raise ValueError("All kernel values must be < 1")
    if not (np.diff(kernel_arr) <= 0).all():
        raise ValueError("Kernel must be non-increasing")

    n = len(df_mae)

    # Fill NaN with a large penalty value so the score function always prefers
    # a model with real data over one with gaps at important horizons.
    fill_val = np.nanmax(df_mae.values) * 10
    df_filled = df_mae.fillna(fill_val)

    # Baseline: backup model used alone.
    best_score = score_func(df_filled[backup_model_name])
    best_model = backup_model_name
    best_weights: np.ndarray | None = None

    backup_last_idx = index_of_last_non_nan_value(df_mae[backup_model_name].values)

    for model in [c for c in df_mae.columns if c != backup_model_name]:

        last_non_nan_idx = index_of_last_non_nan_value(df_mae[model].values)

        if last_non_nan_idx < 0:
            # Model has no valid data at any horizon - skip entirely.
            logger.debug(f"Model '{model}' has no valid MAE data - skipping.")
            continue

        if last_non_nan_idx >= backup_last_idx:
            # Candidate covers at least as much horizon as the backup:
            # evaluate using it at full weight across all horizons.
            candidate_weights = np.ones(n)
            candidate_mae = (
                candidate_weights * df_filled[model]
                + (1 - candidate_weights) * df_filled[backup_model_name]
            )
            score = score_func(candidate_mae)

            if score < best_score:
                best_score = score
                best_weights = candidate_weights
                best_model = model

        else:
            # Candidate has a shorter horizon than the backup: sweep all valid
            # taper-start positions and pick the one with the lowest score.
            max_blend_start_pos = last_non_nan_idx - len(kernel_arr) + 1
            for position in range(max_blend_start_pos + 1):

                # Never start the blend before the minimum forecast horizon.
                if df_mae.index[position] < MIN_FORECAST_HORIZON:
                    continue

                candidate_weights = make_blend_weights_array(
                    size=n,
                    blend_start_index=position,
                    kernel=kernel_arr,
                )
                candidate_mae = (
                    candidate_weights * df_filled[model]
                    + (1 - candidate_weights) * df_filled[backup_model_name]
                )
                score = score_func(candidate_mae)

                if score < best_score:
                    best_score = score
                    best_weights = candidate_weights
                    best_model = model

    # ---------------------------------------------------------------------- #
    # Build the output weight DataFrame                                       #
    # ---------------------------------------------------------------------- #
    if best_model == backup_model_name:
        # No candidate improved on the backup - return backup-only weights.
        backup_weights = np.ones(n)
        backup_weights[df_mae[backup_model_name].isna()] = np.nan
        logger.debug(
            f"No candidate beat baseline score {best_score:.5f}; "
            f"'{backup_model_name}' takes full weight.",
        )
        return pd.DataFrame(
            {backup_model_name: backup_weights},
            index=df_mae.index,
        )

    # best_weights are the primary-model weights; backup gets the complement.
    backup_weights = 1 - best_weights
    # NaN-mask at horizons where each model has no scorecard coverage.
    best_weights[df_mae[best_model].isna()] = np.nan
    backup_weights[df_mae[backup_model_name].isna()] = np.nan

    logger.debug(
        f"Selected '{best_model}' as best candidate "
        f"(score {best_score:.5f} < baseline).",
    )
    return pd.DataFrame(
        {best_model: best_weights, backup_model_name: backup_weights},
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
    client: dp.DataPlatformDataServiceStub,
) -> pd.DataFrame:
    """Produces the final blend weight DataFrame for t0.

    Runs the same two-stage hierarchical optimisation as the UK blend:

      Stage 1 - Find the best day-ahead model to blend against NL_BACKUP_MODEL.
                Uses DA_SCORE_HOURS (36 h) as the optimisation window.
                Constructs a 'da_blend' MAE curve from the result.

      Stage 2 - Find the best intraday model to blend against the da_blend.
                Uses INTRADAY_SCORE_HOURS (8 h) as the optimisation window.

    The day-ahead model weights are then scaled by how much the da_blend
    contributes at each horizon (from stage 2), so the final weights across
    all models always sum to 1.0.

    The returned index is shifted from relative horizon (pd.Timedelta) to
    absolute UTC target times (t0 + horizon), ready for direct look-up in
    blend_forecasts_together.

    Args:
        t0:            Blend reference time (UTC, floored to 30 min).
        location_uuid: Data Platform location UUID.
        df_mae:        (horizon x model) MAE scorecard from load_nl_mae_scorecard.
        max_horizon:   Maximum horizon in the scorecard; used as the max_delay
                       cutoff when fetching init times.
        client:        Authenticated Data Platform gRPC client stub.

    Returns:
        Wide DataFrame indexed by absolute UTC target time, one column per
        participating model. Weights sum to 1.0 at every horizon.
        Returns an empty DataFrame if the shifted MAE frame is empty.
    """
    # ---------------------------------------------------------------------- #
    # 1. Fetch model initialisation times from Data Platform                 #
    # ---------------------------------------------------------------------- #
    model_init_times = await fetch_latest_nl_init_times(
        client=client,
        location_uuid=location_uuid,
        model_names=ALL_NL_MODELS,
        t0=t0,
        max_delay=max_horizon,
    )
    logger.info(f"Fetched model initialisation times: {model_init_times}")

    # ---------------------------------------------------------------------- #
    # 2. Assign penalty delays to any model not found in Data Platform       #
    #    Backup  -> delay = 0          (always used as fallback)             #
    #    Others  -> delay = max_horizon (effectively excluded)               #
    # ---------------------------------------------------------------------- #
    missing = [m for m in ALL_NL_MODELS if m not in model_init_times]
    if missing:
        logger.info(f"No init time found for {missing}; assigning penalty delays.")
    for m in missing:
        if m == NL_BACKUP_MODEL:
            model_init_times[m] = t0
        else:
            model_init_times[m] = t0 - max_horizon

    # ---------------------------------------------------------------------- #
    # 3. Compute numeric delays and shift the MAE scorecard                  #
    # ---------------------------------------------------------------------- #
    delays = calculate_model_delays(model_init_times, t0)
    logger.info(f"Computed model delays relative to t0 ({t0}): {delays}")

    df_delayed_mae = shift_mae_curves(df_mae, delays)

    if df_delayed_mae.empty:
        logger.error(
            "Shifted MAE DataFrame is empty - no model delays overlap with the "
            "scorecard. Cannot produce blend weights.",
        )
        return pd.DataFrame()

    # ---------------------------------------------------------------------- #
    # 4. Stage 1 - select best day-ahead model vs backup                     #
    #    Mirrors: UK calculate_optimal_blend_weights(                        #
    #                 backup_model_name="National_xg",                       #
    #                 score_func=make_avg_mae_func(36))                      #
    # ---------------------------------------------------------------------- #
    da_candidate_cols = [
        c for c in [NL_BACKUP_MODEL, *NL_DAY_AHEAD_MODELS]
        if c in df_delayed_mae.columns
    ]
    logger.info(f"Stage 1 - day-ahead candidates: {da_candidate_cols}")

    df_da_weights = calculate_optimal_blend_weights(
        df_mae=df_delayed_mae[da_candidate_cols],
        backup_model_name=NL_BACKUP_MODEL,
        kernel=BLEND_KERNEL,
        score_func=make_avg_mae_func(DA_SCORE_HOURS),
    )
    logger.info(f"Stage 1 weights (head):\n{df_da_weights.head()}")

    # ---------------------------------------------------------------------- #
    # 5. Compute the expected MAE of the stage-1 day-ahead blend             #
    #    This becomes the 'da_blend' column used as stage-2 backup.          #
    #    Mirrors: UK blend's df_delayed_mae["da_blend"] construction.        #
    # ---------------------------------------------------------------------- #
    # Rows where at least one DA column has a valid (non-NaN) weight.
    valid_mask = ~df_delayed_mae[da_candidate_cols].isnull().all(axis=1)

    df_delayed_mae["da_blend"] = (
        df_da_weights.fillna(0) * df_delayed_mae[da_candidate_cols]
    ).sum(skipna=True, axis=1).where(valid_mask)

    # ---------------------------------------------------------------------- #
    # 6. Stage 2 - select best intraday model vs the da_blend                #
    #    Mirrors: UK calculate_optimal_blend_weights(                        #
    #                 backup_model_name="da_blend",                          #
    #                 score_func=make_avg_mae_func(8))                       #
    # ---------------------------------------------------------------------- #
    intraday_candidate_cols = [
        c for c in [*NL_INTRADAY_MODELS, "da_blend"]
        if c in df_delayed_mae.columns
    ]
    logger.info(f"Stage 2 - intraday candidates: {intraday_candidate_cols}")

    df_intraday_weights = calculate_optimal_blend_weights(
        df_mae=df_delayed_mae[intraday_candidate_cols],
        backup_model_name="da_blend",
        kernel=BLEND_KERNEL,
        score_func=make_avg_mae_func(INTRADAY_SCORE_HOURS),
    )
    logger.info(f"Stage 2 weights (head):\n{df_intraday_weights.head()}")

    # ---------------------------------------------------------------------- #
    # 7. Scale day-ahead weights by the da_blend contribution from stage 2  #
    #    Mirrors: UK blend's                                                 #
    #        df_da_model_weights[col] *= intraday_weights["da_blend"]        #
    #                                                                         #
    #    If the intraday model takes 70% at a horizon, the da_blend takes    #
    #    30%. The individual DA model weights must be scaled by that 30% so  #
    #    the total across all real models sums to 1.0.                       #
    # ---------------------------------------------------------------------- #
    if "da_blend" in df_intraday_weights.columns:
        for col in df_da_weights.columns:
            df_da_weights[col] = df_da_weights[col] * df_intraday_weights["da_blend"]

    # Drop the virtual 'da_blend' column - it is not a real model.
    df_intraday_weights = df_intraday_weights.drop(columns=["da_blend"], errors="ignore")

    # ---------------------------------------------------------------------- #
    # 8. Combine day-ahead and intraday weights into a single DataFrame      #
    # ---------------------------------------------------------------------- #
    df_all_weights = pd.concat([df_da_weights, df_intraday_weights], axis=1)

    # Drop any model that contributes zero weight across all horizons.
    df_all_weights = df_all_weights.loc[:, df_all_weights.sum(axis=0) > 0]

    # ---------------------------------------------------------------------- #
    # 9. Convert relative-horizon index -> absolute UTC target times         #
    # ---------------------------------------------------------------------- #
    df_all_weights.index = df_all_weights.index + t0

    logger.info(
        f"Blend weights computed for {len(df_all_weights)} target times, "
        f"participating models: {list(df_all_weights.columns)}",
    )

    return df_all_weights
