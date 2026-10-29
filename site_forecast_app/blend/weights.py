"""Logic for calculating optimal model weights for NL site blending.

Single-stage blend: the optimiser picks the single best candidate model
to blend against NL_BACKUP_MODEL over the full scorecard horizon.

NL uses one stage because:
  - All NL models run to the same horizon, so there is no meaningful
    short-horizon / long-horizon split to exploit.
  - The MAE optimiser naturally finds the best blend point across the
    full horizon without needing a hardcoded two-stage structure.

Score window = max_horizon (full scorecard extent).
Min forecast horizon = 15 min (from config).

The final weight DataFrame has one or two columns (best candidate +
backup) and the index is absolute UTC target times (t0 + horizon),
ready for direct look-up in blend_forecasts_together.
"""
import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from ocf import dp

from site_forecast_app.blend.config import BlendConfig
from site_forecast_app.blend.data_platform import fetch_latest_nl_init_times
from site_forecast_app.blend.init_times import calculate_model_delays, shift_mae_curves

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score function
# ---------------------------------------------------------------------------

def make_avg_mae_func(
    horizon: pd.Timedelta,
    min_forecast_horizon: pd.Timedelta,
) -> Callable[[pd.Series], float]:
    """Returns a scoring function that averages MAE up to *horizon*.

    Computes the mean MAE over [min_forecast_horizon, horizon], excluding
    the final boundary point (half-open interval).

    Args:
        horizon:              Upper bound of the scoring window (e.g. max_horizon).
        min_forecast_horizon: Lower bound of the scoring window.

    Returns:
        A callable that takes a horizon-MAE Series and returns a float score.
    """
    def _score(horizon_mae: pd.Series) -> float:
        window = horizon_mae.loc[min_forecast_horizon:horizon]
        return float(window.iloc[:-1].mean())

    _score.__name__ = f"avg_mae_up_to_{horizon}"
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
    """
    weights = np.zeros(size)
    weights[:blend_start_index] = 1.0
    weights[blend_start_index : blend_start_index + len(kernel)] = kernel
    return weights


def index_of_last_non_nan_value(x: np.ndarray) -> int:
    """Returns the index of the last non-NaN element in x, or -1 if all NaN."""
    non_nan_indices = np.where(~np.isnan(x))[0]
    if len(non_nan_indices) == 0:
        return -1
    return int(non_nan_indices[-1])


def infer_backup_model_name(df_mae: pd.DataFrame) -> str:
    """Infer the backup model name from the DataFrame of horizon-MAE values.

    The backup model is the one which has the longest forecast horizon in the input DataFrame. Ties
    are broken by lowest average MAE.

    Args:
        df_mae: DataFrame of horizon-MAE values for all models under consideration. These should
            have been modified for the delay of the latest model run.

    Returns:
        The name of the backup model
    """
    candidates: list[tuple[str, pd.Timedelta, float]] = []

    for model_name in df_mae.columns:
        model_mae = df_mae[model_name].dropna()
        if model_mae.empty:
            continue
        candidates.append((model_name, model_mae.index.max(), model_mae.mean()))

    if not candidates:
        logger.warning("Cannot infer backup model from empty MAE data. Returning first.")
        return df_mae.columns[0]

    backup_model_name, _, _ = max(candidates, key=lambda item: (item[1], -item[2]))
    return backup_model_name


# ---------------------------------------------------------------------------
# Core weight optimiser
# ---------------------------------------------------------------------------

def calculate_optimal_blend_weights(
    df_mae: pd.DataFrame,
    backup_model_name: str,
    candidate_models: list[str],
    kernel: list[float],
    score_func: Callable[[pd.Series], float],
    min_forecast_horizon: pd.Timedelta,
) -> pd.DataFrame:
    """Selects the single best candidate to blend against the backup.

    Sweeps all taper-start positions for each candidate and picks the ONE
    model + position that minimises score_func. Returns a two-column
    DataFrame [best_model, backup_model] where weights sum to 1.0 at every
    horizon. If no candidate beats the backup, returns backup-only weights.

    Args:
        df_mae:               Shifted (horizon x model) MAE DataFrame.
        backup_model_name:    Column name of the fallback model.
        candidate_models:     Model names to evaluate (excluding backup).
        kernel:               Taper kernel (e.g. [0.75, 0.5, 0.25]).
        score_func:           Callable(pd.Series) -> float, lower is better.
        min_forecast_horizon: Minimum horizon; taper positions below this are skipped.

    Returns:
        DataFrame with one or two columns. Weights sum to 1.0 at every
        non-NaN horizon.
    """
    if df_mae.empty:
        return pd.DataFrame(columns=df_mae.columns)

    if backup_model_name not in df_mae.columns:
        raise ValueError(
            f"backup_model_name='{backup_model_name}' not found in df_mae columns: "
            f"{list(df_mae.columns)}",
        )

    kernel_arr = np.array(kernel)
    n = len(df_mae)

    # Fill NaN with a large penalty so gaps are always scored worse than data.
    fill_val = np.nanmax(df_mae.values) * 10
    df_filled = df_mae.fillna(fill_val)

    # Baseline: backup model used alone.
    best_score = score_func(df_filled[backup_model_name])
    best_model = backup_model_name
    best_weights: np.ndarray | None = None

    backup_last_idx = index_of_last_non_nan_value(df_mae[backup_model_name].values)

    for model in candidate_models:
        if model not in df_mae.columns:
            logger.debug(f"Candidate '{model}' not in shifted MAE - skipping.")
            continue

        last_non_nan_idx = index_of_last_non_nan_value(df_mae[model].values)

        if last_non_nan_idx < 0:
            logger.debug(f"Model '{model}' has no valid MAE data - skipping.")
            continue

        if last_non_nan_idx >= backup_last_idx:
            # Candidate covers at least as much horizon as the backup.
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
            # Sweep taper-start positions over the candidate's valid range.
            max_blend_start_pos = last_non_nan_idx - len(kernel_arr) + 1
            for position in range(max_blend_start_pos + 1):
                if df_mae.index[position] < min_forecast_horizon:
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

    # Build output DataFrame
    if best_model == backup_model_name:
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

    candidate_missing = df_mae[best_model].isna()
    backup_missing = df_mae[backup_model_name].isna()

    # Exclude the candidate where it has no data.
    best_weights[candidate_missing] = np.nan
    # Backup gets the remaining weight; NaN counts as 0 so the backup isn't NaN too.
    backup_weights = 1 - np.nan_to_num(best_weights, nan=0.0)
    # Candidate missing but backup available: backup takes full weight.
    backup_weights[candidate_missing & ~backup_missing] = 1.0
    # Exclude the backup where it has no data.
    backup_weights[backup_missing] = np.nan

    logger.debug(
        f"Selected '{best_model}' as best candidate (score {best_score:.5f}).",
    )
    return pd.DataFrame(
        {best_model: best_weights, backup_model_name: backup_weights},
        index=df_mae.index,
    )

# ---------------------------------------------------------------------------
# Shared weight computation logic
# ---------------------------------------------------------------------------
async def _compute_weights(
    t0: pd.Timestamp,
    location_uuid: str,
    df_mae: pd.DataFrame,
    max_horizon: pd.Timedelta,
    client: dp.DataPlatformDataServiceStub,
    day_ahead_candidate_models: list[str],
    intraday_candidate_models: list[str],
    label: str,
    backup_model: str,
    kernel: list[float],
    min_forecast_horizon: pd.Timedelta,
) -> pd.DataFrame:
    """Fetches init times, shifts MAE curves, and runs the two-stage optimiser.

    Shared by get_blend_weights and get_regional_blend_weights.

    Args:
        t0:                   Blend reference time (UTC).
        location_uuid:        Data Platform location UUID.
        df_mae:               (horizon x model) MAE scorecard.
        max_horizon:          Maximum scorecard horizon; used as max_delay and as
                              the score window upper bound.
        client:               Authenticated Data Platform gRPC client stub.
        day_ahead_candidate_models: Models to evaluate for the day-ahead blend.
        intraday_candidate_models: Models to evaluate for the intraday blend.
        label:                Short label for log messages ("National"/"Regional").
        backup_model:         Fallback if no day-ahead model has a recent run.
        kernel:               Taper kernel weights (e.g. [0.75, 0.5, 0.25]).
        min_forecast_horizon: Minimum forecast horizon used in the scoring window.

    Returns:
        Wide DataFrame indexed by absolute UTC target time.
        Weights sum to 1.0 at every horizon.
        Returns an empty DataFrame on failure.
    """
    all_models = [
        backup_model,
        *day_ahead_candidate_models,
        *intraday_candidate_models,
    ]

    # Fetch model initialisation times
    model_init_times = await fetch_latest_nl_init_times(
        client=client,
        location_uuid=location_uuid,
        model_names=all_models,
        t0=t0,
        max_delay=max_horizon,
    )
    logger.info(f"[{label}] Fetched model init times: {model_init_times}")

    # Assign penalty delays to missing models
    missing = [m for m in all_models if m not in model_init_times]
    if missing:
        logger.info(f"[{label}] No init time found for {missing}; assigning penalty delays.")
    found_any = any(m in model_init_times for m in [backup_model, *day_ahead_candidate_models])
    for m in missing:
        # Missing models get delay=max_horizon (effectively excluded).
        # Backup gets delay=0 only if no day-ahead model has a recent run.
        model_init_times[m] = t0 if (m == backup_model and not found_any) else t0 - max_horizon

    # Compute delays and shift MAE scorecard
    delays = calculate_model_delays(model_init_times, t0)
    logger.info(f"[{label}] Model delays relative to t0 ({t0}): {delays}")

    df_delayed_mae = shift_mae_curves(df_mae, delays)

    if df_delayed_mae.empty:
        logger.error(
            f"[{label}] Shifted MAE DataFrame is empty - cannot produce blend weights.",
        )
        return pd.DataFrame()

    # Stage 1: optimise the day-ahead blend over the full horizon
    da_score_func = make_avg_mae_func(
        pd.Timedelta(hours=36),
        min_forecast_horizon,
    )

    # Pick the backup dynamically, as in uk-pv-forecast-blend
    da_models = [
        m for m in dict.fromkeys([backup_model, *day_ahead_candidate_models])
        if m in df_delayed_mae.columns
    ]
    da_backup_model = infer_backup_model_name(df_delayed_mae[da_models])
    logger.info(f"[{label}] Day-ahead backup model: {da_backup_model}")

    df_da_model_weights = calculate_optimal_blend_weights(
        df_mae=df_delayed_mae,
        backup_model_name=da_backup_model,
        candidate_models=[m for m in da_models if m != da_backup_model],
        kernel=kernel,
        score_func=da_score_func,
        min_forecast_horizon=min_forecast_horizon,
    )

    logger.info(f"[{label}] Day-ahead weights (head):\n{df_da_model_weights.head()}")

    # Keep relative-horizon index while building the Stage 2 blend
    logger.info(
        f"[{label}] Day-ahead weights computed for {len(df_da_model_weights)} target times, "
        f"participating models: {list(df_da_model_weights.columns)}",
    )

    # Stage 2: optimise intraday models against the day-ahead blend over 8 hours
    df_delayed_mae["blend_day_ahead"] = (
        df_da_model_weights
        .set_axis(df_delayed_mae.index)
        .mul(df_delayed_mae[df_da_model_weights.columns])
        .sum(axis=1, skipna=True)
    )

    intraday_score_func = make_avg_mae_func(
        pd.Timedelta(hours=8),
        min_forecast_horizon,
    )

    intraday_models = list(
        dict.fromkeys(
            [*day_ahead_candidate_models, *intraday_candidate_models],
        ),
    )

    df_intraday_mae = df_delayed_mae.copy()

    # Intraday candidates are unavailable beyond the 8-hour window.
    intraday_horizon = df_intraday_mae.index > pd.Timedelta(hours=8)
    df_intraday_mae.loc[intraday_horizon, intraday_models] = np.nan

    df_intraday_model_weights = calculate_optimal_blend_weights(
        df_mae=df_intraday_mae,
        backup_model_name="blend_day_ahead",
        candidate_models=intraday_models,
        kernel=kernel,
        score_func=intraday_score_func,
        min_forecast_horizon=min_forecast_horizon,
    )

    # Apply the intraday blend weight to the day-ahead model weights
    blend_day_ahead_weights = df_intraday_model_weights["blend_day_ahead"]

    for col in df_da_model_weights.columns:
        df_da_model_weights[col] = (
            df_da_model_weights[col] * blend_day_ahead_weights
        )

    df_intraday_model_weights = df_intraday_model_weights.drop(
        columns="blend_day_ahead",
    )

    df_all_weights = pd.concat(
        [df_da_model_weights, df_intraday_model_weights],
        axis=1,
    )

    df_all_weights = df_all_weights.loc[:, df_all_weights.sum(axis=0) > 0]

    df_all_weights.index = df_all_weights.index + t0

    return df_all_weights

# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def get_blend_weights(
    t0: pd.Timestamp,
    location_uuid: str,
    df_mae: pd.DataFrame,
    max_horizon: pd.Timedelta,
    client: dp.DataPlatformDataServiceStub,
    config: BlendConfig,
) -> pd.DataFrame:
    """Produces the national blend weight DataFrame for t0.

    Two-stage: blends day-ahead models, then blends intraday models on top.
    The day-ahead backup is inferred from the delay-shifted MAE curves.

    Args:
        t0:            Blend reference time (UTC, floored to 15 min).
        location_uuid: Data Platform location UUID.
        df_mae:        (horizon x model) MAE scorecard.
        max_horizon:   Maximum horizon in the scorecard.
        client:        Authenticated Data Platform gRPC client stub.
        config:        Blend configuration supplying model names and kernel.

    Returns:
        Wide DataFrame indexed by absolute UTC target time.
        Weights sum to 1.0 at every horizon.
        Returns an empty DataFrame if the shifted MAE frame is empty.
    """
    return await _compute_weights(
        t0=t0,
        location_uuid=location_uuid,
        df_mae=df_mae,
        max_horizon=max_horizon,
        client=client,
        day_ahead_candidate_models=config.day_ahead_candidate_models,
        intraday_candidate_models=config.intraday_candidate_models,
        label="National",
        backup_model=config.backup_model,
        kernel=config.blend_kernel,
        min_forecast_horizon=config.min_forecast_horizon,
    )


async def get_regional_blend_weights(
    t0: pd.Timestamp,
    location_uuid: str,
    df_mae: pd.DataFrame,
    max_horizon: pd.Timedelta,
    client: dp.DataPlatformDataServiceStub,
    config: BlendConfig,
) -> pd.DataFrame:
    """Produces a regional blend weight DataFrame for t0.

    Identical pipeline to :func:`get_blend_weights` but uses
    config.regional_candidate_models as the candidate set.

    Args:
        t0:            Blend reference time (UTC, floored to 15 min).
        location_uuid: Data Platform location UUID for the specific region.
        df_mae:        (horizon x model) MAE scorecard.
        max_horizon:   Maximum horizon in the scorecard.
        client:        Authenticated Data Platform gRPC client stub.
        config:        Blend configuration supplying model names and kernel.

    Returns:
        Wide DataFrame indexed by absolute UTC target time.
        Weights sum to 1.0 at every horizon.
        Returns an empty DataFrame if the shifted MAE frame is empty.
    """
    return await _compute_weights(
        t0=t0,
        location_uuid=location_uuid,
        df_mae=df_mae,
        max_horizon=max_horizon,
        client=client,
        day_ahead_candidate_models=config.regional_candidate_models,
        intraday_candidate_models=config.intraday_candidate_models,
        label="Regional",
        backup_model=config.backup_model,
        kernel=config.blend_kernel,
        min_forecast_horizon=config.min_forecast_horizon,
    )
