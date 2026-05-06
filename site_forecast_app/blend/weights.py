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
from dp_sdk.ocf import dp

from site_forecast_app.blend.config import load_blend_config
from site_forecast_app.blend.data_platform import fetch_latest_nl_init_times
from site_forecast_app.blend.init_times import calculate_model_delays, shift_mae_curves

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load config - all tuning parameters live in blend/config.yaml
# ---------------------------------------------------------------------------
_cfg = load_blend_config()

NL_BACKUP_MODEL: str = _cfg.backup_model
NL_NATIONAL_CANDIDATE_MODELS: list[str] = _cfg.national_candidate_models
NL_REGIONAL_CANDIDATE_MODELS: list[str] = _cfg.regional_candidate_models
BLEND_KERNEL: list[float] = _cfg.blend_kernel
MIN_FORECAST_HORIZON: pd.Timedelta = _cfg.min_forecast_horizon


# ---------------------------------------------------------------------------
# Score function
# ---------------------------------------------------------------------------

def make_avg_mae_func(horizon: pd.Timedelta) -> Callable[[pd.Series], float]:
    """Returns a scoring function that averages MAE up to *horizon*.

    Computes the mean MAE over [MIN_FORECAST_HORIZON, horizon], excluding
    the final boundary point (half-open interval).

    Args:
        horizon: Upper bound of the scoring window (e.g. max_horizon).

    Returns:
        A callable that takes a horizon-MAE Series and returns a float score.
    """
    def _score(horizon_mae: pd.Series) -> float:
        window = horizon_mae.loc[MIN_FORECAST_HORIZON:horizon]
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


# ---------------------------------------------------------------------------
# Core weight optimiser
# ---------------------------------------------------------------------------

def calculate_optimal_blend_weights(
    df_mae: pd.DataFrame,
    backup_model_name: str,
    candidate_models: list[str],
    kernel: list[float],
    score_func: Callable[[pd.Series], float],
) -> pd.DataFrame:
    """Selects the single best candidate to blend against the backup.

    Sweeps all taper-start positions for each candidate and picks the ONE
    model + position that minimises score_func. Returns a two-column
    DataFrame [best_model, backup_model] where weights sum to 1.0 at every
    horizon. If no candidate beats the backup, returns backup-only weights.

    Args:
        df_mae:            Shifted (horizon x model) MAE DataFrame.
        backup_model_name: Column name of the fallback model.
        candidate_models:  Model names to evaluate (excluding backup).
        kernel:            Taper kernel (e.g. [0.75, 0.5, 0.25]).
        score_func:        Callable(pd.Series) -> float, lower is better.

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

    backup_weights = 1 - best_weights
    best_weights[df_mae[best_model].isna()] = np.nan
    backup_weights[df_mae[backup_model_name].isna()] = np.nan

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
    candidate_models: list[str],
    label: str,
) -> pd.DataFrame:
    """Fetches init times, shifts MAE curves, and runs the single-stage optimiser.

    Shared by get_blend_weights

    Args:
        t0:               Blend reference time (UTC).
        location_uuid:    Data Platform location UUID.
        df_mae:           (horizon x model) MAE scorecard.
        max_horizon:      Maximum scorecard horizon; used as max_delay and as
                          the score window upper bound.
        client:           Authenticated Data Platform gRPC client stub.
        candidate_models: Models to evaluate against NL_BACKUP_MODEL.
        label:            Short label for log messages ("National"/"Regional").

    Returns:
        Wide DataFrame indexed by absolute UTC target time.
        Weights sum to 1.0 at every horizon.
        Returns an empty DataFrame on failure.
    """
    all_models = [NL_BACKUP_MODEL, *candidate_models]

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
    for m in missing:
        # Backup always gets delay=0 (always available).
        # Candidates get delay=max_horizon (effectively excluded).
        model_init_times[m] = t0 if m == NL_BACKUP_MODEL else t0 - max_horizon

    # Compute delays and shift MAE scorecard
    delays = calculate_model_delays(model_init_times, t0)
    logger.info(f"[{label}] Model delays relative to t0 ({t0}): {delays}")

    df_delayed_mae = shift_mae_curves(df_mae, delays)

    if df_delayed_mae.empty:
        logger.error(
            f"[{label}] Shifted MAE DataFrame is empty - cannot produce blend weights.",
        )
        return pd.DataFrame()

    # Single-stage optimisation: best candidate vs backup over full horizon
    score_func = make_avg_mae_func(max_horizon)
    df_weights = calculate_optimal_blend_weights(
        df_mae=df_delayed_mae,
        backup_model_name=NL_BACKUP_MODEL,
        candidate_models=candidate_models,
        kernel=BLEND_KERNEL,
        score_func=score_func,
    )
    logger.info(f"[{label}] Weights (head):\n{df_weights.head()}")

    # Convert relative-horizon index -> absolute UTC target times
    df_weights.index = df_weights.index + t0

    logger.info(
        f"[{label}] Blend weights computed for {len(df_weights)} target times, "
        f"participating models: {list(df_weights.columns)}",
    )
    return df_weights


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def get_blend_weights(
    t0: pd.Timestamp,
    location_uuid: str,
    df_mae: pd.DataFrame,
    max_horizon: pd.Timedelta,
    client: dp.DataPlatformDataServiceStub,
) -> pd.DataFrame:
    """Produces the national blend weight DataFrame for t0.

    Single-stage: picks the best model from NL_NATIONAL_CANDIDATE_MODELS
    to blend against NL_BACKUP_MODEL, scored over the full scorecard horizon.

    Args:
        t0:            Blend reference time (UTC, floored to 15 min).
        location_uuid: Data Platform location UUID.
        df_mae:        (horizon x model) MAE scorecard.
        max_horizon:   Maximum horizon in the scorecard.
        client:        Authenticated Data Platform gRPC client stub.

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
        candidate_models=NL_NATIONAL_CANDIDATE_MODELS,
        label="National",
    )


async def get_regional_blend_weights(
    t0: pd.Timestamp,
    location_uuid: str,
    df_mae: pd.DataFrame,
    max_horizon: pd.Timedelta,
    client: dp.DataPlatformDataServiceStub,
) -> pd.DataFrame:
    """Produces a regional blend weight DataFrame for t0.

    Identical pipeline to :func:`get_blend_weights` but uses
    NL_REGIONAL_CANDIDATE_MODELS as the candidate set, which is typically
    a subset of the national candidates (see config.yaml).

    Args:
        t0:            Blend reference time (UTC, floored to 15 min).
        location_uuid: Data Platform location UUID for the specific region.
        df_mae:        (horizon x model) MAE scorecard.
        max_horizon:   Maximum horizon in the scorecard.
        client:        Authenticated Data Platform gRPC client stub.

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
        candidate_models=NL_REGIONAL_CANDIDATE_MODELS,
        label="Regional",
    )
