"""Logic for managing and extracting forecast initialization times for NL sites."""
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Minimum horizon we will ever emit a blended value for,
MIN_FORECAST_HORIZON = pd.Timedelta("30min")


# ---------------------------------------------------------------------------
# Scorecard loading
# ---------------------------------------------------------------------------

def load_nl_mae_scorecard(filepath: str) -> pd.DataFrame:
    """Loads the NL normalised MAE scorecard from a CSV file.

    Supported CSV shapes
    --------------------
    Long format (preferred):
        Columns: horizon_minutes, model, norm_abs_error
        Pivoted into a (horizon x model) DataFrame with a pd.Timedelta index.

    Wide / legacy format (fallback):
        First column treated as the horizon label; remaining columns are models.

    Returns:
    -------
    DataFrame indexed by pd.Timedelta (forecast horizon), one column per model.

    Raises:
    ------
    Exception if the file cannot be read or parsed - caller is responsible for
    handling this (app.py logs and exits).
    """
    try:
        df = pd.read_csv(filepath)
    except Exception:
        logger.error(f"Cannot read MAE scorecard file '{filepath}'.")
        raise

    if "horizon_minutes" in df.columns and "model" in df.columns:
        df_pivot = df.pivot(
            index="horizon_minutes", columns="model", values="norm_abs_error",
        )
        df_pivot.index = pd.to_timedelta(df_pivot.index, unit="m")
        df_pivot.columns.name = None
        return df_pivot

    # Legacy wide format
    df = pd.read_csv(filepath, index_col=0)
    df.index = pd.to_timedelta(df.index)
    df.columns.name = None
    return df


# ---------------------------------------------------------------------------
# Delay calculation
# ---------------------------------------------------------------------------

def calculate_model_delays(
    model_init_times: dict[str, pd.Timestamp],
    t0: pd.Timestamp,
) -> dict[str, pd.Timedelta]:
    """Calculates each model's effective delay relative to t0.

    The delay is:  t0  -  floor(init_time, 30 min)

    approach: the initialisation time is rounded down to the
    nearest half-hour before subtracting, so a model initialised at 09:47 is
    treated as if it started at 09:30, giving a delay of 30 min when t0=10:00.

    A negative delay (init_time > t0) is clamped to zero.

    Args:
        model_init_times: Model name -> UTC initialisation timestamp.
        t0:               Blend reference time (UTC).

    Returns:
        Model name -> delay as pd.Timedelta.
    """
    if t0.tz is None:
        t0 = t0.tz_localize("UTC")

    delays: dict[str, pd.Timedelta] = {}

    for model_name, init_time in model_init_times.items():
        if init_time is None:
            continue

        if init_time.tz is None:
            init_time = init_time.tz_localize("UTC")

        approx_init = init_time.floor("30min")
        delay = t0 - approx_init

        logger.debug(
            f"Model {model_name} init_time: {init_time}, "
            f"approx_init: {approx_init}, raw delay: {delay}",
        )

        if delay < pd.Timedelta(0):
            logger.debug(f"Clamping negative delay for {model_name} to 0")
            delay = pd.Timedelta(0)

        delays[model_name] = delay

    return delays


# ---------------------------------------------------------------------------
# MAE curve shifting
# ---------------------------------------------------------------------------

def shift_mae_curves(
    df_mae: pd.DataFrame,
    delays: dict[str, pd.Timedelta],
) -> pd.DataFrame:
    """Shifts each model's MAE curve rightward by its delay.

    If a model is delayed by 60 min, its scorecard row at horizon=30 min
    actually represents performance at an effective horizon of 90 min.
    Shifting the index left by the delay brings the curve into the same
    reference frame as the blend t0.

    Only models present in *both* df_mae.columns and delays are included.
    Horizons below MIN_FORECAST_HORIZON are dropped after shifting.
    Rows where all models are NaN are dropped.

    Args:
        df_mae:  (horizon x model) MAE scorecard, index is pd.Timedelta.
        delays:  Model name -> delay as pd.Timedelta.

    Returns:
        Shifted (horizon x model) DataFrame, same dtype as input.
    """
    common_models = set(df_mae.columns) & set(delays.keys())

    if not common_models:
        logger.warning(
            "No overlap between MAE scorecard models and models with known delays. "
            f"Scorecard models: {list(df_mae.columns)}. "
            f"Delay models: {list(delays.keys())}.",
        )
        return pd.DataFrame(columns=df_mae.columns)

    shifted_frames = []
    for model in common_models:
        delay = delays[model]
        logger.debug(f"Shifting MAE curve for {model} by {delay}")
        shifted = pd.DataFrame(
            df_mae[model].values,
            index=df_mae.index - delay,
            columns=[model],
        )
        shifted_frames.append(shifted)

    df_shifted = pd.concat(shifted_frames, axis=1).sort_index()

    # Drop sub-minimum horizons and all-NaN rows
    df_shifted = df_shifted.loc[df_shifted.index >= MIN_FORECAST_HORIZON]
    df_shifted = df_shifted.dropna(axis=0, how="all")

    return df_shifted


# ---------------------------------------------------------------------------
# Init-time extraction (pure function - fully unit-testable)
# ---------------------------------------------------------------------------

def extract_latest_init_times(
    forecasts: list[Any],
    model_names: list[str],
    t0: pd.Timestamp,
    max_delay: pd.Timedelta,
) -> dict[str, pd.Timestamp]:
    """Extracts the most-recent valid initialisation time for each requested model.

    From a list of Data Platform forecast objects.

    "Valid" means: init_time >= t0 - max_delay.

    This is a pure function (no I/O) so it can be unit-tested with mock objects
    without a live Data Platform connection separation of
    concerns between I/O (data_platform.py) and logic (init_times.py).

    Args:
        forecasts:    List of DP forecast objects (real gRPC or mock).
        model_names:  Models we want init times for.
        t0:           Blend reference time (UTC).
        max_delay:    Forecasts initialised before (t0 - max_delay) are ignored.

    Returns:
        Dict mapping model name -> latest valid initialisation timestamp (UTC).
        Only models for which at least one valid forecast exists are included.
    """
    if t0.tz is None:
        t0 = t0.tz_localize("UTC")

    earliest_valid = t0 - max_delay
    model_init_times: dict[str, pd.Timestamp] = {}

    for forecast in forecasts:
        if not (
            hasattr(forecast, "forecaster")
            and hasattr(forecast.forecaster, "forecaster_name")
        ):
            continue

        forecaster_name = forecast.forecaster.forecaster_name
        if forecaster_name not in model_names:
            continue

        raw_init = forecast.initialization_timestamp_utc
        try:
            if hasattr(raw_init, "ToDatetime"):
                init_ts = pd.Timestamp(raw_init.ToDatetime(), tz="UTC")
            else:
                init_ts = pd.Timestamp(raw_init)
        except Exception:
            logger.warning(
                f"Could not parse init timestamp for model '{forecaster_name}': {raw_init!r}",
            )
            continue

        init_ts = (
            init_ts.tz_localize("UTC") if init_ts.tz is None else init_ts.tz_convert("UTC")
        )

        if init_ts < earliest_valid:
            logger.debug(
                f"Ignoring stale forecast for '{forecaster_name}': "
                f"init={init_ts} < earliest_valid={earliest_valid}",
            )
            continue

        if (
            forecaster_name not in model_init_times
            or init_ts > model_init_times[forecaster_name]
        ):
            logger.debug(f"Updating latest valid init time for '{forecaster_name}' to {init_ts}")
            model_init_times[forecaster_name] = init_ts

    return model_init_times
