"""Logic for blending multiple model forecasts into a single timeseries."""
import asyncio
import logging
from datetime import datetime

import pandas as pd

from nl_blend.data_platform import get_all_forecast_values_as_dataframe
from site_forecast_app.save.data_platform import get_dataplatform_client

logger = logging.getLogger(__name__)


def blend_forecasts_together(
    all_model_df: pd.DataFrame,
    weights_df: pd.DataFrame,
) -> pd.DataFrame:
    """Blends per-model forecast arrays using pre-calculated weight trajectories.

    Matches the UK approach:
      - Iterates over each unique target time in the weights index.
      - For each target time, multiplies available model values by their weights
        and sums them. Weights are constructed to sum to 1.0 by design
        (backup = 1 - primary), so no post-hoc normalisation is performed.
      - If weights deviate from 1.0 (e.g. a model is absent), a warning is
        logged rather than silently normalising, preserving observability.
      - Target times present in weights but absent from any model forecast are
        skipped (no value is emitted), matching the UK fallback behaviour.

    Args:
        all_model_df: Long-format DataFrame with columns
                      [target_time, expected_power_generation_megawatts, model_name].
        weights_df:   Wide-format DataFrame indexed by target_time (UTC), columns
                      are model names, values are blend weights.

    Returns:
        DataFrame with columns [target_time, expected_power_generation_megawatts].
    """
    if all_model_df.empty or weights_df.empty:
        return pd.DataFrame(
            columns=["target_time", "expected_power_generation_megawatts"],
        )

    # Build a fast lookup: target_time -> {model_name -> value}
    model_values = (
        all_model_df
        .set_index(["target_time", "model_name"])["expected_power_generation_megawatts"]
        .to_dict()
    )

    blended_rows = []

    for t in weights_df.index:
        t_weights = weights_df.loc[t].dropna()
        if t_weights.empty:
            continue

        blended_value = 0.0
        weight_sum = 0.0

        for model, w in t_weights.items():
            val = model_values.get((t, model))
            if val is None:
                if w > 0:
                    logger.debug(
                        f"Missing forecast value for model {model} at "
                        f"target_time {t} despite weight={w}",
                    )
                continue
            blended_value += val * w
            weight_sum += w

        if weight_sum == 0.0:
            # No model had data for this target time - skip, matching UK behaviour
            continue

        # Warn if weights don't sum to 1.0 (indicates a missing model) so the
        # deviation is visible in logs rather than silently absorbed.
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(
                f"Blend weights for target_time={t} sum to {weight_sum:.4f} "
                f"(expected 1.0). A model may be missing. "
                f"Available models: {list(t_weights.index)}",
            )

        blended_rows.append(
            {
                "target_time": t,
                "expected_power_generation_megawatts": blended_value,
            },
        )

    return pd.DataFrame(blended_rows)


async def get_blend_forecast_values_latest(
    location_uuid: str,
    weights_df: pd.DataFrame,
    start_datetime: datetime | None = None,
) -> pd.DataFrame:
    """Fetches latest forecast timeseries for all models participating in the blend.

    Returns a single blended timeseries.

    Matches the UK approach:
      - A single Data Platform connection is opened and all models are fetched
        concurrently via asyncio.gather (one DP call per model, but within the
        same connection context - not one full-list call per model).
      - Results are concatenated into a long-format frame and passed to
        blend_forecasts_together.

    Args:
        location_uuid:  Data Platform location UUID to fetch forecasts for.
        weights_df:     Wide-format weight DataFrame (index=target_time, cols=model names).
        start_datetime: If provided, only target times >= this are included.

    Returns:
        Blended timeseries DataFrame with columns
        [target_time, expected_power_generation_megawatts].
    """
    if weights_df.empty:
        logger.warning("No weights provided - skipping blend.")
        return pd.DataFrame(
            columns=["target_time", "expected_power_generation_megawatts"],
        )

    model_names = list(weights_df.columns)
    logger.info(
        f"Fetching forecast values for {len(model_names)} model(s): {model_names}",
    )

    # Single connection; all models fetched concurrently - matches UK single-pass pattern
    async with get_dataplatform_client() as client:
        tasks = [
            get_all_forecast_values_as_dataframe(
                client=client,
                location_uuid=location_uuid,
                model_name=model_name,
                start_datetime=start_datetime,
            )
            for model_name in model_names
        ]
        results = await asyncio.gather(*tasks)

    non_empty = [df for df in results if not df.empty]

    if not non_empty:
        logger.warning(
            "No forecast timeseries data returned from any model. "
            "Cannot produce a blended forecast.",
        )
        return pd.DataFrame(
            columns=["target_time", "expected_power_generation_megawatts"],
        )

    all_model_df = pd.concat(non_empty, axis=0, ignore_index=True)

    logger.info(
        f"Fetched {len(all_model_df)} total forecast rows across "
        f"{all_model_df['model_name'].nunique()} model(s).",
    )

    return blend_forecasts_together(all_model_df, weights_df)
