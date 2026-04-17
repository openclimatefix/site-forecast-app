"""Logic for blending multiple model forecasts into a single timeseries."""
import asyncio
import logging
import math
from datetime import datetime

import pandas as pd

from nl_blend.data_platform import get_all_forecast_values_as_dataframe
from dp_sdk.ocf import dp

logger = logging.getLogger(__name__)


def blend_forecasts_together(
    all_model_df: pd.DataFrame,
    weights_df: pd.DataFrame,
) -> pd.DataFrame:
    """Blends per-model forecast arrays using pre-calculated weight trajectories.

    Blends p50 (expected_power_generation_megawatts), p10 (p10_mw), and p90
    (p90_mw) using the same weights for all three quantities.

    Approach:
      - Iterates over each unique target time in the weights index.
      - For each target time, multiplies available model values by their weights
        and sums them. Weights are constructed to sum to 1.0 by design
        (backup = 1 - primary), so no post-hoc normalisation is performed for
        p50.
      - p10/p90 are optional per model per step: if a model is missing either
        quantity at a given time, the blended p10/p90 is normalised only by the
        weight sum of models that did supply that quantity.
      - If p50 weights deviate from 1.0 (e.g. a model is absent), a warning is
        logged rather than silently normalising, preserving observability.
      - Target times present in weights but absent from any model forecast are
        skipped (no value is emitted).

    Args:
        all_model_df: Long-format DataFrame with columns
                      [target_time, expected_power_generation_megawatts,
                       p10_mw, p90_mw, model_name].
        weights_df:   Wide-format DataFrame indexed by target_time (UTC), columns
                      are model names, values are blend weights.

    Returns:
        DataFrame with columns
        [target_time, expected_power_generation_megawatts, p10_mw, p90_mw].
        p10_mw / p90_mw are NaN for steps where no model supplied those values.
    """
    if all_model_df.empty or weights_df.empty:
        return pd.DataFrame(
            columns=[
                "target_time",
                "expected_power_generation_megawatts",
                "p10_mw",
                "p90_mw",
            ],
        )

    # Build fast lookups keyed by (target_time, model_name)
    _idx = ["target_time", "model_name"]
    p50_values = (
        all_model_df.set_index(_idx)["expected_power_generation_megawatts"].to_dict()
    )
    p10_values = (
        all_model_df.set_index(_idx)["p10_mw"].to_dict()
        if "p10_mw" in all_model_df.columns else {}
    )
    p90_values = (
        all_model_df.set_index(_idx)["p90_mw"].to_dict()
        if "p90_mw" in all_model_df.columns else {}
    )

    blended_rows = []

    for t in weights_df.index:
        t_weights = weights_df.loc[t].dropna()
        if t_weights.empty:
            continue

        p50_blend = 0.0
        p10_blend = 0.0
        p90_blend = 0.0
        weight_sum = 0.0
        p10_weight_sum = 0.0
        p90_weight_sum = 0.0

        for model, w in t_weights.items():
            p50_val = p50_values.get((t, model))
            if p50_val is None:
                if w > 0:
                    logger.debug(
                        f"Missing p50 for model {model} at "
                        f"target_time {t} despite weight={w}",
                    )
                continue

            p50_blend += p50_val * w
            weight_sum += w

            # p10 / p90 are optional - include only when present and not NaN.
            p10_val = p10_values.get((t, model))
            if p10_val is not None and not math.isnan(p10_val):
                p10_blend += p10_val * w
                p10_weight_sum += w

            p90_val = p90_values.get((t, model))
            if p90_val is not None and not math.isnan(p90_val):
                p90_blend += p90_val * w
                p90_weight_sum += w

        if weight_sum == 0.0:
            # No model had p50 data for this target time - skip entirely.
            continue

        # Warn if p50 weights don't sum to 1.0 (indicates a missing model).
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(
                f"Blend weights for target_time={t} sum to {weight_sum:.4f} "
                f"(expected 1.0). A model may be missing. "
                f"Available models: {list(t_weights.index)}",
            )

        # Normalise p10/p90 by the weight sum of models that supplied them,
        # so that a missing p10/p90 from one model does not deflate the blend.
        blended_rows.append(
            {
                "target_time": t,
                "expected_power_generation_megawatts": p50_blend / weight_sum,
                "p10_mw": p10_blend / p10_weight_sum if p10_weight_sum > 0.0 else float("nan"),
                "p90_mw": p90_blend / p90_weight_sum if p90_weight_sum > 0.0 else float("nan"),
            },
        )

    return pd.DataFrame(blended_rows)


async def get_blend_forecast_values_latest(
    location_uuid: str,
    weights_df: pd.DataFrame,
    client: dp.DataPlatformDataServiceStub,
    start_datetime: datetime | None = None,
) -> pd.DataFrame:
    """Fetches latest forecast timeseries for all models participating in the blend.

    Returns a single blended timeseries.

    A single Data Platform connection is used and all models are fetched
    concurrently via asyncio.gather (one DP call per model within the same
    connection context).

    Args:
        location_uuid:  Data Platform location UUID to fetch forecasts for.
        weights_df:     Wide-format weight DataFrame (index=target_time, cols=model names).
        client:         Authenticated Data Platform gRPC client stub.
        start_datetime: If provided, only target times >= this are included.

    Returns:
        Blended timeseries DataFrame with columns
        [target_time, expected_power_generation_megawatts, p10_mw, p90_mw].
        p10_mw / p90_mw are NaN for steps where no model supplied those values.
    """
    if weights_df.empty:
        logger.warning("No weights provided - skipping blend.")
        return pd.DataFrame(
            columns=[
                "target_time",
                "expected_power_generation_megawatts",
                "p10_mw",
                "p90_mw",
            ],
        )

    model_names = list(weights_df.columns)
    logger.info(
        f"Fetching forecast values for {len(model_names)} model(s): {model_names}",
    )

    # All models fetched concurrently within the single shared connection.
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
            columns=[
                "target_time",
                "expected_power_generation_megawatts",
                "p10_mw",
                "p90_mw",
            ],
        )

    all_model_df = pd.concat(non_empty, axis=0, ignore_index=True)

    logger.info(
        f"Fetched {len(all_model_df)} total forecast rows across "
        f"{all_model_df['model_name'].nunique()} model(s).",
    )

    return blend_forecasts_together(all_model_df, weights_df)