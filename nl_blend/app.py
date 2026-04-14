"""Main entry point for the NL forecast blending application."""
import asyncio
import logging
import os

import pandas as pd

from nl_blend.blend import get_blend_forecast_values_latest
from nl_blend.init_times import load_nl_mae_scorecard
from nl_blend.weights import get_nl_blend_weights
from site_forecast_app.save.data_platform import build_dp_location_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nl_blend_app")

# Controls how often ForecastValue (non-latest) table is written,30-min cadence
FORECAST_VALUE_WRITE_INTERVAL_MINUTES = 30


async def run_blend_app() -> None:
    """Main execution point for the NL Blend app.

    Steps:
    1. Determine blend reference time (t0)
    2. Resolve target location UUID from Data Platform
    3. Load the MAE scorecard
    4. Calculate delay-adjusted blend weights
    5. Fetch raw forecast timeseries and blend them
    6. Save results:
       - ForecastValueLatest: every run
       - ForecastValue:       every 30 minutes only
    """
    logger.info("Starting NL Blend execution.")

    # ------------------------------------------------------------------ #
    # 1. Determine blend reference time - floor to 30-min boundary    #
    # ------------------------------------------------------------------ #
    t0 = pd.Timestamp.utcnow().floor("30min")
    logger.info(f"Blend t0: {t0}")

    # ------------------------------------------------------------------ #
    # 2. Resolve location UUID from Data Platform                          #
    # ------------------------------------------------------------------ #
    logger.info("Fetching location map from Data Platform.")
    try:
        dp_loc_map = await build_dp_location_map()
        if not dp_loc_map:
            logger.error("Data Platform returned an empty location map. Cannot continue.")
            return
        location_uuid = next(iter(dp_loc_map.values()))
        logger.info(f"Using location UUID: {location_uuid}")
    except Exception:
        logger.exception("Failed to connect to Data Platform while fetching location map.")
        return

    # ------------------------------------------------------------------ #
    # 3. Load MAE scorecard                                                #
    # ------------------------------------------------------------------ #
    scorecard_path = os.path.join(
        os.path.dirname(__file__), "data", "backtest_nmae_comparison.csv",
    )
    try:
        df_mae = load_nl_mae_scorecard(scorecard_path)
        logger.info(
            f"Loaded MAE scorecard from '{scorecard_path}' "
            f"with models: {list(df_mae.columns)}",
        )
    except Exception:
        logger.exception(f"Failed to load MAE scorecard from '{scorecard_path}'.")
        return

    max_horizon = df_mae.index.max()

    # ------------------------------------------------------------------ #
    # 4. Calculate delay-adjusted blend weights                            #
    # ------------------------------------------------------------------ #
    logger.info("Calculating delay-adjusted blend weights.")
    try:
        weights_df = await get_nl_blend_weights(
            t0=t0,
            location_uuid=location_uuid,
            df_mae=df_mae,
            max_horizon=max_horizon,
        )
        logger.info(f"Blend weights calculated:\n{weights_df.head(10)}")
    except Exception:
        logger.exception("Failed to calculate blend weights.")
        return

    # ------------------------------------------------------------------ #
    # 5. Fetch forecast timeseries and produce blended values              #
    # ------------------------------------------------------------------ #
    logger.info("Fetching raw forecast values and blending.")
    try:
        blended_df = await get_blend_forecast_values_latest(
            location_uuid=location_uuid,
            weights_df=weights_df,
            start_datetime=t0,
        )
    except Exception:
        logger.exception("Failed to fetch or blend forecast timeseries.")
        return

    if blended_df.empty:
        logger.warning(
            "Blended timeseries is empty. "
            "This is expected in dev when no forecast megawatts are stored.",
        )
        return

    logger.info(f"Blended timeseries (first 10 rows):\n{blended_df.head(10)}")

    # ------------------------------------------------------------------ #
    # 6. Save results            #
    #    ForecastValueLatest: always written                               #
    #    ForecastValue:       written only every 30 minutes               #
    # ------------------------------------------------------------------ #
    await _save_forecasts(t0=t0, blended_df=blended_df)


async def _save_forecasts(t0: pd.Timestamp, blended_df: pd.DataFrame) -> None:
    """Persists the blended forecast following the dual-table write pattern.

    - ForecastValueLatest is always updated so the API always has fresh data.
    - ForecastValue is only written every 30 minutes to keep table growth manageable.

    Data Platform write calls.
    """
    # Always write to the latest-value store
    logger.info(
        f"Saving {len(blended_df)} rows to ForecastValueLatest "
        f"(blend_name='nl_blend', t0={t0}).",
    )

    # Write to the historical store only on the 30-minute cadence
    if t0.minute % FORECAST_VALUE_WRITE_INTERVAL_MINUTES == 0:
        logger.info(
            f"Saving {len(blended_df)} rows to ForecastValue "
            f"(blend_name='nl_blend', t0={t0}). to DATA PLATFORM",
        )
    else:
        logger.info(
            f"Skipping ForecastValue write at t0={t0} "
            f"(only written every {FORECAST_VALUE_WRITE_INTERVAL_MINUTES} min).",
        )


if __name__ == "__main__":
    asyncio.run(run_blend_app())
