"""Main entry point for the NL forecast blending application."""
import asyncio
import logging
import os

import pandas as pd
from dp_sdk.ocf import dp

from nl_blend.blend import get_blend_forecast_values_latest
from nl_blend.config import load_nl_blend_config
from nl_blend.data_platform import (
    build_forecast_value_objects,
    fetch_location_capacity_watts,
)
from nl_blend.init_times import load_nl_mae_scorecard
from nl_blend.weights import get_nl_blend_weights
from site_forecast_app.save.data_platform import (
    create_forecaster_if_not_exists,
    fetch_dp_location_map,
    get_dataplatform_client,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nl_blend_app")

_cfg = load_nl_blend_config()

# Forecaster name written to the Data Platform
NL_BLEND_FORECASTER_NAME = _cfg.forecaster_name


async def run_blend_app() -> None:
    """Main execution point for the NL Blend app.

    Steps:
    1. Determine blend reference time (t0)
    2. Resolve target location UUID from Data Platform
    3. Fetch location capacity (watts) from Data Platform
    4. Load the MAE scorecard
    5. Calculate delay-adjusted blend weights
    6. Fetch raw forecast timeseries and blend them
    7. Save results to Data Platform
    """
    logger.info("Starting NL Blend execution.")

    # ------------------------------------------------------------------ #
    # Determine blend reference time - floor to 15-min boundary        #
    # ------------------------------------------------------------------ #
    t0 = pd.Timestamp.utcnow().floor("15min")
    logger.info(f"Blend t0: {t0}")

    # ------------------------------------------------------------------ #
    # Open a single Data Platform connection for the entire run           #
    # ------------------------------------------------------------------ #
    async with get_dataplatform_client() as client:

        # -------------------------------------------------------------- #
        # Resolve location UUID from Data Platform                     #
        # -------------------------------------------------------------- #
        logger.info("Fetching location map from Data Platform.")
        try:
            dp_loc_map = await fetch_dp_location_map(client)
            if not dp_loc_map:
                logger.error("Data Platform returned an empty location map. Cannot continue.")
                return
            location_uuid = next(iter(dp_loc_map.values()))
            logger.info(f"Using location UUID: {location_uuid}")
        except Exception:
            logger.exception("Failed to connect to Data Platform while fetching location map.")
            return

        # -------------------------------------------------------------- #
        # Fetch location capacity                                      #
        # -------------------------------------------------------------- #
        try:
            capacity_watts = await fetch_location_capacity_watts(
                client=client,
                location_uuid=location_uuid,
            )
            if capacity_watts <= 0:
                logger.error(
                    f"Location {location_uuid} has capacity_watts={capacity_watts}. "
                    "Cannot convert MW values to fractions - aborting.",
                )
                return
            logger.info(
                f"Location capacity: {capacity_watts:,} W "
                f"({capacity_watts / 1_000_000:.3f} MW)",
            )
        except Exception:
            logger.exception("Failed to fetch location capacity from Data Platform.")
            return

        # -------------------------------------------------------------- #
        # Load MAE scorecard                                           #
        # -------------------------------------------------------------- #
        scorecard_path = os.path.join(
            os.path.dirname(__file__), _cfg.scorecard_path,
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

        # -------------------------------------------------------------- #
        # Calculate delay-adjusted blend weights                       #
        # -------------------------------------------------------------- #
        logger.info("Calculating delay-adjusted blend weights.")
        try:
            weights_df = await get_nl_blend_weights(
                t0=t0,
                location_uuid=location_uuid,
                df_mae=df_mae,
                max_horizon=max_horizon,
                client=client,
            )
            logger.info(f"Blend weights calculated:\n{weights_df.head(10)}")
        except Exception:
            logger.exception("Failed to calculate blend weights.")
            return

        # -------------------------------------------------------------- #
        # Fetch forecast timeseries and produce blended values         #
        # -------------------------------------------------------------- #
        logger.info("Fetching raw forecast values and blending.")
        try:
            blended_df = await get_blend_forecast_values_latest(
                location_uuid=location_uuid,
                weights_df=weights_df,
                client=client,
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

        # -------------------------------------------------------------- #
        # Save results                                                 #

        # -------------------------------------------------------------- #
        await _save_forecasts(
            client=client,
            t0=t0,
            location_uuid=location_uuid,
            blended_df=blended_df,
        )


async def _save_forecasts(
    client: dp.DataPlatformDataServiceStub,
    t0: pd.Timestamp,
    location_uuid: str,
    blended_df: pd.DataFrame,
) -> None:
    """Persists the blended forecast to the Data Platform.

    Args:
        client:         Active Data Platform gRPC client stub.
        t0:             Blend reference time (UTC); used as the forecast init_time.
        location_uuid:  DP location UUID to write forecasts under.
        blended_df:     DataFrame with columns [target_time,
                        expected_power_generation_megawatts, p10_mw (opt),
                        p90_mw (opt)].
    """
    n_rows = len(blended_df)
    has_p10 = "p10_mw" in blended_df.columns
    has_p90 = "p90_mw" in blended_df.columns
    n_p10 = int(blended_df["p10_mw"].notna().sum()) if has_p10 else 0
    n_p90 = int(blended_df["p90_mw"].notna().sum()) if has_p90 else 0

    logger.info(
        f"Blended forecast summary: {n_rows} rows | "
        f"p50={n_rows} | p10={n_p10} | p90={n_p90} rows with valid values.",
    )

    # ------------------------------------------------------------------ #
    # Build the DP value objects (shared between both write paths)        #
    # ------------------------------------------------------------------ #
    try:
        forecast_values = build_forecast_value_objects(
            blended_df=blended_df,
            init_time_utc=t0.to_pydatetime(),
        )
    except Exception:
        logger.exception("Failed to build DP forecast value objects - skipping save.")
        return

    if not forecast_values:
        logger.warning("No forecast value objects produced - skipping save.")
        return

    # ------------------------------------------------------------------ #
    # Resolve / create the forecaster record                              #
    # ------------------------------------------------------------------ #
    try:
        forecaster = await create_forecaster_if_not_exists(
            client=client,
            model_tag=NL_BLEND_FORECASTER_NAME,
        )
        logger.info(
            f"Forecaster resolved: {forecaster.forecaster_name!r} "
            f"v{forecaster.forecaster_version}",
        )
    except Exception:
        logger.exception("Failed to resolve/create nl_blend forecaster - skipping save.")
        return

    base_request = dp.CreateForecastRequest(
        forecaster=forecaster,
        location_uuid=location_uuid,
        energy_source=dp.EnergySource.SOLAR,
        init_time_utc=t0.to_pydatetime(),
        values=forecast_values,
    )

    # ------------------------------------------------------------------ #
    # Always write to Data Platform                                       #
    # ------------------------------------------------------------------ #
    logger.info(
        f"Saving {n_rows} rows to Data Platform "
        f"(forecaster='nl_blend', t0={t0}, location={location_uuid}) - "
        f"p50={n_rows}, p10={n_p10}, p90={n_p90} valid rows.",
    )
    try:
        await client.create_forecast(base_request)
        logger.info("Forecast write succeeded.")
    except Exception:
        logger.exception("Failed to write forecast.")


if __name__ == "__main__":
    asyncio.run(run_blend_app())
