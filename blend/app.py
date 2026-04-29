"""Main entry point for the NL forecast blending application."""
import asyncio
import logging
import os

import pandas as pd
from dp_sdk.ocf import dp

from blend.blend import get_blend_forecast_values_latest
from blend.config import load_blend_config
from blend.data_platform import (
    build_forecast_value_objects,
)
from blend.init_times import load_nl_mae_scorecard
from blend.weights import get_blend_weights
from site_forecast_app.save.data_platform import (
    create_forecaster_if_not_exists,
    fetch_dp_location_map,
    get_dataplatform_client,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("blend_app")

# location_type key that identifies the national location in the DP location map.
NL_NATIONAL_LOCATION_KEY = "nl_national"


async def run_blend_app() -> None:
    """Main execution point for the NL Blend app.

    Steps:
    1. Determine blend reference time (t0)
    2. Fetch full location map from Data Platform (national + regional)
    3. Load the MAE scorecard
    4. Calculate national blend weights (used for national location)
    5. Calculate regional blend weights (used for all regional locations)
    6. For each location: fetch + blend + save
    """
    _cfg = load_blend_config()
    logger.info("Starting NL Blend execution.")

    # ------------------------------------------------------------------ #
    # Determine blend reference time - floor to 15-min boundary          #
    # ------------------------------------------------------------------ #
    t0 = pd.Timestamp.utcnow().floor("15min")
    logger.info(f"Blend t0: {t0}")

    # ------------------------------------------------------------------ #
    # Open a single Data Platform connection for the entire run           #
    # ------------------------------------------------------------------ #
    async with get_dataplatform_client() as client:

        # -------------------------------------------------------------- #
        # Fetch full location map (national + all regional)               #
        # -------------------------------------------------------------- #
        logger.info("Fetching location map from Data Platform.")
        try:
            dp_loc_map = await fetch_dp_location_map(client)
            if not dp_loc_map:
                logger.error("Data Platform returned an empty location map. Cannot continue.")
                return
            logger.info(f"Fetched {len(dp_loc_map)} location(s) from Data Platform.")
        except Exception:
            logger.exception("Failed to connect to Data Platform while fetching location map.")
            return

        # Identify the national location UUID
        national_location_uuid = dp_loc_map.get(NL_NATIONAL_LOCATION_KEY)
        if national_location_uuid is None:
            # Fallback: treat the first entry as national (original behaviour).
            national_location_uuid = next(iter(dp_loc_map.values()))
            logger.warning(
                f"Key '{NL_NATIONAL_LOCATION_KEY}' not found in location map; "
                f"using first entry as national: {national_location_uuid}",
            )

        # -------------------------------------------------------------- #
        # Load MAE scorecard (shared across all locations)                #
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
        # Calculate national blend weights for the national location  #
        # -------------------------------------------------------------- #
        logger.info("Calculating national blend weights.")
        try:
            national_weights_df = await get_blend_weights(
                t0=t0,
                location_uuid=national_location_uuid,
                df_mae=df_mae,
                max_horizon=max_horizon,
                client=client,
            )
            logger.info(f"National blend weights calculated:\n{national_weights_df.head(10)}")
        except Exception:
            logger.exception("Failed to calculate national blend weights.")
            return


        # -------------------------------------------------------------- #
        # Blend and save for the national location only                   #
        # -------------------------------------------------------------- #
        location_key = NL_NATIONAL_LOCATION_KEY
        location_uuid = national_location_uuid
        logger.info(
            f"Blending forecasts for national location '{location_key}' "
            f"(uuid={location_uuid})",
        )

        try:
            blended_df = await get_blend_forecast_values_latest(
                location_uuid=location_uuid,
                weights_df=national_weights_df,
                client=client,
                start_datetime=t0,
            )

            if blended_df.empty:
                logger.warning(
                    f"Blended timeseries is empty for location '{location_key}'. "
                    "This is expected in dev when no forecast megawatts are stored.",
                )
                return

            logger.info(
                f"Blended timeseries for '{location_key}' "
                f"(first 5 rows):\n{blended_df.head(5)}",
            )

            await _save_forecasts(
                client=client,
                t0=t0,
                location_uuid=location_uuid,
                location_key=location_key,
                blended_df=blended_df,
                forecaster_name=_cfg.forecaster_name,
            )

        except Exception:
            logger.exception(
                f"Failed to blend or save forecasts for national location '{location_key}' "
                f"(uuid={location_uuid}).",
            )


async def _save_forecasts(
    client: dp.DataPlatformDataServiceStub,
    t0: pd.Timestamp,
    location_uuid: str,
    location_key: str,
    blended_df: pd.DataFrame,
    forecaster_name: str,
) -> None:
    """Persists the blended forecast to the Data Platform.

    Args:
        client:           Active Data Platform gRPC client stub.
        t0:               Blend reference time (UTC); used as the forecast init_time.
        location_uuid:    DP location UUID to write forecasts under.
        location_key:     Human-readable location identifier.
        blended_df:       DataFrame with columns [target_time,
                          expected_power_generation_megawatts, p10_mw (opt),
                          p90_mw (opt)].
        forecaster_name:  Forecaster tag written to the Data Platform.
    """
    n_rows = len(blended_df)
    has_p10 = "p10_mw" in blended_df.columns
    has_p90 = "p90_mw" in blended_df.columns
    n_p10 = int(blended_df["p10_mw"].notna().sum()) if has_p10 else 0
    n_p90 = int(blended_df["p90_mw"].notna().sum()) if has_p90 else 0

    logger.info(
        f"Blended forecast summary for '{location_key}': {n_rows} rows | "
        f"p50={n_rows} | p10={n_p10} | p90={n_p90} rows with valid values.",
    )

    # ------------------------------------------------------------------ #
    # Build the DP value objects                                          #
    # ------------------------------------------------------------------ #
    try:
        forecast_values = build_forecast_value_objects(
            blended_df=blended_df,
            init_time_utc=t0.to_pydatetime(),
        )
    except Exception:
        logger.exception(
            f"Failed to build DP forecast value objects for '{location_key}' - skipping save.",
        )
        return

    if not forecast_values:
        logger.warning(
            f"No forecast value objects produced for '{location_key}' - skipping save.",
        )
        return

    # ------------------------------------------------------------------ #
    # Resolve / create the forecaster record                              #
    # ------------------------------------------------------------------ #
    try:
        forecaster = await create_forecaster_if_not_exists(
            client=client,
            model_tag=forecaster_name,
        )
        logger.info(
            f"Forecaster resolved: {forecaster.forecaster_name!r} "
            f"v{forecaster.forecaster_version}",
        )
    except Exception:
        logger.exception(
            f"Failed to resolve/create blend forecaster for '{location_key}' - skipping save.",
        )
        return

    base_request = dp.CreateForecastRequest(
        forecaster=forecaster,
        location_uuid=location_uuid,
        energy_source=dp.EnergySource.SOLAR,
        init_time_utc=t0.to_pydatetime(),
        values=forecast_values,
    )

    # ------------------------------------------------------------------ #
    # Write to Data Platform                                              #
    # ------------------------------------------------------------------ #
    logger.info(
        f"Saving {n_rows} rows to Data Platform "
        f"(forecaster='blend', t0={t0}, location='{location_key}') - "
        f"p50={n_rows}, p10={n_p10}, p90={n_p90} valid rows.",
    )
    try:
        await client.create_forecast(base_request)
        logger.info(f"Forecast write succeeded for '{location_key}'.")
    except Exception:
        logger.exception(f"Failed to write forecast for '{location_key}'.")


if __name__ == "__main__":
    asyncio.run(run_blend_app())
