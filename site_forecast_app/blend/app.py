"""Main entry point for the NL forecast blending application."""
import asyncio
import logging
import os

import pandas as pd
from dp_sdk.ocf import dp

from site_forecast_app.blend.blend import get_blend_forecast_values_latest
from site_forecast_app.blend.config import load_blend_config
from site_forecast_app.blend.data_platform import (
    build_forecast_value_objects,
)
from site_forecast_app.blend.init_times import load_nl_mae_scorecard
from site_forecast_app.blend.weights import get_blend_weights
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
    2. Fetch full location map from Data Platform
    3. Load the MAE scorecard
    4. Calculate blend weights and run blend for main models
    5. Save main forecast under {forecaster_name}
    6. If use_adjuster=True:
       - Calculate blend weights and run blend for adjuster models
         ({model_name}_adjust) — full pipeline runs unchanged
       - Save adjuster blend under {forecaster_name}_adjust
    """
    _cfg = load_blend_config()
    logger.info(
        f"Starting NL Blend execution. "
        f"use_adjuster={_cfg.use_adjuster}, "
        f"forecaster='{_cfg.forecaster_name}'",
    )

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
        # Main blend                                                      #
        # -------------------------------------------------------------- #
        await _run_blend_pass(
            client=client,
            t0=t0,
            location_uuid=national_location_uuid,
            location_key=NL_NATIONAL_LOCATION_KEY,
            df_mae=df_mae,
            max_horizon=max_horizon,
            forecaster_name=_cfg.forecaster_name,
        )

        # -------------------------------------------------------------- #
        # Adjuster blend (only if use_adjuster=True in config)           #
        # Weights are computed from the same module-level constants.     #
        # Weight column names are suffixed with '_adjust' so that        #
        # get_blend_forecast_values_latest fetches {model}_adjust from   #
        # the Data Platform instead of the base model forecasters.       #
        # -------------------------------------------------------------- #
        if _cfg.use_adjuster:
            logger.info("use_adjuster=True: running adjuster blend pass.")
            await _run_blend_pass(
                client=client,
                t0=t0,
                location_uuid=national_location_uuid,
                location_key=NL_NATIONAL_LOCATION_KEY,
                df_mae=df_mae,
                max_horizon=max_horizon,
                forecaster_name=_cfg.adjuster_forecaster_name,
                use_adjuster=True,
            )


async def _run_blend_pass(
    client: dp.DataPlatformDataServiceStub,
    t0: pd.Timestamp,
    location_uuid: str,
    location_key: str,
    df_mae: pd.DataFrame,
    max_horizon: pd.Timedelta,
    forecaster_name: str,
    use_adjuster: bool = False,
) -> None:
    """Runs the full blend pipeline for one set of models and saves the result.

    Shared by the main blend pass and the adjuster blend pass.

    Blend weights are always computed from the module-level constants in
    ``weights.py`` (NL_BACKUP_MODEL / NL_NATIONAL_CANDIDATE_MODELS).
    When *use_adjuster* is True, the weight column names are renamed with an
    ``_adjust`` suffix before fetching, so that
    :func:`get_blend_forecast_values_latest` fetches ``{model}_adjust``
    forecasters from the Data Platform instead of the base model forecasters.

    Args:
        client:          Active Data Platform gRPC client stub.
        t0:              Blend reference time (UTC).
        location_uuid:   DP location UUID to blend and save for.
        location_key:    Human-readable location identifier (for logging).
        df_mae:          (horizon x model) MAE scorecard.
        max_horizon:     Maximum scorecard horizon.
        forecaster_name: Forecaster tag to save under.
        use_adjuster:    When True, fetches {model}_adjust forecasters and
                         saves under {forecaster_name} (caller sets the
                         correct adjuster forecaster name).
    """
    log_prefix = "adjuster" if use_adjuster else "blend"
    logger.info(
        f"[{log_prefix}] Starting blend pass for '{location_key}' "
        f"(forecaster='{forecaster_name}', use_adjuster={use_adjuster})",
    )

    # Weights are always computed from the module-level constants.
    try:
        weights_df = await get_blend_weights(
            t0=t0,
            location_uuid=location_uuid,
            df_mae=df_mae,
            max_horizon=max_horizon,
            client=client,
        )
        logger.info(f"[{log_prefix}] Blend weights calculated:\n{weights_df.head(10)}")
    except Exception:
        logger.exception(f"[{log_prefix}] Failed to calculate blend weights.")
        return

    # For the adjuster pass: rename columns so DP fetches {model}_adjust.
    if use_adjuster:
        weights_df = weights_df.rename(
            columns={col: f"{col}_adjust" for col in weights_df.columns},
        )
        logger.info(
            f"[{log_prefix}] Weight columns renamed with '_adjust' suffix: "
            f"{list(weights_df.columns)}",
        )

    # Fetch and blend
    try:
        blended_df = await get_blend_forecast_values_latest(
            location_uuid=location_uuid,
            weights_df=weights_df,
            client=client,
            start_datetime=t0,
        )
    except Exception:
        logger.exception(f"[{log_prefix}] Failed to fetch or blend forecast timeseries.")
        return

    if blended_df.empty:
        logger.warning(
            f"[{log_prefix}] Blended timeseries is empty for '{location_key}'. "
            "This is expected in dev when no forecast megawatts are stored.",
        )
        return

    logger.info(
        f"[{log_prefix}] Blended timeseries for '{location_key}' "
        f"(first 5 rows):\n{blended_df.head(5)}",
    )

    await _save_forecasts(
        client=client,
        t0=t0,
        location_uuid=location_uuid,
        location_key=location_key,
        blended_df=blended_df,
        forecaster_name=forecaster_name,
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
        client:          Active Data Platform gRPC client stub.
        t0:              Blend reference time (UTC).
        location_uuid:   DP location UUID to write forecasts under.
        location_key:    Human-readable location identifier (for logging only).
        blended_df:      DataFrame with columns [target_time,
                         expected_power_generation_megawatts, p10_mw (opt),
                         p90_mw (opt)].
        forecaster_name: Forecaster tag written to the Data Platform.
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

    # Build DP value objects
    try:
        forecast_values = build_forecast_value_objects(
            blended_df=blended_df,
            init_time_utc=t0.to_pydatetime(),
        )
    except Exception:
        logger.exception(
            f"Failed to build DP forecast value objects for "
            f"'{location_key}' - skipping save.",
        )
        return

    if not forecast_values:
        logger.warning(
            f"No forecast value objects produced for "
            f"'{location_key}' - skipping save.",
        )
        return

    # Resolve / create forecaster record
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
            f"Failed to resolve/create forecaster '{forecaster_name}' "
            f"for '{location_key}' - skipping save.",
        )
        return

    # Write to Data Platform
    logger.info(
        f"Saving {n_rows} rows to Data Platform "
        f"(forecaster='{forecaster_name}', t0={t0}, location='{location_key}') - "
        f"p50={n_rows}, p10={n_p10}, p90={n_p90} valid rows.",
    )
    try:
        await client.create_forecast(
            dp.CreateForecastRequest(
                forecaster=forecaster,
                location_uuid=location_uuid,
                energy_source=dp.EnergySource.SOLAR,
                init_time_utc=t0.to_pydatetime(),
                values=forecast_values,
            ),
        )
        logger.info(f"Forecast write succeeded for '{location_key}'.")
    except Exception:
        logger.exception(f"Failed to write forecast for '{location_key}'.")


if __name__ == "__main__":
    asyncio.run(run_blend_app())
