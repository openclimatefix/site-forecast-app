"""Functions for saving forecasts."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from importlib.metadata import version
from typing import TYPE_CHECKING
from uuid import UUID

import pandas as pd
from betterproto.lib.google.protobuf import Struct
from dp_sdk.ocf import dp
from dp_sdk.ocf.dp import EnergySource, LocationType
from grpclib.client import Channel

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

# Type alias for the Data Platform client stub
DataPlatformClient = dp.DataPlatformDataServiceStub

def save_forecast(
    _db_session: Session,
    forecast: dict,
    _write_to_db: bool,
    ml_model_name: str | None = None,
    _ml_model_version: str | None = None,
    _use_adjuster: bool = True,
    _adjuster_average_minutes: int | None = 60,
) -> None:
    """Saves a forecast for a given site & timestamp.

    Args:
            _db_session: A SQLAlchemy session (unused, for backward compatibility)
            forecast: a forecast dict containing forecast meta and predicted values
            _write_to_db: If true, forecast values are written to db, otherwise to stdout
            ml_model_name: Name of the ML model used for the forecast
            _ml_model_version: Version of the ML model used for the forecast
            _use_adjuster: Make new model, adjusted by last 7 days of ME values
            _adjuster_average_minutes: The number of minutes that results are average over
                when calculating adjuster values

    Raises:
            IOError: An error if database save fails
    """
    log.info(f"Saving forecast for location_id={forecast['meta']['location_uuid']}...")

    forecast_meta = {
        "location_uuid": forecast["meta"]["location_uuid"],
        "timestamp_utc": forecast["meta"]["timestamp"],
        "forecast_version": forecast["meta"]["version"],
    }
    forecast_values_df = pd.DataFrame(forecast["values"])
    forecast_values_df["horizon_minutes"] = (
        (forecast_values_df["start_utc"] - forecast_meta["timestamp_utc"]) / pd.Timedelta("60s")
    ).astype("int")

    # if write_to_db:
    #     insert_forecast_values(
    #         db_session,
    #         forecast_meta,
    #         forecast_values_df,
    #         ml_model_name=ml_model_name,
    #         ml_model_version=ml_model_version,
    #     )

    # if use_adjuster:
    #     log.info(f"Adjusting forecast for location_id={forecast_meta['location_uuid']}...")
    #     forecast_values_df_adjust = adjust_forecast_with_adjuster(
    #         db_session,
    #         forecast_meta,
    #         forecast_values_df,
    #         ml_model_name=ml_model_name,
    #         average_minutes=adjuster_average_minutes,
    #     )

    #     if write_to_db:
    #         insert_forecast_values(
    #             db_session,
    #             forecast_meta,
    #             forecast_values_df_adjust,
    #             ml_model_name=f"{ml_model_name}_adjust",
    #             ml_model_version=ml_model_version,
    #         )

    output = f"Forecast for location_id={forecast_meta['location_uuid']},\
               timestamp={forecast_meta['timestamp_utc']},\
               version={forecast_meta['forecast_version']}:"
    log.info(output.replace("  ", ""))
    log.info(f"\n{forecast_values_df.to_string()}\n")

    if os.getenv("SAVE_TO_DATA_PLATFORM", "false").lower() == "true":
        log.info("Saving to Data Platform...")

        async def run_async_save() -> None:
            channel = Channel(
                host=os.getenv("DP_HOST", "localhost"),
                port=int(os.getenv("DP_PORT", "50051")),
            )
            client = dp.DataPlatformDataServiceStub(channel)
            try:
                await save_forecast_to_dataplatform(
                    forecast_df=forecast_values_df,
                    location_uuid=UUID(str(forecast_meta["location_uuid"])),
                    model_tag=ml_model_name if ml_model_name else "default-model",
                    init_time_utc=forecast_meta["timestamp_utc"],
                    client=client,
                )
            finally:
                channel.close()

        try:
            asyncio.run(run_async_save())
        except Exception as e:
            log.error(f"Failed to save to Data Platform: {e}")

async def _create_forecaster_if_not_exists(
    client: DataPlatformClient,
    model_tag: str,
) -> dp.Forecaster:
    """Create the current forecaster if it does not exist."""
    forecaster_name = model_tag.replace("-", "_")
    app_version = version("site-forecast-app")

    list_forecasters_request = dp.ListForecastersRequest(
        forecaster_names_filter=[forecaster_name],
    )
    list_forecasters_response = await client.list_forecasters(list_forecasters_request)

    if len(list_forecasters_response.forecasters) > 0:
        filtered_forecasters = [
            f
            for f in list_forecasters_response.forecasters
            if f.forecaster_version == app_version
        ]
        if len(filtered_forecasters) == 1:
            return filtered_forecasters[0]
        else:
            update_forecaster_request = dp.UpdateForecasterRequest(
                name=forecaster_name,
                new_version=app_version,
            )
            update_forecaster_response = await client.update_forecaster(update_forecaster_request)
            return update_forecaster_response.forecaster
    else:
        create_forecaster_request = dp.CreateForecasterRequest(
            name=forecaster_name,
            version=app_version,
        )
        create_forecaster_response = await client.create_forecaster(create_forecaster_request)
        return create_forecaster_response.forecaster


def _limit_adjuster(delta_fraction: float, value_fraction: float, capacity_mw: float) -> float:
    """Limit the adjuster to 10% of forecast and max 1000 MW."""
    # limit adjusted fractions to 10% of fv.p50_fraction
    max_delta = 0.1 * value_fraction
    if delta_fraction > max_delta:
        delta_fraction = max_delta
    elif delta_fraction < -max_delta:
        delta_fraction = -max_delta

    # limit adjust to 1000 MW
    max_delta_absolute = 1000.0 / capacity_mw if capacity_mw > 0 else 0
    if delta_fraction > max_delta_absolute:
        delta_fraction = max_delta_absolute
    elif delta_fraction < -max_delta_absolute:
        delta_fraction = -max_delta_absolute

    return delta_fraction


async def _make_forecaster_adjuster(
    client: DataPlatformClient,
    location_uuid: str,
    init_time_utc: datetime,
    forecast_values: list[dp.CreateForecastRequestForecastValue],
    model_tag: str,
    forecaster: dp.Forecaster,
) -> dp.CreateForecastRequest:
    """Make a forecaster adjuster based on week average deltas."""
    # get delta values
    deltas_request = dp.GetWeekAverageDeltasRequest(
        location_uuid=location_uuid,
        energy_source=dp.EnergySource.SOLAR,
        pivot_timestamp_utc=init_time_utc.replace(tzinfo=UTC),
        forecaster=forecaster,
        observer_name="pvlive_day_after",
    )
    deltas_response = await client.get_week_average_deltas(deltas_request)
    deltas = deltas_response.deltas

    # adjust the current forecast values
    new_forecast_values = []
    for fv in forecast_values:
        horizon_mins = fv.horizon_mins
        delta_fractions = [d.delta_fraction for d in deltas if d.horizon_mins == horizon_mins]
        delta_fraction = delta_fractions[0] if len(delta_fractions) > 0 else 0

        # get location
        location = await client.get_location(
            dp.GetLocationRequest(
                location_uuid=location_uuid,
                energy_source=dp.EnergySource.SOLAR,
                include_geometry=False,
            ),
        )
        capacity_mw = location.effective_capacity_watts / 1_000_000.0

        # limit adjuster
        delta_fraction = _limit_adjuster(
            delta_fraction=delta_fraction,
            value_fraction=fv.p50_fraction,
            capacity_mw=capacity_mw,
        )

        # delta values are forecast - observed, so we need to subtract
        new_p50 = max(0.0, min(1.0, fv.p50_fraction - delta_fraction))

        # adjust p10 and p90s
        new_other_statistics = {}
        for key, val in fv.other_statistics_fractions.items():
            new_val = max(0.0, min(1.0, val - delta_fraction))
            new_other_statistics[key] = new_val

        new_forecast_values.append(
            dp.CreateForecastRequestForecastValue(
                horizon_mins=fv.horizon_mins,
                p50_fraction=new_p50,
                metadata=fv.metadata,
                other_statistics_fractions=new_other_statistics,
            ),
        )

    # make new forecast
    forecaster = await _create_forecaster_if_not_exists(
        client=client,
        model_tag=model_tag + "_adjust",
    )

    # make forecast
    adjusted_forecast_request = dp.CreateForecastRequest(
        forecaster=forecaster,
        location_uuid=location_uuid,
        energy_source=dp.EnergySource.SOLAR,
        init_time_utc=init_time_utc.replace(tzinfo=UTC),
        values=new_forecast_values,
    )

    return adjusted_forecast_request


async def save_forecast_to_dataplatform(
    forecast_df: pd.DataFrame,
    location_uuid: UUID,
    model_tag: str,
    init_time_utc: datetime,
    client: DataPlatformClient,
) -> None:
    """Save forecast to data platform."""
    # Ensure init_time_utc is timezone aware
    if isinstance(init_time_utc, pd.Timestamp):
        if init_time_utc.tz is None:
            init_time_utc = init_time_utc.tz_localize("UTC")
    elif init_time_utc.tzinfo is None:
        init_time_utc = init_time_utc.replace(tzinfo=UTC)

    log.info("Writing to data platform")

    # Resolve UUID - check if there's a mapping from legacy to DP UUID
    legacy_uuid_str = str(location_uuid)
    target_uuid_str = legacy_uuid_str

    try:
        # Fetch locations to find mapping
        # Note: This is inefficient for many sites/calls, but acceptable for
        # current scale/CLI usage.
        resp = await client.list_locations(dp.ListLocationsRequest())

        found = False
        for loc in resp.locations:
            if loc.metadata and loc.metadata.fields:
                val = loc.metadata.fields.get("legacy_uuid")
                if val and val.string_value == legacy_uuid_str:
                    target_uuid_str = loc.location_uuid
                    found = True
                    break

        if found:
            log.info(
                f"Mapped legacy UUID {legacy_uuid_str} to DP UUID {target_uuid_str}",
            )
        else:
            log.debug(
                f"Could not find DP location mapping for UUID {legacy_uuid_str}. "
                "Using original.",
            )

    except Exception as e:
        log.warning(f"Failed to lookup UUID mapping: {e}. Proceeding with original UUID.")

    try:
        # Get or create forecaster
        forecaster = await _create_forecaster_if_not_exists(
            client=client,
            model_tag=model_tag,
        )

        # Load Location
        location = await client.get_location(
            dp.GetLocationRequest(
                location_uuid=target_uuid_str,
                energy_source=dp.EnergySource.SOLAR,
                include_geometry=False,
            ),
        )

        # Prepare forecast values
        capacity_watts = location.effective_capacity_watts
        if capacity_watts == 0:
            log.warning(f"Location {location_uuid} has 0 capacity, skipping data platform save")
            return

        forecast_values = []
        for _, row in forecast_df.iterrows():
            # Calculate horizon if not present
            if "horizon_minutes" in row and pd.notna(row["horizon_minutes"]):
                horizon_mins = int(row["horizon_minutes"])
            else:
                # Ensure both are pd.Timestamp
                start_ts = pd.Timestamp(row["start_utc"])
                init_ts = pd.Timestamp(init_time_utc)

                # Make start_ts UTC aware
                if start_ts.tz is None:
                    start_ts = start_ts.tz_localize("UTC")
                else:
                    start_ts = start_ts.tz_convert("UTC")

                # Make init_ts UTC aware
                if init_ts.tz is None:
                    init_ts = init_ts.tz_localize("UTC")
                else:
                    init_ts = init_ts.tz_convert("UTC")

                horizon_mins = int(
                    (start_ts - init_ts).total_seconds() / 60,
                )

            # Convert Power kW to Fraction
            # Fraction = (kW * 1000) / Watts
            p50_fraction = (row["forecast_power_kw"] * 1000) / capacity_watts
            # Clamp to [0, 1.0] to satisfy validation
            p50_fraction = max(0.0, min(p50_fraction, 1.0))

            other_stats = {}
            if row.get("probabilistic_values"):
                try:
                    probs = json.loads(row["probabilistic_values"])
                    for key, val_kw in probs.items():
                        frac = (val_kw * 1000) / capacity_watts
                        # Clamp to [0, 1.0] to satisfy validation
                        other_stats[key] = max(0.0, min(frac, 1.0))
                except (json.JSONDecodeError, TypeError):
                    log.warning("Failed to parse probabilistic_values")

            forecast_values.append(
                dp.CreateForecastRequestForecastValue(
                    horizon_mins=horizon_mins,
                    p50_fraction=p50_fraction,
                    metadata=Struct().from_pydict({}),
                    other_statistics_fractions=other_stats,
                ),
            )

        if len(forecast_values) > 0:
            forecast_request = dp.CreateForecastRequest(
                forecaster=forecaster,
                location_uuid=target_uuid_str,
                energy_source=dp.EnergySource.SOLAR,
                init_time_utc=init_time_utc,
                values=forecast_values,
            )
            await client.create_forecast(forecast_request)

    except Exception as e:
        import traceback
        log.error(f"Failed to save forecast to data platform with error {e}")
        log.error(traceback.format_exc())

