"""Functions for saving forecasts."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from importlib.metadata import version
from uuid import UUID, uuid4

import pandas as pd
from betterproto.lib.google.protobuf import Struct
from dp_sdk.ocf import dp
from grpclib.client import Channel
from pvsite_datamodel.read.model import get_or_create_model
from pvsite_datamodel.sqlmodels import ForecastSQL, ForecastValueSQL
from sqlalchemy.orm import Session  # noqa: TC002

from site_forecast_app.adjuster import adjust_forecast_with_adjuster

log = logging.getLogger(__name__)

MAX_DELTA_ABSOLUTE = 0.1


# Type alias for the Data Platform client stub
DataPlatformClient = dp.DataPlatformDataServiceStub


def _insert_forecast_values(
    db_session: Session,
    forecast_meta: dict,
    forecast_values_df: pd.DataFrame,
    *,
    ml_model_name: str | None,
    ml_model_version: str | None,
) -> None:
    """Insert a forecast + its values into the DB.

    This mirrors how fixtures insert data in tests and is intentionally minimal.
    """
    model_name = ml_model_name or "default"
    model_version = ml_model_version or "0.0.0"
    ml_model = get_or_create_model(db_session, model_name, model_version)

    forecast_uuid = uuid4()
    created_utc = forecast_meta["timestamp_utc"]

    forecast = ForecastSQL(
        location_uuid=str(forecast_meta["location_uuid"]),
        timestamp_utc=forecast_meta["timestamp_utc"],
        forecast_version=str(forecast_meta["forecast_version"]),
        created_utc=created_utc,
        forecast_uuid=forecast_uuid,
    )
    db_session.add(forecast)

    values_to_add: list[ForecastValueSQL] = []
    for _, row in forecast_values_df.iterrows():
        probabilistic_values = row.get("probabilistic_values")
        if isinstance(probabilistic_values, dict):
            probabilistic_values = json.dumps(probabilistic_values)
        elif probabilistic_values is not None and not isinstance(probabilistic_values, str):
            # Best-effort: coerce unknown types to JSON.
            probabilistic_values = json.dumps(probabilistic_values)

        values_to_add.append(
            ForecastValueSQL(
                horizon_minutes=int(row["horizon_minutes"]),
                forecast_power_kw=float(row["forecast_power_kw"]),
                start_utc=row["start_utc"],
                end_utc=row["end_utc"],
                ml_model_uuid=ml_model.model_uuid,
                forecast_uuid=forecast_uuid,
                created_utc=created_utc,
                probabilistic_values=probabilistic_values,
            ),
        )

    db_session.add_all(values_to_add)
    # Ensure queries in the same session can see newly added rows.
    db_session.commit()


def _write_forecast_to_db(
    db_session: Session,
    forecast_meta: dict,
    forecast_values_df: pd.DataFrame,
    *,
    write_to_db: bool,
    ml_model_name: str | None,
    ml_model_version: str | None,
) -> None:
    """Write a forecast dataframe to DB when enabled."""
    if not write_to_db:
        return

    _insert_forecast_values(
        db_session,
        forecast_meta,
        forecast_values_df,
        ml_model_name=ml_model_name,
        ml_model_version=ml_model_version,
    )

def save_forecast(
    db_session: Session,
    forecast: dict,
    write_to_db: bool = False,
    ml_model_name: str | None = None,
    ml_model_version: str | None = None,
    use_adjuster: bool = True,
    adjuster_average_minutes: int | None = 60,
) -> None:
    """Saves a forecast for a given site & timestamp.

    Args:
            db_session: A SQLAlchemy session
            forecast: a forecast dict containing forecast meta and predicted values
            write_to_db: If true, forecast values are written to db, otherwise to stdout
            ml_model_name: Name of the ML model used for the forecast
            ml_model_version: Version of the ML model used for the forecast
            use_adjuster: Make new model, adjusted by last 7 days of ME values
            adjuster_average_minutes: The number of minutes that results are average over
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

    _write_forecast_to_db(
        db_session,
        forecast_meta,
        forecast_values_df,
        write_to_db=bool(write_to_db),
        ml_model_name=ml_model_name,
        ml_model_version=ml_model_version,
    )

    if use_adjuster and ml_model_name is not None:
        _adjust_and_save_forecast(
            db_session,
            forecast_meta,
            forecast_values_df,
            ml_model_name=ml_model_name,
            ml_model_version=ml_model_version,
            adjuster_average_minutes=adjuster_average_minutes,
            write_to_db=bool(write_to_db),
        )

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
                    use_adjuster=use_adjuster and ml_model_name is not None,
                )
            except Exception as e:
                import traceback
                log.error(f"Failed to save forecast to data platform with error {e}")
                log.error(traceback.format_exc())
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
    raw_version = version("site-forecast-app")
    # DP validates version with a restricted charset; normalize local package versions
    # (which may contain e.g. '+' build metadata) to a compatible value.
    app_version = "".join(
        ch if (ch.isalnum() or ch in "._-") else "."
        for ch in raw_version.lower()
    )
    while ".." in app_version:
        app_version = app_version.replace("..", ".")
    app_version = app_version.strip("._-")
    if len(app_version) < 2:
        app_version = "0.0"

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


def _limit_adjuster(delta_fraction: float, value_fraction: float) -> float:
    """Limit adjuster magnitude to a fraction of forecast and absolute cap."""
    max_delta = 0.1 * value_fraction
    delta_fraction = min(max(delta_fraction, -max_delta), max_delta)

    delta_fraction = min(max(delta_fraction, -MAX_DELTA_ABSOLUTE), MAX_DELTA_ABSOLUTE)
    return delta_fraction


def add_or_convert_to_utc(timestamp: object) -> pd.Timestamp:
    """Ensure a timestamp is a timezone-aware UTC pd.Timestamp."""
    ts = pd.Timestamp(timestamp)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    return ts


def _adjust_and_save_forecast(
    db_session: Session,
    forecast_meta: dict,
    forecast_values_df: pd.DataFrame,
    ml_model_name: str,
    ml_model_version: str | None,
    adjuster_average_minutes: int | None,
    write_to_db: bool,
) -> None:
    """Adjust forecast using the adjuster and save to DB."""
    log.info(f"Adjusting forecast for location_id={forecast_meta['location_uuid']}...")
    try:
        forecast_values_df_adjust = adjust_forecast_with_adjuster(
            db_session,
            forecast_meta,
            forecast_values_df,
            ml_model_name=ml_model_name,
            average_minutes=adjuster_average_minutes,
        )
        log.info(f"Adjusted forecast shape: {forecast_values_df_adjust.shape}")

        _write_forecast_to_db(
            db_session,
            forecast_meta,
            forecast_values_df_adjust,
            write_to_db=write_to_db,
            ml_model_name=f"{ml_model_name}_adjust",
            ml_model_version=ml_model_version,
        )
    except Exception as e:
        import traceback

        log.error(f"Failed to adjust/save forecast for {ml_model_name}: {e}")
        log.error(traceback.format_exc())


async def _make_forecaster_adjuster(
    client: DataPlatformClient,
    location_uuid: str,
    init_time_utc: datetime,
    forecast_values: list[dp.CreateForecastRequestForecastValue],
    model_tag: str,
    forecaster: dp.Forecaster,
) -> dp.CreateForecastRequest:
    """Create adjusted forecast request using week-average deltas."""
    deltas_request = dp.GetWeekAverageDeltasRequest(
        location_uuid=location_uuid,
        energy_source=dp.EnergySource.SOLAR,
        pivot_timestamp_utc=init_time_utc.replace(tzinfo=UTC),
        forecaster=forecaster,
        observer_name="pvlive_day_after",
    )
    deltas_response = await client.get_week_average_deltas(deltas_request)
    deltas = deltas_response.deltas

    adjusted_values: list[dp.CreateForecastRequestForecastValue] = []
    for fv in forecast_values:
        delta_candidates = [d.delta_fraction for d in deltas if d.horizon_mins == fv.horizon_mins]
        delta_fraction = delta_candidates[0] if len(delta_candidates) > 0 else 0

        delta_fraction = _limit_adjuster(
            delta_fraction=delta_fraction,
            value_fraction=fv.p50_fraction,
        )

        new_p50 = max(0.0, min(1.0, fv.p50_fraction - delta_fraction))
        new_other_stats: dict[str, float] = {}
        for key, val in fv.other_statistics_fractions.items():
            new_val = max(0.0, min(1.0, val - delta_fraction))
            new_other_stats[key] = new_val

        adjusted_values.append(
            dp.CreateForecastRequestForecastValue(
                horizon_mins=fv.horizon_mins,
                p50_fraction=new_p50,
                metadata=fv.metadata,
                other_statistics_fractions=new_other_stats,
            ),
        )

    adjuster_forecaster = await _create_forecaster_if_not_exists(
        client=client,
        model_tag=model_tag + "_adjust",
    )

    return dp.CreateForecastRequest(
        forecaster=adjuster_forecaster,
        location_uuid=location_uuid,
        energy_source=dp.EnergySource.SOLAR,
        init_time_utc=init_time_utc.replace(tzinfo=UTC),
        values=adjusted_values,
    )


async def save_forecast_to_dataplatform(
    forecast_df: pd.DataFrame,
    location_uuid: UUID,
    model_tag: str,
    init_time_utc: datetime,
    client: DataPlatformClient,
    use_adjuster: bool = True,
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
            start_ts = add_or_convert_to_utc(row["start_utc"])
            init_ts = add_or_convert_to_utc(init_time_utc)

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
            probs = json.loads(row["probabilistic_values"])
            for key, val_kw in probs.items():
                frac = (val_kw * 1000) / capacity_watts
                # Clamp to [0, 1.0] to satisfy validation
                other_stats[key] = max(0.0, min(frac, 1.0))

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

        # Save adjusted forecast based on recent deltas
        if use_adjuster:
            adjusted_forecast_request = await _make_forecaster_adjuster(
                client=client,
                location_uuid=target_uuid_str,
                init_time_utc=init_time_utc,
                forecast_values=forecast_values,
                model_tag=model_tag,
                forecaster=forecaster,
            )
            await client.create_forecast(adjusted_forecast_request)


