"""Functions for saving forecasts."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from uuid import uuid4

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


async def fetch_dp_location_map(client: DataPlatformClient) -> dict[str, str]:
    """Fetch all SITE locations from the Data Platform once and return a name → UUID map.

    Calling this once before iterating over sites avoids a separate list_locations
    gRPC call for every forecast save.
    """
    resp = await client.list_locations(
        dp.ListLocationsRequest(location_type_filter=dp.LocationType.SITE),
    )
    return {loc.location_name: loc.location_uuid for loc in resp.locations}


def build_dp_location_map() -> dict[str, str]:
    """Synchronous wrapper: open a throwaway channel, fetch the location map, close."""

    async def _run() -> dict[str, str]:
        channel = Channel(
            host=os.getenv("DP_HOST", "localhost"),
            port=int(os.getenv("DP_PORT", "50051")),
        )
        client = dp.DataPlatformDataServiceStub(channel)
        try:
            return await fetch_dp_location_map(client)
        finally:
            channel.close()

    return asyncio.run(_run())


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
    location_map: dict[str, str] | None = None,
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
            location_map: Optional pre-fetched mapping of DP location name to UUID.
                When provided, avoids a list_locations gRPC call per site.

    Raises:
            IOError: An error if database save fails
    """
    log.info(f"Saving forecast for location_id={forecast['meta']['location_uuid']}...")

    forecast_meta = {
        "location_uuid": forecast["meta"]["location_uuid"],
        "timestamp_utc": forecast["meta"]["timestamp"],
        "forecast_version": forecast["meta"]["version"],
        "client_location_name": forecast["meta"].get("client_location_name"),
        "capacity_kw": forecast["meta"].get("capacity_kw"),
        "latitude": forecast["meta"].get("latitude"),
        "longitude": forecast["meta"].get("longitude"),
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
                    client_location_name=forecast_meta.get("client_location_name"),
                    model_tag=ml_model_name if ml_model_name else "default-model",
                    init_time_utc=forecast_meta["timestamp_utc"],
                    client=client,
                    use_adjuster=use_adjuster and ml_model_name is not None,
                    capacity_kw=forecast_meta.get("capacity_kw"),
                    latitude=forecast_meta.get("latitude"),
                    longitude=forecast_meta.get("longitude"),
                    location_map=location_map,
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
    forecaster_name = model_tag.replace("-", "_").lower()
    raw_version = version("site-forecast-app")
    # DP validates version with a restricted charset; normalize local package versions
    # (which may contain e.g. '+' build metadata) to a compatible value.
    app_version = "".join(
        ch if (ch.isalnum() or ch in "._-") else "."
        for ch in raw_version.lower()
    )

    list_forecasters_request = dp.ListForecastersRequest(
        forecaster_names_filter=[forecaster_name],
    )
    try:
        list_forecasters_response = await client.list_forecasters(list_forecasters_request)
        existing_forecasters = list_forecasters_response.forecasters
    except Exception as e:
        if "NOT_FOUND" in str(e) or "No forecasters found" in str(e):
            existing_forecasters = []
        else:
            raise

    if len(existing_forecasters) > 0:
        filtered_forecasters = [
            f
            for f in existing_forecasters
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
    try:
        deltas_response = await client.get_week_average_deltas(deltas_request)
        deltas = deltas_response.deltas
    except Exception as e:
        if "No observer" in str(e) or "NOT_FOUND" in str(e):
            log.warning("Observer 'pvlive_day_after' not found. Creating it...")
            try:
                await client.create_observer(dp.CreateObserverRequest(name="pvlive_day_after"))
                deltas_response = await client.get_week_average_deltas(deltas_request)
                deltas = deltas_response.deltas
            except Exception as create_error:
                log.error(f"Failed to create observer or retry deltas: {create_error}")
                deltas = []
        else:
            raise e

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
    client_location_name: str | None,
    model_tag: str,
    init_time_utc: datetime,
    client: DataPlatformClient,
    use_adjuster: bool = True,
    capacity_kw: float | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    location_map: dict[str, str] | None = None,
) -> None:
    """Save forecast to data platform."""
    # Early validation
    if forecast_df.empty:
        log.info("Empty forecast dataframe, skipping save")
        return

    if not client_location_name:
        raise ValueError("client_location_name is required to save to the Data Platform")

    # Just an additional check init_time_utc is timezone aware
    init_time_utc = _ensure_timezone_aware(init_time_utc)

    log.info("Writing to data platform")

    # Resolve DP location UUID and create forecaster concurrently
    target_uuid_task = _resolve_target_uuid(client, client_location_name, location_map)
    forecaster_task = _create_forecaster_if_not_exists(client=client, model_tag=model_tag)

    target_uuid_str, forecaster = await asyncio.gather(target_uuid_task, forecaster_task)

    # If location doesn't exist in DP yet, create it now
    if target_uuid_str is None:
        target_uuid_str = await _create_new_location(
            client, client_location_name, capacity_kw or 0.0, latitude, longitude, init_time_utc,
        )

    # Fetch location capacity
    capacity_watts = await _get_location_capacity(client=client, target_uuid_str=target_uuid_str)

    # Early exit if no capacity
    if capacity_watts == 0:
        log.warning(f"Location {target_uuid_str} has 0 capacity, skipping data platform save")
        return

    # Vectorized forecast value preparation
    forecast_values = _prepare_forecast_values_vectorized(
        forecast_df, init_time_utc, capacity_watts,
    )

    if not forecast_values:
        log.info("No valid forecast values to save")
        return

    # Create forecast requests
    tasks = []

    # Main forecast
    forecast_request = dp.CreateForecastRequest(
        forecaster=forecaster,
        location_uuid=target_uuid_str,
        energy_source=dp.EnergySource.SOLAR,
        init_time_utc=init_time_utc,
        values=forecast_values,
    )
    tasks.append(client.create_forecast(forecast_request))

    # Adjusted forecast if needed
    if use_adjuster:
        adjusted_task = _create_adjusted_forecast(
            client=client,
            location_uuid=target_uuid_str,
            init_time_utc=init_time_utc,
            forecast_values=forecast_values,
            model_tag=model_tag,
            forecaster=forecaster,
        )
        tasks.append(adjusted_task)

    # Execute forecasts concurrently
    await asyncio.gather(*tasks)


def _ensure_timezone_aware(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware and always in UTC."""
    if isinstance(dt, pd.Timestamp):
        return dt.tz_localize("UTC") if dt.tz is None else dt.tz_convert("UTC")
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


async def _resolve_target_uuid(
    client: DataPlatformClient,
    client_location_name: str,
    location_map: dict[str, str] | None = None,
) -> str | None:
    """Look up the DP location UUID by name.

    If a pre-fetched *location_map* (name → UUID) is supplied it is used directly,
    avoiding an extra list_locations gRPC call.  When None, the map is fetched
    on-demand as before.

    Returns the UUID string if found, or None if the location does not exist yet.
    Raises on unexpected gRPC errors.
    """
    if location_map is None:
        resp = await client.list_locations(dp.ListLocationsRequest())
        location_map = {loc.location_name: loc.location_uuid for loc in resp.locations}

    if client_location_name in location_map:
        target_uuid = location_map[client_location_name]
        log.info(f"Mapped client location '{client_location_name}' to DP UUID {target_uuid}")
        return target_uuid

    log.warning(f"DP location '{client_location_name}' not found — will create it.")
    return None


async def _get_location_capacity(
    client: DataPlatformClient,
    target_uuid_str: str,
) -> int:
    """Fetch effective capacity (watts) for an existing DP location."""
    location = await client.get_location(
        dp.GetLocationRequest(
            location_uuid=target_uuid_str,
            energy_source=dp.EnergySource.SOLAR,
            include_geometry=False,
        ),
    )
    return location.effective_capacity_watts


async def _create_new_location(
    client: DataPlatformClient,
    client_location_name: str,
    capacity_kw: float,
    latitude: float | None,
    longitude: float | None,
    init_time_utc: datetime,
) -> int:
    """Create a new location and return its capacity."""
    log.warning(f"Location {client_location_name} not found. Attempting to create it...")

    # Create geometry
    delta = 0.001
    lon, lat = longitude or 0.0, latitude or 0.0
    coords = [
        (lon - delta, lat - delta),
        (lon + delta, lat - delta),
        (lon + delta, lat + delta),
        (lon - delta, lat + delta),
        (lon - delta, lat - delta),
    ]
    wkt = f"POLYGON (({', '.join(f'{x} {y}' for x, y in coords)}))"

    capacity_watts = int(capacity_kw * 1000)

    try:
        create_req = dp.CreateLocationRequest(
            location_name=client_location_name,
            energy_source=dp.EnergySource.SOLAR,
            geometry_wkt=wkt,
            effective_capacity_watts=capacity_watts,
            location_type=dp.LocationType.SITE,
            valid_from_utc=init_time_utc - timedelta(days=7),
        )
        create_resp = await client.create_location(create_req)
        log.info(f"Created new location {create_resp.location_uuid} for '{client_location_name}'")
        return create_resp.location_uuid
    except Exception as create_error:
        log.error(f"Failed to create location: {create_error}")
        raise


def _prepare_forecast_values_vectorized(
    forecast_df: pd.DataFrame,
    init_time_utc: datetime,
    capacity_watts: int,
) -> list:
    """Prepare forecast values using vectorized operations where possible."""
    # Precompute timezone-aware init_time if needed
    init_ts = add_or_convert_to_utc(init_time_utc)

    forecast_values = []

    # Pre-parse probabilistic values if they exist
    prob_values_parsed = {}
    if "probabilistic_values" in forecast_df.columns:
        for idx, prob_str in forecast_df["probabilistic_values"].items():
            if pd.notna(prob_str):
                try:
                    prob_values_parsed[idx] = json.loads(prob_str)
                except json.JSONDecodeError:
                    prob_values_parsed[idx] = {}

    # Use itertuples for better performance than iterrows
    for row in forecast_df.itertuples():
        # Calculate horizon
        if hasattr(row, "horizon_minutes") and pd.notna(row.horizon_minutes):
            horizon_mins = int(row.horizon_minutes)
        else:
            start_ts = add_or_convert_to_utc(row.start_utc)
            horizon_mins = int((start_ts - init_ts).total_seconds() / 60)

        # Convert and clamp power fraction
        p50_fraction = max(0.0, min(1.0, (row.forecast_power_kw * 1000) / capacity_watts))

        # Process probabilistic values
        other_stats = {}
        if row.Index in prob_values_parsed:
            for key, val_kw in prob_values_parsed[row.Index].items():
                frac = max(0.0, min(1.0, (val_kw * 1000) / capacity_watts))
                other_stats[key] = frac

        forecast_values.append(
            dp.CreateForecastRequestForecastValue(
                horizon_mins=horizon_mins,
                p50_fraction=p50_fraction,
                metadata=Struct().from_pydict({}),
                other_statistics_fractions=other_stats,
            ),
        )

    return forecast_values


async def _create_adjusted_forecast(
    client: DataPlatformClient,
    location_uuid: str,
    init_time_utc: datetime,
    forecast_values: list,
    model_tag: str,
    forecaster: dp.Forecaster,
) -> None:
    """Create adjusted forecast."""
    adjusted_forecast_request = await _make_forecaster_adjuster(
        client=client,
        location_uuid=location_uuid,
        init_time_utc=init_time_utc,
        forecast_values=forecast_values,
        model_tag=model_tag,
        forecaster=forecaster,
    )
    await client.create_forecast(adjusted_forecast_request)


