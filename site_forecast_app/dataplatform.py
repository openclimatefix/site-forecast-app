"""Data Platform operations: location management, forecaster lifecycle, and forecast saving."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import traceback
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import pandas as pd
from dp_sdk.ocf import dp
from grpclib.client import Channel

from site_forecast_app.utils import add_or_convert_to_utc, ensure_timezone_aware, limit_adjuster

log = logging.getLogger(__name__)

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
        async with get_dataplatform_client() as client:
            return await fetch_dp_location_map(client)

    return asyncio.run(_run())


@contextlib.asynccontextmanager
async def get_dataplatform_client() -> AsyncIterator[DataPlatformClient]:
    """Async context manager that opens a gRPC channel and yields a ready-to-use client.

    Usage::

        async with get_dataplatform_client() as client:
            await save_forecast_to_dataplatform(..., client=client)

    The channel is always closed on exit, even if an exception is raised.
    Host and port are read from the ``DP_HOST`` / ``DP_PORT`` environment variables
    (defaulting to ``localhost:50051``).
    """
    channel = Channel(
        host=os.getenv("DP_HOST", "localhost"),
        port=int(os.getenv("DP_PORT", "50051")),
    )
    try:
        yield dp.DataPlatformDataServiceStub(channel)
    finally:
        channel.close()


def save_to_dataplatform(
    forecast_df: pd.DataFrame,
    forecast_meta: dict,
    ml_model_name: str | None,
    use_adjuster: bool,
    location_map: dict[str, str] | None = None,
) -> None:
    """Save Forecast to Dataplatform."""
    client_location_name = forecast_meta.get("client_location_name")
    model_tag = ml_model_name if ml_model_name else "default-model"
    init_time_utc = forecast_meta["timestamp_utc"]
    capacity_kw = forecast_meta.get("capacity_kw")

    log.info(
        "Starting DP save | "
        f"location={client_location_name!r}  model={model_tag!r}  "
        f"init_time={init_time_utc}  capacity_kw={capacity_kw}  "
        f"use_adjuster={use_adjuster}  df_rows={len(forecast_df)}  "
        f"location_map_size={len(location_map) if location_map else None}",
    )

    async def _run() -> None:
        async with get_dataplatform_client() as client:
            await save_forecast_to_dataplatform(
                forecast_df=forecast_df,
                client_location_name=client_location_name,
                model_tag=model_tag,
                init_time_utc=init_time_utc,
                client=client,
                use_adjuster=use_adjuster and ml_model_name is not None,
                capacity_kw=capacity_kw,
                latitude=forecast_meta.get("latitude"),
                longitude=forecast_meta.get("longitude"),
                location_map=location_map,
            )

    try:
        asyncio.run(_run())
        log.info(f"Save complete for location={client_location_name!r}")
    except Exception as e:
        log.error(f"Failed to save forecast to Data Platform: {e}")
        log.error(traceback.format_exc())


async def resolve_target_uuid(
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


async def get_location_capacity(
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


async def create_new_location(
    client: DataPlatformClient,
    client_location_name: str,
    capacity_kw: float,
    latitude: float | None,
    longitude: float | None,
    init_time_utc: datetime,
) -> int:
    """Create a new location in the Data Platform and return its UUID."""
    log.warning(f"Location {client_location_name} not found. Attempting to create it...")

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



async def create_forecaster_if_not_exists(
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


def prepare_forecast_values(
    forecast_df: pd.DataFrame,
    init_time_utc: datetime,
    capacity_watts: int,
) -> list[dp.CreateForecastRequestForecastValue]:
    """Convert a forecast DataFrame to a list of DP forecast value objects."""
    init_ts = add_or_convert_to_utc(init_time_utc)

    forecast_values: list[dp.CreateForecastRequestForecastValue] = []

    # Pre-parse probabilistic values if they exist
    prob_values_parsed: dict = {}
    if "probabilistic_values" in forecast_df.columns:
        for idx, prob_str in forecast_df["probabilistic_values"].items():
            if pd.notna(prob_str):
                try:
                    prob_values_parsed[idx] = json.loads(prob_str)
                except json.JSONDecodeError:
                    prob_values_parsed[idx] = {}

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
        other_stats: dict[str, float] = {}
        if row.Index in prob_values_parsed:
            for key, val_kw in prob_values_parsed[row.Index].items():
                frac = max(0.0, min(1.0, (val_kw * 1000) / capacity_watts))
                other_stats[key] = frac

        forecast_values.append(
            dp.CreateForecastRequestForecastValue(
                horizon_mins=horizon_mins,
                p50_fraction=p50_fraction,
                other_statistics_fractions=other_stats,
            ),
        )

    return forecast_values


async def make_adjuster_forecast_request(
    client: DataPlatformClient,
    location_uuid: str,
    init_time_utc: datetime,
    forecast_values: list[dp.CreateForecastRequestForecastValue],
    model_tag: str,
    forecaster: dp.Forecaster,
) -> dp.CreateForecastRequest:
    """Build an adjusted forecast request using week-average deltas."""
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

        delta_fraction = limit_adjuster(
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

    adjuster_forecaster = await create_forecaster_if_not_exists(
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


async def create_adjusted_forecast(
    client: DataPlatformClient,
    location_uuid: str,
    init_time_utc: datetime,
    forecast_values: list[dp.CreateForecastRequestForecastValue],
    model_tag: str,
    forecaster: dp.Forecaster,
) -> None:
    """Build and submit the adjusted forecast to the Data Platform."""
    adjusted_forecast_request = await make_adjuster_forecast_request(
        client=client,
        location_uuid=location_uuid,
        init_time_utc=init_time_utc,
        forecast_values=forecast_values,
        model_tag=model_tag,
        forecaster=forecaster,
    )
    await client.create_forecast(adjusted_forecast_request)


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
    """Save forecast to the Data Platform."""
    if forecast_df.empty:
        log.warning("forecast dataframe is empty")
        return

    if not client_location_name:
        log.error("client_location_name is None/empty — cannot save")
        raise ValueError("client_location_name is required to save to the Data Platform")

    init_time_utc = ensure_timezone_aware(init_time_utc)
    log.info(
        f"location={client_location_name!r}  "
        f"model={model_tag!r}  init_time={init_time_utc}  rows={len(forecast_df)}",
    )

    log.info(f"resolving UUID and forecaster for {client_location_name!r}")
    target_uuid_str, forecaster = await asyncio.gather(
        resolve_target_uuid(client, client_location_name, location_map),
        create_forecaster_if_not_exists(client=client, model_tag=model_tag),
    )
    log.info(
        f"uuid={target_uuid_str}  "
        f"forecaster={forecaster.forecaster_name!r} v{forecaster.forecaster_version}",
    )

    if target_uuid_str is None:
        log.warning(
            f"location {client_location_name!r} not in DP — creating it  "
            f"(capacity_kw={capacity_kw}, lat={latitude}, lon={longitude})",
        )
        target_uuid_str = await create_new_location(
            client, client_location_name, capacity_kw or 0.0, latitude, longitude, init_time_utc,
        )
        log.info(f"created location uuid={target_uuid_str}")
    else:
        log.info(f"location already exists uuid={target_uuid_str}")

    capacity_watts = await get_location_capacity(client=client, target_uuid_str=target_uuid_str)
    log.info(
        f"capacity_watts={capacity_watts:,}  ({capacity_watts / 1000:,.1f} kW)",
    )

    if capacity_watts == 0:
        log.error(
            f"location {target_uuid_str} has 0 W capacity — "
            "no forecast values can be expressed as fractions; skipping save",
        )
        return

    forecast_values = prepare_forecast_values(forecast_df, init_time_utc, capacity_watts)
    log.info(f"prepared {len(forecast_values)} forecast value(s)")

    if forecast_values:
        sample = forecast_values[0]
        log.info(
            f"sample[0]: horizon_mins={sample.horizon_mins}  "
            f"p50_fraction={sample.p50_fraction:.6f}  "
            f"other_stats={dict(sample.other_statistics_fractions)}",
        )
        p50s = [fv.p50_fraction for fv in forecast_values]
        log.info(
            f"p50 range: min={min(p50s):.6f}  max={max(p50s):.6f}  "
            f"mean={sum(p50s)/len(p50s):.6f}",
        )
    else:
        log.warning("no forecast values after preparation")
        return

    base_request = dp.CreateForecastRequest(
        forecaster=forecaster,
        location_uuid=target_uuid_str,
        energy_source=dp.EnergySource.SOLAR,
        init_time_utc=init_time_utc,
        values=forecast_values,
    )
    log.info(
        f"submitting base forecast  "
        f"forecaster={forecaster.forecaster_name!r}  "
        f"location={target_uuid_str}  values={len(forecast_values)}",
    )

    tasks = [client.create_forecast(base_request)]

    if use_adjuster:
        log.info("also queuing adjusted forecast")
        tasks.append(
            create_adjusted_forecast(
                client=client,
                location_uuid=target_uuid_str,
                init_time_utc=init_time_utc,
                forecast_values=forecast_values,
                model_tag=model_tag,
                forecaster=forecaster,
            ),
        )
    else:
        log.info("adjuster disabled, skipping adjusted forecast")

    await asyncio.gather(*tasks)
    log.info(
        f"{'base + adjusted' if use_adjuster else 'base'} forecast(s) submitted "
        f"for {client_location_name!r}",
    )
