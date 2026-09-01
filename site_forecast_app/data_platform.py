"""Data Platform operations: location management, forecaster lifecycle, and forecast saving."""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import AsyncIterator  # noqa: TC003
from datetime import UTC, datetime

import pandas as pd
from dp_sdk.ocf import dp
from grpclib.client import Channel
from pvsite_datamodel.sqlmodels import LocationSQL  # noqa: TC002

# from site_forecast_app.save.utils import (
#     add_or_convert_to_utc,
#     ensure_timezone_aware,
#     limit_adjuster,
# )

log = logging.getLogger(__name__)

# -- Version --
# we need to keep this static so that the adjust and api works,
# even if we change version
# we will put the app version in the metadata
dp_forecaster_version = "1.4.0"

# Type alias for the Data Platform client stub
DataPlatformClient = dp.DataPlatformDataServiceStub


async def get_locations_filtered(sites_uuids: list[str]) -> dict[str, str]:
    async with get_dataplatform_client() as client:
        resp = await client.list_locations(dp.ListLocationsRequest(location_uuids_filter=sites_uuids))
        return resp.locations


async def fetch_dp_location_map(client: DataPlatformClient) -> dict[str, str]:
    """Fetch all locations (SITE, NATION, STATE, etc.) from the Data Platform.

    Returns a name → UUID map. Pre-fetching avoids separate list_locations calls
    for every forecast save.
    """
    resp = await client.list_locations(dp.ListLocationsRequest())
    return {loc.location_name: loc.location_uuid for loc in resp.locations}


async def build_dp_location_map() -> dict[str, str]:
    """Async wrapper: open a channel, fetch the location map, close."""
    async with get_dataplatform_client() as client:
        return await fetch_dp_location_map(client)



@contextlib.asynccontextmanager
async def get_dataplatform_client() -> AsyncIterator[DataPlatformClient]:
    """Async context manager that opens a gRPC channel and yields a ready-to-use client.

    Usage::

        async with get_dataplatform_client() as client:
            await save_forecast_to_dataplatform(..., client=client)

    The channel is always closed on exit, even if an exception is raised.
    Host and port are read from the ``DATA_PLATFORM_HOST``/
    ``DATA_PLATFORM_PORT`` environment variables
    (defaulting to ``localhost:50051``).
    """
    channel = Channel(
        host=os.getenv("DATA_PLATFORM_HOST","localhost"),
        port=int(os.getenv("DATA_PLATFORM_PORT", "50051")),
    )
    try:
        yield dp.DataPlatformDataServiceStub(channel)
    finally:
        channel.close()

def determine_energy_source(site: LocationSQL) -> dp.EnergySource:
    """Determine the Data Platform EnergySource based on site asset type."""
    asset_type = site.asset_type.name if hasattr(site.asset_type, "name") else str(site.asset_type)
    if asset_type.lower() == "wind":
        return dp.EnergySource.WIND
    return dp.EnergySource.SOLAR


def ensure_timezone_aware(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware and always in UTC."""
    if isinstance(dt, pd.Timestamp):
        return dt.tz_localize("UTC") if dt.tz is None else dt.tz_convert("UTC")
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
