"""Script to synchronize Legacy Site Locations with the Data Platform.

WHY THIS SCRIPT IS CRITICAL:
This script creates "bridge" location entities in the Data Platform for every site
in the legacy Postgres database. It attaches a critical metadata field `legacy_uuid`
to each location.

The application (`site_forecast_app`) relies on this `legacy_uuid` metadata to map
the internal site IDs it processes to the correct Data Platform Location UUIDs.
Without this script running, the app will fail to find matching locations in the
Data Platform and will not be able to save forecasts.

Run this script:
1. During initial setup.
2. Whenever new sites are added to the legacy database.
"""

import asyncio
import logging
import os

from betterproto.lib.google.protobuf import Struct, Value
from dotenv import load_dotenv
from dp_sdk.ocf import dp
from grpclib.client import Channel
from pvsite_datamodel import DatabaseConnection
from pvsite_datamodel.read import get_sites_by_country

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main() -> None:
    """Synchronize legacy sites from Postgres to Data Platform."""
    # Load environment variables
    #load_dotenv(".env.local")

    # 1. Get legacy sites from Postgres
    db_url = os.environ["DB_URL"]
    logger.info(f"Connecting to DB: {db_url}")
    db = DatabaseConnection(db_url, echo=False)

    with db.get_session() as session:
        country = os.getenv("COUNTRY", "nl")
        client_name = os.getenv("CLIENT_NAME", "nl")
        logger.info(
            f"Getting sites for client={client_name}, country={country}",
        )
        legacy_sites = get_sites_by_country(
            session, country=country, client_name=client_name,
        )

    if not legacy_sites:
        logger.warning("No legacy sites found.")
        return

    logger.info(f"Found {len(legacy_sites)} legacy sites.")

    # 2. Connect to Data Platform
    host = os.getenv("DP_HOST", "localhost")
    port = int(os.getenv("DP_PORT", "50051"))
    logger.info(f"Connecting to Data Platform at {host}:{port}")

    channel = Channel(host=host, port=port)
    client = dp.DataPlatformDataServiceStub(channel)

    try:
        # 3. Get existing locations from Data Platform
        logger.info("Fetching existing locations from Data Platform...")
        dp_response = await client.list_locations(dp.ListLocationsRequest())

        legacy_uuid_map = {}
        for loc in dp_response.locations:
            if loc.metadata and "legacy_uuid" in loc.metadata:
                legacy_uuid_map[loc.metadata["legacy_uuid"]] = loc.location_uuid

        logger.info(f"Found {len(legacy_uuid_map)} existing synced sites.")

        # 4. Sync sites
        for site in legacy_sites:
            legacy_uuid = str(site.location_uuid)

            if legacy_uuid in legacy_uuid_map:
                logger.info(
                    f"Site {legacy_uuid} already exists as "
                    f"{legacy_uuid_map[legacy_uuid]}. Skipping.",
                )
                continue

            logger.info(f"Creating site {legacy_uuid} in DP...")

            # WKT Point
            wkt = f"POINT ({site.longitude} {site.latitude})"

            # Metadata with legacy_uuid
            metadata = Struct(fields={"legacy_uuid": Value(string_value=legacy_uuid)})

            req = dp.CreateLocationRequest(
                location_name=f"synced_site_{legacy_uuid[:8]}",
                energy_source=dp.EnergySource.SOLAR,
                geometry_wkt=wkt,
                effective_capacity_watts=int(site.capacity_kw * 1000) if site.capacity_kw else 0,
                location_type=dp.LocationType.SITE,
                metadata=metadata,
            )

            try:
                resp = await client.create_location(req)
                logger.info(
                    f"Successfully created location {resp.location_uuid} "
                    f"for legacy {legacy_uuid}",
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Failed to create location for {legacy_uuid}: {type(e)} {e}")

    finally:
        channel.close()

if __name__ == "__main__":
    asyncio.run(main())
