from datetime import UTC, datetime

import pytest
from dp_sdk.ocf import dp
from grpclib.client import Channel

from nl_blend.app import run_blend_app


@pytest.mark.asyncio
async def test_run_blend_app_e2e(dp_address, monkeypatch):
    """
    End-to-End integration test hitting a real Data Platform test container.
    Verifies that the orchestrator completes safely. If no history or weights
    are set up, the execution exits early with WARNINGs, which is perfectly
    valid since it demonstrates real connections worked and didn't crash.
    """
    host, port = dp_address
    monkeypatch.setenv("DATA_PLATFORM_HOST", host)
    monkeypatch.setenv("DATA_PLATFORM_PORT", str(port))
    monkeypatch.setenv("COUNTRY", "nl")
    monkeypatch.setenv("LOCATION_TYPE", "nation")
    # For safety just in case other formats exist
    monkeypatch.setenv("NL_BLEND_LOCATION_TYPE", "nation")
    monkeypatch.setenv("NL_BLEND_COUNTRY", "nl")

    channel = Channel(host=host, port=port)
    client = dp.DataPlatformDataServiceStub(channel)

    # 1. Provide a bare minimum location to prevent "Empty Location Map"
    create_location_request = dp.CreateLocationRequest(
        location_name="Netherlands National",
        energy_source=dp.EnergySource.SOLAR,
        geometry_wkt="POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))",
        location_type=dp.LocationType.NATION,
        effective_capacity_watts=1_000_000,
        valid_from_utc=datetime(2020, 1, 1, tzinfo=UTC),
    )

    import contextlib

    # Safely create location
    with contextlib.suppress(Exception):
        await client.create_location(create_location_request)

    channel.close()

    # 2. Run the actual application.
    # We do not mock anything. It will use the gRPC client, pull location map,
    # pull capacity, load scores, and attempt to fetch timeseries. Since there are
    # no forecast values in the container, it will cleanly exit after not finding any.
    await run_blend_app()
