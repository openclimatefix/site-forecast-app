import os

# Workaround for betterproto forward reference issue - must happen before dp import
import dp_sdk.ocf.dp as _dp
import pytest_asyncio
from dp_sdk.ocf import dp
from dp_sdk.ocf.dp import EnergySource, LocationType
from grpclib.client import Channel

# Apply the workaround
_dp.EnergySource = EnergySource
_dp.LocationType = LocationType


@pytest_asyncio.fixture
async def client_channel():
    """Create a gRPC channel for integration tests."""
    host = os.getenv("DP_HOST", "localhost")
    port = int(os.getenv("DP_PORT", "50051"))
    channel = Channel(host=host, port=port)
    yield channel
    channel.close()

@pytest_asyncio.fixture
async def client(client_channel):
    """Create a Data Platform client."""
    return dp.DataPlatformDataServiceStub(client_channel)
