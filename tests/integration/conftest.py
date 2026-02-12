import os

import pytest_asyncio
from dp_sdk.ocf import dp
from dp_sdk.ocf.dp import EnergySource
from grpclib.client import Channel


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
