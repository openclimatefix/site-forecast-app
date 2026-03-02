import time
from importlib.metadata import version

import pytest_asyncio
from dp_sdk.ocf import dp
from grpclib.client import Channel
from testcontainers.core.container import DockerContainer
from testcontainers.postgres import PostgresContainer


@pytest_asyncio.fixture(scope="module")
async def client():
    """
    Fixture to spin up a PostgreSQL container and Data Platform container for each test module.
    This fixture uses `testcontainers` to start fresh containers and provides
    the data platform client dynamically for use in integration tests.
    """

    with PostgresContainer(
        "ghcr.io/openclimatefix/data-platform-pgdb:logging",
        username="postgres",
        password="postgres",  # noqa: S106
        dbname="postgres",
        env={"POSTGRES_HOST": "db"},
    ) as postgres:
        database_url = postgres.get_connection_url()
        database_url = database_url.replace("postgresql+psycopg2", "postgres")
        database_url = database_url.replace("localhost", "host.docker.internal")
        try:
             ver = version("dp_sdk")
        except Exception:
             ver = "latest"

        # If ver matches the wheel version 0.16.0, we can use that tag for the docker image
        # The solar-consumer example uses 0.16.0 likely.

        # If the version from uv source is used, it might be 0.16.0.

        with DockerContainer(
            image=f"ghcr.io/openclimatefix/data-platform:{ver}",
            env={"DATABASE_URL": database_url},
            ports=[50051],
        ) as data_platform_server:
            time.sleep(1)  # Give some time for the server to start

            port = data_platform_server.get_exposed_port(50051)
            host = data_platform_server.get_container_host_ip()
            channel = Channel(host=host, port=port)
            client = dp.DataPlatformDataServiceStub(channel)
            yield client
            channel.close()
