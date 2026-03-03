import time
from importlib.metadata import version

import pytest
from dp_sdk.ocf import dp
from grpclib.client import Channel
from testcontainers.core.container import DockerContainer
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="module")
def dp_address():
    """
    Fixture to spin up a PostgreSQL container and Data Platform container for each test module.
    Yields (host, port) for the Data Platform server.
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

        with DockerContainer(
            image=f"ghcr.io/openclimatefix/data-platform:{ver}",
            env={"DATABASE_URL": database_url},
            ports=[50051],
        ) as data_platform_server:
            time.sleep(1)  # Give some time for the server to start

            port = data_platform_server.get_exposed_port(50051)
            host = data_platform_server.get_container_host_ip()
            yield host, port
