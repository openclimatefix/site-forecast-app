from datetime import UTC, datetime, timedelta

import pytest
from dp_sdk.ocf import dp
from grpclib.client import Channel
from grpclib.const import Status
from grpclib.exceptions import GRPCError

from nl_blend.app import run_blend_app


@pytest.mark.asyncio
async def test_run_blend_app_e2e(dp_address, monkeypatch):
    """
    End-to-End integration test hitting a real Data Platform test container.

    Steps:
    1. Create the nl_national location.
    2. Seed fake source-model forecasts (backup + one candidate) so the
       blender has real data to work with.
    3. Run run_blend_app().
    4. Verify that nl_blend wrote forecast values to the Data Platform.
    """
    host, port = dp_address
    monkeypatch.setenv("DATA_PLATFORM_HOST", host)
    monkeypatch.setenv("DATA_PLATFORM_PORT", str(port))
    monkeypatch.setenv("COUNTRY", "nl")
    monkeypatch.setenv("LOCATION_TYPE", "nation")
    monkeypatch.setenv("NL_BLEND_LOCATION_TYPE", "nation")
    monkeypatch.setenv("NL_BLEND_COUNTRY", "nl")

    channel = Channel(host=host, port=port)
    try:
        client = dp.DataPlatformDataServiceStub(channel)

        # 1. Create the national location
        try:
            create_resp = await client.create_location(
                dp.CreateLocationRequest(
                    location_name="nl_national",
                    energy_source=dp.EnergySource.SOLAR,
                    geometry_wkt="POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))",
                    location_type=dp.LocationType.NATION,
                    effective_capacity_watts=1_000_000,
                    valid_from_utc=datetime(2020, 1, 1, tzinfo=UTC),
                ),
            )
            location_uuid = create_resp.location_uuid
        except GRPCError as e:
            if e.status != Status.ALREADY_EXISTS:
                raise
            # Location already exists — look it up.
            list_resp = await client.list_locations(dp.ListLocationsRequest())
            location_uuid = next(
                loc.location_uuid
                for loc in list_resp.locations
                if loc.location_name == "nl_national"
            )

        # 2. Seed source model forecasts so the blender has data to work with.
        # We seed the backup model + one candidate, matching the config.yaml registry.
        # Floor to 15-min boundary to match the blend app's own t0 calculation.
        t0 = datetime.now(tz=UTC)
        t0 = t0.replace(minute=(t0.minute // 15) * 15, second=0, microsecond=0)
        for model_name in ("nl_regional_2h_pv_ecmwf", "nl_regional_48h_pv_ecmwf"):
            await _seed_source_model_forecast(
                client=client,
                location_uuid=location_uuid,
                model_name=model_name,
                init_time=t0,
                n_steps=96,  # 24 h at 15-min resolution
            )
    finally:
        channel.close()

    # 3. Run the actual application — it will blend the seeded forecasts.
    await run_blend_app()

    # 4. Verify the blended forecast was written to the Data Platform.
    verify_channel = Channel(host=host, port=port)
    try:
        verify_client = dp.DataPlatformDataServiceStub(verify_channel)

        # Resolve the nl_blend forecaster.
        list_forecasters_resp = await verify_client.list_forecasters(
            dp.ListForecastersRequest(forecaster_names_filter=["nl_blend"]),
        )
        assert list_forecasters_resp.forecasters, (
            "nl_blend forecaster was not created in the Data Platform."
        )
        forecaster = list_forecasters_resp.forecasters[0]

        # Resolve the national location UUID.
        list_locations_resp = await verify_client.list_locations(dp.ListLocationsRequest())
        national_uuid = next(
            (
                loc.location_uuid
                for loc in list_locations_resp.locations
                if loc.location_name == "nl_national"
            ),
            None,
        )
        assert national_uuid is not None, (
            "nl_national location not found in Data Platform."
        )

        # Query the blended timeseries over a wide window to capture any t0.
        now = datetime.now(tz=UTC)
        forecast_resp = await verify_client.get_forecast_as_timeseries(
            dp.GetForecastAsTimeseriesRequest(
                location_uuid=national_uuid,
                energy_source=dp.EnergySource.SOLAR,
                time_window=dp.TimeWindow(
                    start_timestamp_utc=now - timedelta(hours=1),
                    end_timestamp_utc=now + timedelta(hours=48),
                ),
                forecaster=forecaster,
            ),
        )
        assert len(forecast_resp.values) > 0, (
            "nl_blend wrote no forecast values to the Data Platform."
        )
    finally:
        verify_channel.close()


async def _seed_source_model_forecast(
    client: dp.DataPlatformDataServiceStub,
    location_uuid: str,
    model_name: str,
    init_time: datetime,
    n_steps: int,
) -> None:
    """Creates a forecaster and seeds fake 15-min forecast values into the DP.

    Mirrors the create_test_forecast helper in uk-pv-forecast-blend.

    Args:
        client:        Active DP stub.
        location_uuid: Target location UUID.
        model_name:    Forecaster name to register / reuse.
        init_time:     Forecast initialisation time (UTC).
        n_steps:       Number of 15-min steps to generate.
    """
    # Register or reuse the forecaster.
    try:
        create_forecaster_resp = await client.create_forecaster(
            dp.CreateForecasterRequest(name=model_name, version="1.0.0"),
        )
        forecaster = create_forecaster_resp.forecaster
    except GRPCError as e:
        if e.status != Status.ALREADY_EXISTS:
            raise
        list_resp = await client.list_forecasters(
            dp.ListForecastersRequest(forecaster_names_filter=[model_name]),
        )
        forecaster = list_resp.forecasters[0]

    # Build fake forecast values at 15-min intervals.
    # p50_fraction is a unitless fraction of capacity (0 - 1), not MW.
    p50_fraction = 0.5  # 50 % of the location's capacity

    values = [
        dp.CreateForecastRequestForecastValue(
            horizon_mins=15 * (i + 1),
            p50_fraction=p50_fraction,
        )
        for i in range(n_steps)
    ]

    await client.create_forecast(
        dp.CreateForecastRequest(
            forecaster=forecaster,
            location_uuid=location_uuid,
            energy_source=dp.EnergySource.SOLAR,
            init_time_utc=init_time,
            values=values,
        ),
    )
