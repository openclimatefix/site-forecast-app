"""Integration test for the DE blend against a real Data Platform container.
"""

from datetime import UTC, datetime, timedelta

import pytest
from betterproto.lib.google.protobuf import Struct, Value
from dp_sdk.ocf import dp
from grpclib.client import Channel

from site_forecast_app.blend.app import run_blend_app

DE_NATIONAL = "de_national"
DE_ZONES = ["de_50hertz", "de_amprion", "de_tennet", "de_transnetbw"]

DE_LOCATIONS = [
    # location_name, region, region_id, capacity_watts, latitude, longitude
    ("de_national", "de", 0, 58_212_000_000, 51.16, 10.45),
    ("de_50hertz", "50hertz", 1, 15_359_000_000, 52.5, 13.5),
    ("de_amprion", "amprion", 2, 13_151_000_000, 51.5, 7.5),
    ("de_tennet", "tennet", 3, 21_514_000_000, 51.0, 10.5),
    ("de_transnetbw", "transnetbw", 4, 7_864_000_000, 48.5, 9.0),
]

# de_ecmwf_only is the configured backup; the rest are the candidates the
# optimiser chooses between.
DE_MODELS = [
    "de_ecmwf_only",
    "de_ecmwf_pv",
    "de_ecmwf_pv_mo_sat",
    "de_mo_only",
    "de_sat_only",
]

N_STEPS = 192  # 48 h at 15-min resolution


@pytest.mark.asyncio
async def test_de_blend_writes_national_and_zone_forecasts(
    dp_address, monkeypatch, de_blend_config,
):
    """The blend writes de_blend everywhere, and de_blend_adjust nationally only."""
    host, port = dp_address
    monkeypatch.setenv("DATA_PLATFORM_HOST", host)
    monkeypatch.setenv("DATA_PLATFORM_PORT", str(port))
    monkeypatch.setenv("COUNTRY", "de")

    client = dp.DataPlatformDataServiceStub(Channel(host=host, port=port))

    # location_name -> location_uuid, which is how the blend reads the location map
    locations: dict[str, str] = {}
    for name, region, region_id, capacity_watts, latitude, longitude in DE_LOCATIONS:
        locations[name] = await _create_location(
            client=client,
            name=name,
            region=region,
            region_id=region_id,
            capacity_watts=capacity_watts,
            latitude=latitude,
            longitude=longitude,
        )

    # Only the national location gets {model}_adjust forecasts, as in production.
    forecasters = {name: await _create_forecaster(client, name) for name in DE_MODELS}
    adjust_forecasters = {
        name: await _create_forecaster(client, f"{name}_adjust") for name in DE_MODELS
    }

    t0 = datetime.now(tz=UTC)
    t0 = t0.replace(minute=(t0.minute // 15) * 15, second=0, microsecond=0)

    for location_uuid in locations.values():
        for forecaster in forecasters.values():
            await _seed_forecast(client, forecaster, location_uuid, t0)
    for forecaster in adjust_forecasters.values():
        await _seed_forecast(client, forecaster, locations[DE_NATIONAL], t0)

    await run_blend_app(config=de_blend_config)

    national_forecasters = await _forecaster_names_at(client, locations[DE_NATIONAL])
    assert de_blend_config.forecaster_name in national_forecasters
    assert de_blend_config.adjuster_forecaster_name in national_forecasters

    for zone in DE_ZONES:
        zone_forecasters = await _forecaster_names_at(client, locations[zone])
        assert de_blend_config.forecaster_name in zone_forecasters, (
            f"{zone} was not blended"
        )
        assert de_blend_config.adjuster_forecaster_name not in zone_forecasters, (
            f"{zone} should not have an adjusted blend"
        )

    # The national blend must contain real values, not just a registered forecaster.
    values = await _blended_values(client, locations[DE_NATIONAL], de_blend_config.forecaster_name)
    assert len(values) > 0, "blend wrote no forecast values to the Data Platform"


async def _create_location(
    client: dp.DataPlatformDataServiceStub,
    name: str,
    region: str,
    region_id: int,
    capacity_watts: int,
    latitude: float,
    longitude: float,
) -> str:
    """Create one DE location and return its UUID."""
    response = await client.create_location(
        dp.CreateLocationRequest(
            location_name=name,
            energy_source=dp.EnergySource.SOLAR,
            geometry_wkt=f"POINT({longitude} {latitude})",
            location_type=dp.LocationType.NATION if region_id == 0 else dp.LocationType.STATE,
            effective_capacity_watts=capacity_watts,
            valid_from_utc=datetime(2020, 1, 1, tzinfo=UTC),
            # the Data Platform derives latlng from the point, and rejects both
            metadata=Struct(
                fields={
                    "region": Value(string_value=region),
                    "country": Value(string_value="de"),
                    "region_id": Value(number_value=region_id),
                },
            ),
        ),
    )
    return response.location_uuid


async def _create_forecaster(client: dp.DataPlatformDataServiceStub, name: str):
    """Register one forecaster and return it."""
    response = await client.create_forecaster(
        dp.CreateForecasterRequest(name=name, version="1.0.0"),
    )
    return response.forecaster


async def _seed_forecast(
    client: dp.DataPlatformDataServiceStub,
    forecaster,
    location_uuid: str,
    init_time: datetime,
) -> None:
    """Seed 48 h of 15-min forecast values so the blend has something to read."""
    values = [
        dp.CreateForecastRequestForecastValue(
            horizon_mins=15 * (i + 1),
            p50_fraction=0.5,
        )
        for i in range(N_STEPS)
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


async def _forecaster_names_at(
    client: dp.DataPlatformDataServiceStub,
    location_uuid: str,
) -> set[str]:
    """Every forecaster with a latest forecast at this location."""
    response = await client.get_latest_forecasts(
        dp.GetLatestForecastsRequest(
            location_uuid=location_uuid,
            energy_source=dp.EnergySource.SOLAR,
        ),
    )
    return {forecast.forecaster.forecaster_name for forecast in response.forecasts}


async def _blended_values(
    client: dp.DataPlatformDataServiceStub,
    location_uuid: str,
    forecaster_name: str,
) -> list:
    """The forecast values written under a given forecaster at a location."""
    forecasters = await client.list_forecasters(
        dp.ListForecastersRequest(forecaster_names_filter=[forecaster_name]),
    )
    now = datetime.now(tz=UTC)
    response = await client.get_forecast_as_timeseries(
        dp.GetForecastAsTimeseriesRequest(
            location_uuid=location_uuid,
            energy_source=dp.EnergySource.SOLAR,
            forecaster=forecasters.forecasters[0],
            time_window=dp.TimeWindow(
                start_timestamp_utc=now - timedelta(hours=1),
                end_timestamp_utc=now + timedelta(hours=48),
            ),
        ),
    )
    return list(response.values)
