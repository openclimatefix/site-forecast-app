import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from betterproto.lib.google.protobuf import Struct, Value
from dp_sdk.ocf import dp

from site_forecast_app.app import get_sites
from site_forecast_app.data_platform import (
    get_sites_from_data_platform,
    location_name_matches,
    ml_id_for_location,
)
from site_forecast_app.models.pydantic_models import Model


def _make_location(
    location_name: str,
    location_type: dp.LocationType,
    energy_source: dp.EnergySource = dp.EnergySource.SOLAR,
    effective_capacity_watts: int = 5_000_000,
    location_uuid: str | None = None,
    metadata: dict | None = None,
) -> dp.ListLocationsResponseLocationSummary:
    """Makes a Data Platform location.

    This uses the real message type rather than a mock, so the test fails if the
    Data Platform ever renames one of the fields we read.
    """
    return dp.ListLocationsResponseLocationSummary(
        location_uuid=location_uuid or str(uuid.uuid4()),
        location_name=location_name,
        location_type=location_type,
        energy_source=energy_source,
        effective_capacity_watts=effective_capacity_watts,
        latlng=dp.LatLng(latitude=52.0, longitude=5.0),
        metadata=_metadata(metadata or {}),
    )


def _metadata(values: dict) -> Struct:
    """Makes a Data Platform metadata struct."""
    return Struct(
        fields={
            key: Value(string_value=value) if isinstance(value, str) else Value(number_value=value)
            for key, value in values.items()
        },
    )


def _mock_client(locations: list[dp.ListLocationsResponseLocationSummary]) -> AsyncMock:
    """Fakes list_locations, honouring the location type and energy source filters.

    The Data Platform returns one entry per energy source, so a location carrying both
    solar and wind appears twice, once for each, with that source's capacity.
    """
    client = AsyncMock()

    def list_locations(request: dp.ListLocationsRequest) -> MagicMock:
        return MagicMock(
            locations=[
                location
                for location in locations
                if location.location_type == request.location_type_filter
                and location.energy_source == request.energy_source_filter
            ],
        )

    client.list_locations.side_effect = list_locations
    ctx = AsyncMock()
    ctx.__aenter__.return_value = client
    return ctx


def _model_config(**kwargs) -> Model:
    """Makes a model config, so these tests do not depend on all_models.yaml."""
    return Model(
        name="test_model",
        id="openclimatefix-models/test",
        version="abc123",
        satellite_archive_version="v0",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_get_sites_from_data_platform():
    """Test loading the NL states, and the national location, from the Data Platform."""
    locations = [
        _make_location("nl_groningen", dp.LocationType.STATE, metadata={"region_id": 2}),
        _make_location("nl_drenthe", dp.LocationType.STATE, metadata={"region_id": 1}),
        _make_location("de_bayern", dp.LocationType.STATE, metadata={"region_id": 1}),
        _make_location("nl_national", dp.LocationType.NATION),
    ]

    model_config = _model_config(
        client="nl",
        asset_type="pv",
        location_type="state",
        summation_location_type="nation",
    )

    with patch(
        "site_forecast_app.data_platform.get_dataplatform_client",
        return_value=_mock_client(locations),
    ):
        sites = await get_sites_from_data_platform(model_config)

    # the DE state is not loaded, and the national location keeps ml_id 0
    assert [site.client_location_name for site in sites] == [
        "nl_drenthe",
        "nl_groningen",
        "nl_national",
    ]
    assert [site.ml_id for site in sites] == [1, 2, 0]
    assert sites[0].capacity_kw == 5000
    assert sites[0].asset_type.name == "pv"
    assert sites[0].latitude == 52.0
    assert sites[0].longitude == 5.0


@pytest.mark.asyncio
async def test_get_sites_from_data_platform_shared_location():
    """Test that a wind model loads the wind half of a shared location.

    RUVNL solar and wind live on one location with one uuid, split by energy source,
    and each source carries its own capacity.
    """
    ruvnl_uuid = str(uuid.uuid4())
    locations = [
        _make_location(
            "ruvnl",
            dp.LocationType.STATE,
            energy_source=dp.EnergySource.SOLAR,
            effective_capacity_watts=5_000_000,
            location_uuid=ruvnl_uuid,
        ),
        _make_location(
            "ruvnl",
            dp.LocationType.STATE,
            energy_source=dp.EnergySource.WIND,
            effective_capacity_watts=3_000_000,
            location_uuid=ruvnl_uuid,
        ),
    ]

    model_config = _model_config(
        client="ruvnl",
        asset_type="wind",
        location_type="state",
        dp_location_name="ruvnl",
    )

    with patch(
        "site_forecast_app.data_platform.get_dataplatform_client",
        return_value=_mock_client(locations),
    ):
        sites = await get_sites_from_data_platform(model_config)

    # only the wind source is loaded, with the wind capacity and not the solar one
    assert len(sites) == 1
    assert sites[0].client_location_name == "ruvnl"
    assert sites[0].asset_type.name == "wind"
    assert sites[0].capacity_kw == 3000
    assert str(sites[0].location_uuid) == ruvnl_uuid


def test_location_name_matches_on_client_prefix():
    """Without a dp_location_name, locations are matched on the client prefix."""
    model_config = _model_config(client="nl")

    assert location_name_matches("nl_drenthe", model_config)
    assert not location_name_matches("de_bayern", model_config)


def test_location_name_matches_on_dp_location_name():
    """With a dp_location_name, only that exact location is matched."""
    model_config = _model_config(client="ruvnl", dp_location_name="ruvnl")

    assert location_name_matches("ruvnl", model_config)
    assert not location_name_matches("ruvnl_jaisalmer", model_config)


def test_get_sites_needs_a_model_config_to_load_from_the_data_platform(monkeypatch):
    """The model config is what says which locations to load, so it cannot be None."""
    monkeypatch.setenv("LOAD_SITES_FROM_DATA_PLATFORM", "true")

    with pytest.raises(ValueError, match="LOAD_SITES_FROM_DATA_PLATFORM"):
        get_sites(db_session=None, country="nl", client_name="nl")


def test_ml_id_is_zero_for_the_national_location():
    """The national location keeps ml_id 0, whatever its metadata says."""
    location = _make_location("nl_national", dp.LocationType.NATION, metadata={"region_id": 9})

    assert ml_id_for_location(location) == 0


def test_ml_id_comes_from_region_id_before_ml_id():
    """region_id wins when a location carries both."""
    location = _make_location(
        "nl_drenthe",
        dp.LocationType.STATE,
        metadata={"region_id": 3, "ml_id": 7},
    )

    assert ml_id_for_location(location) == 3


def test_ml_id_falls_back_to_ml_id_metadata():
    """A location without a region_id uses its ml_id metadata."""
    location = _make_location("ad_site_1", dp.LocationType.SITE, metadata={"ml_id": 7})

    assert ml_id_for_location(location) == 7


def test_ml_id_reads_a_metadata_value_written_as_a_string():
    """Metadata values can arrive as strings rather than numbers."""
    location = _make_location("ad_site_1", dp.LocationType.SITE, metadata={"ml_id": "7"})

    assert ml_id_for_location(location) == 7


def test_ml_id_defaults_to_one_without_metadata():
    """Locations that carry no id at all fall back to 1."""
    location = _make_location("ruvnl", dp.LocationType.STATE)

    assert ml_id_for_location(location) == 1


def test_ml_id_defaults_to_one_when_the_metadata_is_not_a_number():
    """A metadata value that is not a whole number is ignored rather than raising."""
    location = _make_location("ad_site_1", dp.LocationType.SITE, metadata={"ml_id": "not a number"})

    assert ml_id_for_location(location) == 1


def test_ml_id_metadata_of_zero_is_used():
    """A metadata id of 0 is a real value, not a missing one."""
    location = _make_location("ad_site_1", dp.LocationType.SITE, metadata={"ml_id": 0})

    assert ml_id_for_location(location) == 0
