"""
Fixtures for testing
"""

import datetime as dt
import logging
import os
import random
from uuid import uuid4

# Workaround for betterproto forward reference issue with dp_sdk
# Import dp module early and ensure all enum types are in its namespace
# This must happen before any protobuf classes are instantiated
try:
    import dp_sdk.ocf.dp as _dp
    # Ensure types are accessible for get_type_hints()
    if not hasattr(_dp, 'EnergySource'):
        from dp_sdk.ocf.dp import EnergySource, LocationType
        _dp.EnergySource = EnergySource
        _dp.LocationType = LocationType
except ImportError:
    pass  # dp_sdk not installed in this environment

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from pvsite_datamodel import DatabaseConnection
from pvsite_datamodel.read.model import get_or_create_model
from pvsite_datamodel.sqlmodels import (
    Base,
    ForecastSQL,
    ForecastValueSQL,
    GenerationSQL,
    LocationGroupSQL,
    LocationSQL,
)
from sqlalchemy import create_engine
from testcontainers.postgres import PostgresContainer

log = logging.getLogger(__name__)

random.seed(42)


@pytest.fixture(scope="session")
def engine():
    """Database engine fixture."""

    with PostgresContainer("postgres:14.5") as postgres:
        url = postgres.get_connection_url()
        os.environ["DB_URL"] = url
        engine = create_engine(url)

        yield engine

        engine.dispose()


@pytest.fixture()
def db_conn(engine):
    """Create db connections and create/drop tables at the start/end of each test"""

    connection = DatabaseConnection(engine.url, echo=False)
    engine = connection.engine

    Base.metadata.create_all(engine)

    yield connection

    Base.metadata.drop_all(engine)


@pytest.fixture()
def db_session(db_conn, engine):  # noqa: ARG001
    """Return a sqlalchemy session, which tears down everything properly post-test."""

    with db_conn.get_session() as session:
        # begin the nested transaction
        session.begin()
        # use the connection with the already started transaction
        yield session

        # roll back the broader transaction
        session.rollback()


@pytest.fixture()
def sites(db_session):
    """Seed some initial data into DB."""

    sites = []
    # PV site
    site = LocationSQL(
        client_location_id=1,
        client_location_name="test_site_nl",
        latitude=51,
        longitude=5,
        capacity_kw=20000,
        ml_id=0,
        asset_type="pv",
        country="nl",
    )
    db_session.add(site)
    sites.append(site)

    # Create 12 dummy regional sites for Netherlands
    capacity_step = 20000 / sum(list(range(1, 13)))

    for i in range(1, 13):
        site = LocationSQL(
            client_location_id=1,
            client_location_name="test_site_nl_regional",
            latitude=51,
            longitude=5.1,
            capacity_kw=i * capacity_step,
            ml_id=i,
            asset_type="pv",
            country="nl",
        )
        db_session.add(site)
        sites.append(site)

    # Although this site is an india site,
    # we want it to be in the test data so we adjust the lat and lon
    site = LocationSQL(
        client_location_id=1,
        client_location_name="test_site_ad",
        latitude=52,
        longitude=5,
        capacity_kw=20000,
        ml_id=1,
        asset_type="pv",
        country="india",
    )
    db_session.add(site)
    sites.append(site)

    # create site group for nl
    site_group = LocationGroupSQL(
        location_group_name="Test Site Group NL",
        # TODO this uuid is currently copied over from the model_config,
        # It could be worth in the future doing this dynamically
        location_group_uuid="4dce2381-9e8d-467e-bbdd-4e342c9e1d89",
    )
    # create regional site group for NL
    site_group_regional = LocationGroupSQL(
        location_group_name="Test Site Group NL Regional",
        # TODO this uuid is currently copied over from the model_config,
        # It could be worth in the future doing this dynamically
        location_group_uuid="418259c9-0a31-4df7-a657-e53df9623897",
    )
    site_group.locations = [sites[0]]
    site_group_regional.locations = sites[:13]
    db_session.add(site_group)
    db_session.add(site_group_regional)

    db_session.commit()

    return sites


@pytest.fixture()
def generation_db_values(db_session, sites, init_timestamp):
    """Create some fake generations"""

    n = 450  # 22.5 hours of readings
    start_times = [init_timestamp - dt.timedelta(minutes=x * 3) for x in range(n)]

    # remove some of the most recent readings (to simulate missing timestamps)
    del start_times[20]
    del start_times[8]
    del start_times[3]

    # Random power values in the range 0-10000kw
    power_values = [random.random() * 10000 for _ in range(len(start_times))]  # noqa: S311

    all_generations = []
    for site in sites:
        for i in range(0, len(start_times)):
            generation = GenerationSQL(
                location_uuid=site.location_uuid,
                generation_power_kw=power_values[i],
                start_utc=start_times[i],
                end_utc=start_times[i] + dt.timedelta(minutes=3),
            )
            all_generations.append(generation)

    db_session.add_all(all_generations)
    db_session.commit()

    return all_generations


@pytest.fixture()
def forecast_values():
    """Dummy forecast values"""

    n = 10  # number of forecast values
    step = 15  # in minutes
    init_utc = dt.datetime.now(dt.UTC)
    start_utc = [init_utc + dt.timedelta(minutes=i * step) for i in range(n)]
    end_utc = [d + dt.timedelta(minutes=step) for d in start_utc]
    forecast_power_kw = [i * 10 for i in range(n)]
    forecast_values = {
        "start_utc": start_utc,
        "end_utc": end_utc,
        "forecast_power_kw": forecast_power_kw,
    }

    return forecast_values


def generate_probabilistic_values():
    """Generate probabilistic values for forecast"""
    return {
        "p10": round(random.uniform(0, 5000), 2),  # noqa: S311
        "p50": round(random.uniform(5000, 10000), 2),  # noqa: S311
        "p90": round(random.uniform(10000, 15000), 2),  # noqa: S311
    }


@pytest.fixture()
def forecasts(db_session, sites):
    """Make fake forecasts"""
    init_timestamp = pd.Timestamp(dt.datetime.now(tz=None)).floor(  # noqa: DTZ005
        dt.timedelta(minutes=15),
    )

    n = 24 * 4  # 24 hours of readings of 15
    start_times = [init_timestamp - dt.timedelta(minutes=x * 15) for x in range(n)]
    start_times = start_times[::-1]

    for site in sites:
        forecast_uuid = uuid4()
        model = get_or_create_model(db_session, "test", "0.0.0")
        forecast = ForecastSQL(
            location_uuid=site.location_uuid,
            timestamp_utc=start_times[-1],
            forecast_version="0.0.0",
            created_utc=start_times[-1],
            forecast_uuid=forecast_uuid,
        )
        db_session.add(forecast)

        forecast_values = []
        for i in range(0, len(start_times)):
            forecast_value = ForecastValueSQL(
                horizon_minutes=i * 15,
                forecast_power_kw=random.random() * 10000,  # noqa: S311
                start_utc=start_times[i],
                end_utc=start_times[i] + dt.timedelta(minutes=15),
                ml_model_uuid=model.model_uuid,
                forecast_uuid=forecast_uuid,
                created_utc=start_times[-1],
                probabilistic_values=generate_probabilistic_values(),
            )
            forecast_values.append(forecast_value)

        db_session.add_all(forecast_values)


@pytest.fixture(scope="session")
def init_timestamp():
    """Returns a datetime floored to the last 15 mins"""

    return pd.Timestamp(dt.datetime.now(tz=None)).floor(dt.timedelta(minutes=15))  # noqa: DTZ005


@pytest.fixture(scope="session")
def time_before_present():
    """Returns a fixed time in the past with specified offset"""

    now = pd.Timestamp.now(tz=None)

    def _time_before_present(dt: dt.timedelta):
        return now - dt

    return _time_before_present


@pytest.fixture(scope="session")
def nwp_data(tmp_path_factory, time_before_present):
    """Dummy NWP data"""

    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(f"{os.path.dirname(os.path.abspath(__file__))}/test_data/nwp-no-data-3.zarr")

    # Last t0 to at least 6 hours ago and floor to 6-hour interval
    t0_datetime_utc = time_before_present(dt.timedelta(hours=0)).floor("6h")
    t0_datetime_utc = t0_datetime_utc - dt.timedelta(hours=6)
    ds.init_time.values[:] = pd.date_range(
        t0_datetime_utc - dt.timedelta(hours=6 * (len(ds.init_time) - 1)),
        t0_datetime_utc,
        freq=dt.timedelta(hours=1),
    )

    # force lat and lon to be in 0.1 steps
    ds.latitude.values[:] = [65.0 - i * 0.1 for i in range(len(ds.latitude))]
    ds.longitude.values[:] = [3.0 + i * 0.1 for i in range(len(ds.longitude))]

    # This is important to avoid saving errors
    for v in list(ds.coords.keys()):
        if ds.coords[v].dtype == object:
            ds[v].encoding.clear()

    for v in list(ds.variables.keys()):
        if ds[v].dtype == object:
            ds[v].encoding.clear()

    # Add data to dataset
    ds["ecmwf"] = xr.DataArray(
        np.zeros([len(ds[c]) for c in ds.xindexes]),
        coords=[ds[c] for c in ds.xindexes],
    )

    # change variables values
    ds.variable.values[0:14] = [
        "temperature_sl",
        "wind_u_component_10m",
        "wind_v_component_10m",
        "downward_shortwave_radiation_flux_gl",
        "direct_shortwave_radiation_flux_gl",
        "downward_longwave_radiation_flux_gl",
        "downward_ultraviolet_radiation_flux_gl",
        "cloud_cover_high",
        "cloud_cover_low",
        "cloud_cover_medium",
        "cloud_cover_total",
        "snow_depth_gl",
        "visibility_sl",
        "total_precipitation_rate_gl",
    ]

    # AS NWP data is loaded by the app from environment variable,
    # save out data and set paths as environmental variables
    temp_nwp_path_ecmwf = f"{tmp_path_factory.mktemp('data')}/nwp_ecmwf.zarr"
    os.environ["NWP_ECMWF_ZARR_PATH"] = temp_nwp_path_ecmwf
    ds.to_zarr(temp_nwp_path_ecmwf)


@pytest.fixture(scope="session")
def nwp_mo_global_data(tmp_path_factory, time_before_present):
    """Dummy NWP data"""

    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(f"{os.path.dirname(os.path.abspath(__file__))}/test_data/nwp-no-data-3.zarr")

    # Last t0 to at least 4 hours ago and floor to 3-hour interval
    t0_datetime_utc = time_before_present(dt.timedelta(hours=0)).floor("3h")
    t0_datetime_utc = t0_datetime_utc - dt.timedelta(hours=4)
    ds.init_time.values[:] = pd.date_range(
        t0_datetime_utc - dt.timedelta(hours=12 * (len(ds.init_time) - 1)),
        t0_datetime_utc,
        freq=dt.timedelta(hours=1),
    )

    # force lat and lon to be in 0.1 steps
    ds.latitude.values[:] = [65.0 - i * 0.1 for i in range(len(ds.latitude))]
    ds.longitude.values[:] = [3.0 + i * 0.1 for i in range(len(ds.longitude))]

    # This is important to avoid saving errors
    for v in list(ds.coords.keys()):
        if ds.coords[v].dtype == object:
            ds[v].encoding.clear()

    for v in list(ds.variables.keys()):
        if ds[v].dtype == object:
            ds[v].encoding.clear()

    # change variables values to for MO global
    ds.variable.values[0:14] = [
        "temperature_sl",
        "wind_u_component_10m",
        "wind_v_component_10m",
        "downward_shortwave_radiation_flux_gl",
        "cloud_cover_high",
        "cloud_cover_low",
        "cloud_cover_medium",
        "cloud_cover_total",
        "relative_humidity_sl",
        "snow_depth_gl",
        "visibility_sl",
        "downward_longwave_radiation_flux_gl",
        "direct_shortwave_radiation_flux_gl",
        "total_precipitation_rate_gl",
    ]

    # interpolate 3 hourly step to 1 hour steps
    steps = pd.TimedeltaIndex(np.arange(49) * 3600 * 1e9, freq="infer")
    ds = ds.interp(step=steps, method="linear")

    ds["mo_global"] = xr.DataArray(
        np.zeros([len(ds[c]) for c in ds.xindexes]),
        coords=[ds[c] for c in ds.xindexes],
    )

    # AS NWP data is loaded by the app from environment variable,
    # save out data and set paths as environmental variables
    temp_nwp_path_gfs = f"{tmp_path_factory.mktemp('data')}/nwp_mo_global.zarr"

    os.environ["NWP_MO_GLOBAL_ZARR_PATH"] = temp_nwp_path_gfs
    ds.to_zarr(temp_nwp_path_gfs)


@pytest.fixture(scope="session")
def satellite_data(tmp_path_factory, init_timestamp):
    """Dummy Satellite data"""
    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/non_hrv_shell_3.zarr",
    )
    # remove time dim and geostationary dims and expand them
    ds = ds.drop_vars(["time", "x_geostationary", "y_geostationary"])
    n_hours = 3

    # Add times so they lead up to present
    t0_datetime_utc = init_timestamp - dt.timedelta(minutes=0)
    times = pd.date_range(
        t0_datetime_utc - dt.timedelta(hours=n_hours),
        t0_datetime_utc,
        freq=dt.timedelta(minutes=5),
    )
    ds = ds.expand_dims(time=times)

    # set geostationary cords for India
    # TODO update for NL
    ds = ds.expand_dims(
        x_geostationary=np.arange(5000000.0, -5000000.0, -5000),
        y_geostationary=np.arange(-5000000.0, 5000000.0, 5000),
    )

    # Add data to dataset
    ds["data"] = xr.DataArray(
        np.zeros([len(ds[c]) for c in ds.xindexes]),
        coords=[ds[c] for c in ds.xindexes],
    )

    # Add stored attributes to DataArray
    ds.data.attrs = ds.attrs["_data_attrs"]
    del ds.attrs["_data_attrs"]

    # In production sat zarr is zipped
    temp_sat_path = f"{tmp_path_factory.mktemp('data')}/temp_sat.zarr.zip"

    # save out data and set paths as environmental variables
    os.environ["SATELLITE_ZARR_PATH"] = temp_sat_path
    with zarr.storage.ZipStore(temp_sat_path, mode="x") as store:
        ds.to_zarr(store)


@pytest.fixture(scope="function")
def small_satellite_data(tmp_path_factory, init_timestamp):
    """Small amount of non-zero dummy satellite data"""
    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/non_hrv_shell_3.zarr",
    )

    # remove time dim and geostationary dims and expand them
    ds = ds.drop_vars(["time", "x_geostationary", "y_geostationary"])
    n_hours = 3

    # Add times so they lead up to present
    t0_datetime_utc = init_timestamp - dt.timedelta(minutes=0)
    times = pd.date_range(
        t0_datetime_utc - dt.timedelta(hours=n_hours),
        t0_datetime_utc,
        freq=dt.timedelta(minutes=5),
    )
    ds = ds.expand_dims(time=times)

    # set geostationary cords for small area
    ds = ds.expand_dims(
        x_geostationary=np.arange(50000.0, -50000.0, -5000),
        y_geostationary=np.arange(-50000.0, 50000.0, 5000),
    )

    # Add data to dataset with random values between 1 and 100
    ds["data"] = xr.DataArray(
        np.random.uniform(1, 100, size=[len(ds[c]) for c in ds.xindexes]),
        coords=[ds[c] for c in ds.xindexes],
    )

    # Add stored attributes to DataArray
    ds.data.attrs = ds.attrs["_data_attrs"]
    del ds.attrs["_data_attrs"]

    # In production sat zarr is zipped
    temp_sat_path = f"{tmp_path_factory.mktemp('data')}/temp_small_sat.zarr.zip"

    # save out data and set paths as environmental variables
    os.environ["SATELLITE_ZARR_PATH"] = temp_sat_path
    with zarr.storage.ZipStore(temp_sat_path, mode="x") as store:
        ds.to_zarr(store)


@pytest.fixture(scope="function")
def use_satellite():
    """Set use satellite env var"""
    os.environ["USE_SATELLITE"] = "true"
