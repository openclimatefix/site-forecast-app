"""
Tests for functions in app.py
"""

import datetime as dt
import multiprocessing as mp
import uuid

import pandas as pd
from freezegun import freeze_time
from pvsite_datamodel.sqlmodels import ForecastSQL, ForecastValueSQL, LocationGroupSQL, MLModelSQL

from site_forecast_app.app import (
    get_sites,
    run_model,
    save_forecast,
)
from site_forecast_app.data.generation import get_generation_data
from site_forecast_app.models.pvnet.model import PVNetModel
from site_forecast_app.models.pydantic_models import get_all_models

mp.set_start_method("spawn", force=True)
now = pd.Timestamp.now().floor("15min") + pd.Timedelta(minutes=1)


def test_get_sites(db_session, sites):
    """Test for correct site ids"""

    sites = get_sites(db_session)
    sites = sorted(sites, key=lambda s: s.client_location_id)

    assert len(sites) == 13
    for site in sites:
        assert isinstance(site.location_uuid, uuid.UUID)
        assert sites[0].asset_type.name == "pv"


def test_get_sites_with_model_config(db_session, sites):
    """Test for correct site ids"""

    # make site_group
    location_group = LocationGroupSQL(
        location_group_name="Test Site Group",
    )
    db_session.add(location_group)
    db_session.commit()
    location_group.locations = sites

    model_config = get_all_models().models[0]
    model_config.client = None
    model_config.site_group_uuid = location_group.location_group_uuid

    sites = get_sites(db_session, model_config=model_config)
    sites = sorted(sites, key=lambda s: s.client_location_id)

    assert len(sites) == 15
    for site in sites:
        assert isinstance(site.location_uuid, uuid.UUID)
        assert sites[0].asset_type.name == "pv"


@freeze_time(now)
def test_get_model(
    db_session,
    sites,
    nwp_data,
    generation_db_values,  # noqa: ARG001
    init_timestamp,
    satellite_data,
    monkeypatch,
):
    """Test for getting valid model"""
    monkeypatch.setenv("SATELLITE_ZARR_PATH", satellite_data)
    monkeypatch.setenv("NWP_ECMWF_ZARR_PATH", nwp_data)

    all_models = get_all_models()
    ml_model = all_models.models[0]
    gen_sites = [s for s in sites if s.client_location_name == "test_site_nl"]
    gen_data = get_generation_data(db_session, gen_sites, timestamp=init_timestamp)
    model = PVNetModel(
        timestamp=init_timestamp,
        generation_data=gen_data,
        hf_version=ml_model.version,
        hf_repo=ml_model.id,
        name="test",
        site_uuid=str(gen_sites[0].location_uuid),
    )

    assert hasattr(model, "version")
    assert isinstance(model.version, str)
    assert hasattr(model, "predict")


@freeze_time(now)
def test_run_model(
    db_session,
    sites,
    nwp_data,
    nwp_mo_global_data_nl,
    generation_db_values,  # noqa: ARG001
    init_timestamp,
    satellite_data,
    monkeypatch,
    ):
    """Test for running a PV model"""

    monkeypatch.setenv("NWP_ECMWF_ZARR_PATH", nwp_data)
    monkeypatch.setenv("NWP_MO_GLOBAL_ZARR_PATH", nwp_mo_global_data_nl)
    monkeypatch.setenv("SATELLITE_ZARR_PATH", satellite_data)

    all_models = get_all_models()
    ml_model = all_models.models[0]
    gen_sites = [s for s in sites if s.client_location_name == "test_site_nl"]
    gen_data = get_generation_data(db_session, sites=gen_sites, timestamp=init_timestamp)
    model = PVNetModel(
        timestamp=init_timestamp,
        generation_data=gen_data,
        hf_version=ml_model.version,
        hf_repo=ml_model.id,
        name="test",
        site_uuid=str(uuid.uuid4()),
    )
    forecast = run_model(model=model, timestamp=init_timestamp)

    assert isinstance(forecast, list)
    assert len(forecast) == 36*4  # value for every 15mins over 36 hours
    assert all(isinstance(value["start_utc"], dt.datetime) for value in forecast)
    assert all(isinstance(value["end_utc"], dt.datetime) for value in forecast)
    assert all(isinstance(value["forecast_power_kw"], int) for value in forecast)


def test_save_forecast(db_session, sites, forecast_values):
    """Test for saving forecast"""

    site = sites[0]

    forecast = {
        "meta": {
            "location_uuid": site.location_uuid,
            "version": "0.0.0",
            "timestamp": dt.datetime.now(tz=dt.UTC),
        },
        "values": forecast_values,
    }

    save_forecast(
        db_session,
        forecast,
        write_to_db=True,
        ml_model_name="test",
        ml_model_version="0.0.0",
    )

    assert db_session.query(ForecastSQL).count() == 2
    assert db_session.query(ForecastValueSQL).count() == 10 * 2
    assert db_session.query(MLModelSQL).count() == 2

