"""
Tests for functions in app.py
"""

import datetime as dt
import json
import multiprocessing as mp
import os
import uuid

import pytest
from pvsite_datamodel.sqlmodels import ForecastSQL, ForecastValueSQL, LocationGroupSQL, MLModelSQL

from site_forecast_app.app import (
    app,
    get_sites,
    run_model,
    save_forecast,
)
from site_forecast_app.data.generation import get_generation_data
from site_forecast_app.models.pvnet.model import PVNetModel
from site_forecast_app.models.pydantic_models import get_all_models

from ._utils import run_click_script

mp.set_start_method("spawn", force=True)


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

    assert len(sites) == 14
    for site in sites:
        assert isinstance(site.location_uuid, uuid.UUID)
        assert sites[0].asset_type.name == "pv"


def test_get_model(
    db_session,
    sites,
    nwp_data, # noqa: ARG001
    generation_db_values,  # noqa: ARG001
    init_timestamp,
    satellite_data,  # noqa: ARG001
):
    """Test for getting valid model"""

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
    )

    assert hasattr(model, "version")
    assert isinstance(model.version, str)
    assert hasattr(model, "predict")


def test_run_model(
    db_session,
    sites,
    nwp_data,  # noqa: ARG001
    generation_db_values,  # noqa: ARG001
    init_timestamp,
    satellite_data,  # noqa: ARG001
):
    """Test for running PV and wind models"""

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
    )
    forecast = run_model(model=model, site_uuid=str(uuid.uuid4()), timestamp=init_timestamp)

    assert isinstance(forecast, list)
    assert len(forecast) == 192  # value for every 15mins over 2 days
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


@pytest.mark.parametrize("write_to_db", [True, False])
def test_app(
    write_to_db, db_session, sites, nwp_data, generation_db_values, satellite_data,  # noqa: ARG001
):
    """Test for running app from command line"""

    init_n_forecasts = db_session.query(ForecastSQL).count()
    init_n_forecast_values = db_session.query(ForecastValueSQL).count()

    args = ["--date", dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d-%H-%M")]
    if write_to_db:
        args.append("--write-to-db")

    result = run_click_script(app, args)
    assert result.exit_code == 0

    n_forecasts = 4+12
    n_models = 4
    # 1 site, 4 models:
    #   1 model does 48 hours
    #   3 models do 36 hours
    # 1 regional model also does 36 hours for 12 more sites
    # average number of forecast is:
    n_fv = ((48+36*15)/n_forecasts)*4

    if write_to_db:
        assert db_session.query(ForecastSQL).count() == init_n_forecasts + n_forecasts * 2
        assert db_session.query(MLModelSQL).count() == n_models*2
        forecast_values = db_session.query(ForecastValueSQL).all()
        assert len(forecast_values) == init_n_forecast_values + (n_forecasts * 2 * n_fv)
        assert forecast_values[0].probabilistic_values is not None
        assert json.loads(forecast_values[0].probabilistic_values)["p10"] is not None

    else:
        assert db_session.query(ForecastSQL).count() == init_n_forecasts
        assert db_session.query(ForecastValueSQL).count() == init_n_forecast_values


def test_app_ad(
    db_session, sites, nwp_data, nwp_mo_global_data, generation_db_values, satellite_data,  # noqa: ARG001
):
    """Test for running app from command line"""

    init_n_forecasts = db_session.query(ForecastSQL).count()
    init_n_forecast_values = db_session.query(ForecastValueSQL).count()

    args = ["--date", dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d-%H-%M")]
    args.append("--write-to-db")

    os.environ["CLIENT_NAME"] = "ad"
    os.environ["COUNTRY"] = "india"

    result = run_click_script(app, args)
    assert result.exit_code == 0

    n = 4  # 1 site, 4 models
    assert db_session.query(ForecastSQL).count() == init_n_forecasts + n * 2
    assert db_session.query(MLModelSQL).count() == n * 2
    forecast_values = db_session.query(ForecastValueSQL).all()
    assert len(forecast_values) == init_n_forecast_values + (n * 2 * 16)


def test_app_no_pv_data(db_session, sites, nwp_data, satellite_data):  # noqa: ARG001
    """Test for running app from command line"""

    init_n_forecasts = db_session.query(ForecastSQL).count()
    init_n_forecast_values = db_session.query(ForecastValueSQL).count()

    args = ["--date", dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d-%H-%M")]
    args.append("--write-to-db")

    result = run_click_script(app, args)
    assert result.exit_code == 0

    n = 4  # 1 site, 4 models

    assert db_session.query(ForecastSQL).count() == init_n_forecasts + 2 * n
    assert db_session.query(ForecastValueSQL).count() == init_n_forecast_values + (2 * n * 16)
