"""
Tests for functions in app.py
"""

import datetime as dt
import json
import multiprocessing as mp
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from freezegun import freeze_time
from pvsite_datamodel.sqlmodels import ForecastSQL, ForecastValueSQL, MLModelSQL

from site_forecast_app.app import (
    app,
)
from tests.end_to_end._utils import run_click_script

mp.set_start_method("spawn", force=True)
now = pd.Timestamp.now().floor("15min") + pd.Timedelta(minutes=1)



@freeze_time(now)
@patch("site_forecast_app.curtailment.EntsoePandasClient")
@pytest.mark.parametrize("write_to_db", [True, False])
def test_app(
    mock_entsoe_pandas_client,
    write_to_db,
    db_session,
    sites,  # noqa: ARG001
    nwp_data,  # noqa: ARG001
    nwp_mo_global_data_nl,  # noqa: ARG001
    generation_db_values,   # noqa: ARG001
    satellite_data,  # noqa: ARG001
    mock_da_prices,
    monkeypatch,
):
    """Test for running app from command line"""
    monkeypatch.setenv("CLIENT_NAME", "nl")
    monkeypatch.setenv("COUNTRY", "nl")

    mock_entsoe_pandas_client_instance = MagicMock()
    mock_entsoe_pandas_client.return_value = mock_entsoe_pandas_client_instance
    mock_entsoe_pandas_client_instance.query_day_ahead_prices.return_value = mock_da_prices

    init_n_forecasts = db_session.query(ForecastSQL).count()
    init_n_forecast_values = db_session.query(ForecastValueSQL).count()

    args = ["--date", dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d-%H-%M")]
    if write_to_db:
        args.append("--write-to-db")

    result = run_click_script(app, args)
    assert result.exit_code == 0

    fv_per_hour = 4 # 15 min resolution = 4 values per hour
    n_forecasts = 7 + 12*5
    n_models = 7
    # 1 site, 5 models do 36 hours
    # 4 regional models also do 36 hours for 12 more sites
    # average number of forecast is:
    n_fv = ((36 * n_forecasts) / n_forecasts) * fv_per_hour

    if write_to_db:
        assert db_session.query(ForecastSQL).count() == init_n_forecasts + n_forecasts * 2
        assert db_session.query(MLModelSQL).count() == n_models * 2
        forecast_values = db_session.query(ForecastValueSQL).all()
        assert len(forecast_values) == init_n_forecast_values + (n_forecasts * 2 * n_fv)
        assert forecast_values[0].probabilistic_values is not None
        assert json.loads(forecast_values[0].probabilistic_values)["p10"] is not None

    else:
        assert db_session.query(ForecastSQL).count() == init_n_forecasts
        assert db_session.query(ForecastValueSQL).count() == init_n_forecast_values


@freeze_time(now)
def test_app_ad(
    db_session,
    sites, # noqa: ARG001
    nwp_data, # noqa: ARG001
    nwp_mo_global_data_india, # noqa: ARG001
    generation_db_values, # noqa: ARG001
    satellite_data, # noqa: ARG001
    monkeypatch,
):
    """Test for running app from command line"""

    init_n_forecasts = db_session.query(ForecastSQL).count()
    init_n_forecast_values = db_session.query(ForecastValueSQL).count()

    args = ["--date", dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d-%H-%M")]
    args.append("--write-to-db")

    monkeypatch.setenv("CLIENT_NAME", "ad")
    monkeypatch.setenv("COUNTRY", "india")

    result = run_click_script(app, args)
    assert result.exit_code == 0

    n = 4  # 1 site, 4 models
    assert db_session.query(ForecastSQL).count() == init_n_forecasts + n * 2
    assert db_session.query(MLModelSQL).count() == n * 2
    forecast_values = db_session.query(ForecastValueSQL).all()
    assert len(forecast_values) == init_n_forecast_values + (n * 2 * 16)


@freeze_time(now)
def test_app_no_pv_data(db_session, sites, nwp_data, satellite_data, monkeypatch):  # noqa: ARG001
    """Test for running app from command line"""

    init_n_forecasts = db_session.query(ForecastSQL).count()
    init_n_forecast_values = db_session.query(ForecastValueSQL).count()

    args = ["--date", dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d-%H-%M")]
    args.append("--write-to-db")

    monkeypatch.setenv("CLIENT_NAME", "ad")
    monkeypatch.setenv("COUNTRY", "india")

    result = run_click_script(app, args)
    assert result.exit_code == 0

    n = 4  # 1 site, 4 models

    assert db_session.query(ForecastSQL).count() == init_n_forecasts + 2 * n
    assert db_session.query(ForecastValueSQL).count() == init_n_forecast_values + (2 * n * 16)


@freeze_time(now)
def test_app_ruvnl(
    db_session, sites, nwp_data_india, nwp_data_gencast, generation_db_values,  # noqa: ARG001
):
    """Test for running app from command line"""

    init_n_forecasts = db_session.query(ForecastSQL).count()
    init_n_forecast_values = db_session.query(ForecastValueSQL).count()

    args = ["--date", dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d-%H-%M")]
    args.append("--write-to-db")

    os.environ["CLIENT_NAME"] = "ruvnl"
    os.environ["COUNTRY"] = "india"

    result = run_click_script(app, args)
    assert result.exit_code == 0

    n = 2  # 1 site, 2 models with GenCast data
    assert db_session.query(ForecastSQL).count() == init_n_forecasts + n * 2
    assert db_session.query(MLModelSQL).count() == n * 2
    forecast_values = db_session.query(ForecastValueSQL).all()
    assert len(forecast_values) == init_n_forecast_values + (n * 2 * 192)
