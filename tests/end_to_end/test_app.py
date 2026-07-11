"""
Tests for functions in app.py
"""

import datetime as dt
import json
from unittest.mock import MagicMock, patch

import pandas as pd
from freezegun import freeze_time
from pvsite_datamodel.sqlmodels import ForecastSQL, ForecastValueSQL, MLModelSQL

from site_forecast_app.app import (
    app,
)
from tests.end_to_end._utils import run_click_script

now = pd.Timestamp.now().floor("15min") + pd.Timedelta(minutes=1)


def _base_args(write_to_db: bool = False) -> list[str]:
    """Build common CLI args for app tests."""
    args = ["--date", dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d-%H-%M")]
    if write_to_db:
        args.append("--write-to-db")
    return args


@freeze_time(now)
@patch("site_forecast_app.curtailment.EntsoePandasClient")
def test_app(
    mock_entsoe_pandas_client,
    db_session,
    sites,  # noqa: ARG001
    nwp_data,
    nwp_mo_global_data_nl,
    generation_db_values,  # noqa: ARG001
    satellite_data,
    mock_da_prices,
    monkeypatch,
):
    """Test for running app from command line"""
    monkeypatch.setenv("CLIENT_NAME", "nl")
    monkeypatch.setenv("COUNTRY", "nl")
    monkeypatch.setenv("NWP_ECMWF_ZARR_PATH", nwp_data)
    monkeypatch.setenv("NWP_MO_GLOBAL_ZARR_PATH", nwp_mo_global_data_nl)
    monkeypatch.setenv("SATELLITE_ZARR_PATH", satellite_data)

    mock_entsoe_pandas_client_instance = MagicMock()
    mock_entsoe_pandas_client.return_value = mock_entsoe_pandas_client_instance
    mock_entsoe_pandas_client_instance.query_day_ahead_prices.return_value = mock_da_prices

    init_n_forecasts = db_session.query(ForecastSQL).count()
    init_n_forecast_values = db_session.query(ForecastValueSQL).count()

    write_to_db = True
    args = _base_args(write_to_db)

    result = run_click_script(app, args)
    assert result.exit_code == 0

    fv_per_hour = 4  # 15 min resolution = 4 values per hour
    n_national_models = 2
    n_regional_models = 9
    n_uncurtailed_saves = 1  # nl_regional_pv_ecmwf_mo_sat saves uncurtailed forecasts too
    # each regional model writes 12 regional sites + 1 national summation = 13 forecasts
    n_forecasts = n_national_models + (n_regional_models + n_uncurtailed_saves) * 13
    n_models = n_national_models + n_regional_models + n_uncurtailed_saves
    # each forecast has 36 hours of values
    n_fv = 36 * fv_per_hour

    assert db_session.query(ForecastSQL).count() == init_n_forecasts + n_forecasts * 2
    assert db_session.query(MLModelSQL).count() == n_models * 2
    forecast_values = db_session.query(ForecastValueSQL).all()
    assert len(forecast_values) == init_n_forecast_values + (n_forecasts * 2 * n_fv)
    assert forecast_values[0].probabilistic_values is not None
    assert json.loads(forecast_values[0].probabilistic_values)["p10"] is not None


@freeze_time(now)
def test_app_ad(
    db_session,
    sites,  # noqa: ARG001
    nwp_data_india,
    nwp_mo_global_data_india,
    nwp_data_fgn,
    generation_db_values,  # noqa: ARG001
    satellite_data,
    monkeypatch,
):
    """Test for running app from command line"""

    monkeypatch.setenv("CLIENT_NAME", "ad")
    monkeypatch.setenv("COUNTRY", "india")
    monkeypatch.setenv("NWP_ECMWF_ZARR_PATH", nwp_data_india)
    monkeypatch.setenv("NWP_MO_GLOBAL_ZARR_PATH", nwp_mo_global_data_india)
    monkeypatch.setenv("SATELLITE_ZARR_PATH", satellite_data)
    monkeypatch.setenv("NWP_GENCAST_GCS_BUCKET_PATH", nwp_data_fgn["bucket"])
    monkeypatch.setenv("NWP_GENCAST_ZARR_PATH", nwp_data_fgn["zarr"])

    init_n_forecasts = db_session.query(ForecastSQL).count()
    init_n_forecast_values = db_session.query(ForecastValueSQL).count()

    args = _base_args(write_to_db=True)

    result = run_click_script(app, args)
    assert result.exit_code == 0

    n = 6  # 6 models across 2 sites
    assert db_session.query(ForecastSQL).count() == init_n_forecasts + n * 2
    assert db_session.query(MLModelSQL).count() == n * 2
    forecast_values = db_session.query(ForecastValueSQL).all()
    # 4 forecasts (n-2) with 16 forecast steps and 2 with 192 forecast steps
    assert len(forecast_values) == init_n_forecast_values + ((n - 2) * 2 * 16) + (2 * 2 * 192)


@freeze_time(now)
def test_app_no_pv_data(
    db_session, sites, nwp_data_india, satellite_data, nwp_data_fgn, monkeypatch, # noqa: ARG001
):
    """Test for running app from command line"""

    monkeypatch.setenv("CLIENT_NAME", "ad")
    monkeypatch.setenv("COUNTRY", "india")
    monkeypatch.setenv("NWP_ECMWF_ZARR_PATH", nwp_data_india)
    monkeypatch.setenv("SATELLITE_ZARR_PATH", satellite_data)
    monkeypatch.setenv("NWP_GENCAST_GCS_BUCKET_PATH", nwp_data_fgn["bucket"])
    monkeypatch.setenv("NWP_GENCAST_ZARR_PATH", nwp_data_fgn["zarr"])

    init_n_forecasts = db_session.query(ForecastSQL).count()
    init_n_forecast_values = db_session.query(ForecastValueSQL).count()

    args = _base_args(write_to_db=True)

    result = run_click_script(app, args)
    assert result.exit_code == 0

    n = 6  # 1 site, 4 models

    assert db_session.query(ForecastSQL).count() == init_n_forecasts + 2 * n
    # 4 forecasts (n-2) with 16 forecast steps and 2 with 192 forecast steps
    assert db_session.query(ForecastValueSQL).count() == init_n_forecast_values + (
        (n - 2) * 2 * 16
    ) + (2 * 2 * 192)


@freeze_time(now)
def test_app_ruvnl(
    db_session,
    sites, # noqa: ARG001
    nwp_data_india,
    nwp_data_gencast,
    generation_db_values, # noqa: ARG001
    monkeypatch,
):
    """Test for running app from command line"""

    monkeypatch.setenv("CLIENT_NAME", "ruvnl")
    monkeypatch.setenv("COUNTRY", "india")
    monkeypatch.setenv("NWP_ECMWF_ZARR_PATH", nwp_data_india)
    monkeypatch.setenv("NWP_GENCAST_GCS_BUCKET_PATH", nwp_data_gencast["bucket"])
    monkeypatch.setenv("NWP_GENCAST_ZARR_PATH", nwp_data_gencast["zarr"])

    init_n_forecasts = db_session.query(ForecastSQL).count()
    init_n_forecast_values = db_session.query(ForecastValueSQL).count()

    args = _base_args(write_to_db=True)

    result = run_click_script(app, args)
    assert result.exit_code == 0

    n = 2  # 1 site, 2 models with GenCast data
    assert db_session.query(ForecastSQL).count() == init_n_forecasts + n * 2
    assert db_session.query(MLModelSQL).count() == n * 2
    forecast_values = db_session.query(ForecastValueSQL).all()
    assert len(forecast_values) == init_n_forecast_values + (n * 2 * 192)


@freeze_time(now)
@patch("site_forecast_app.curtailment.EntsoePandasClient")
def test_app_critical_only(
    mock_entsoe_pandas_client,
    db_session,
    sites,  # noqa: ARG001
    nwp_data,
    nwp_mo_global_data_nl,
    generation_db_values,  # noqa: ARG001
    satellite_data,
    mock_da_prices,
    monkeypatch,
):
    """Test that RUN_CRITICAL_MODELS_ONLY=true skips non-critical models."""
    monkeypatch.setenv("CLIENT_NAME", "nl")
    monkeypatch.setenv("COUNTRY", "nl")
    monkeypatch.setenv("NWP_ECMWF_ZARR_PATH", nwp_data)
    monkeypatch.setenv("NWP_MO_GLOBAL_ZARR_PATH", nwp_mo_global_data_nl)
    monkeypatch.setenv("SATELLITE_ZARR_PATH", satellite_data)
    monkeypatch.setenv("RUN_CRITICAL_MODELS_ONLY", "true")

    mock_entsoe_pandas_client_instance = MagicMock()
    mock_entsoe_pandas_client.return_value = mock_entsoe_pandas_client_instance
    mock_entsoe_pandas_client_instance.query_day_ahead_prices.return_value = mock_da_prices

    init_n_forecasts = db_session.query(ForecastSQL).count()

    args = _base_args(write_to_db=True)

    result = run_click_script(app, args)
    assert result.exit_code == 0

    # With critical only, the 4 single-source models (is_critical=false) should be skipped
    # so fewer forecasts than test_app
    assert db_session.query(ForecastSQL).count() > init_n_forecasts
