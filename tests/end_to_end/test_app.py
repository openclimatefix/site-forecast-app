"""
Tests for functions in app.py
"""

import contextlib
import datetime as dt
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
from dp_sdk.ocf import dp
from freezegun import freeze_time
from pvsite_datamodel.sqlmodels import ForecastSQL, ForecastValueSQL, MLModelSQL

from site_forecast_app.app import (
    app,
)
from tests.end_to_end._utils import run_click_script

now = pd.Timestamp.now().floor("15min") + pd.Timedelta(minutes=1)


def _base_args(write_to_db: bool = True) -> list[str]:
    """Build common CLI args for app tests."""
    args = ["--date", dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d-%H-%M")]
    args.append("--write-to-db" if write_to_db else "--no-write-to-db")
    return args

class FakeDataPlatform:
    """Fake Data Platform that serves locations and generation, and records forecasts."""

    def __init__(self, locations: list):
        self.locations = locations
        self.capacity_watts = {
            location.location_uuid: location.effective_capacity_watts for location in locations
        }
        self.forecasts: list[dp.CreateForecastRequest] = []

        self.client = AsyncMock()
        self.client.list_locations.side_effect = self._list_locations
        self.client.get_location.side_effect = self._get_location
        self.client.get_observations_as_timeseries.side_effect = self._get_observations
        self.client.create_forecaster.side_effect = self._create_forecaster
        self.client.create_forecast.side_effect = self.forecasts.append
        # no forecasters exist yet, so each one is created on first use
        self.client.list_forecasters.return_value = MagicMock(forecasters=[])
        # no history to adjust against, so the adjusted forecast equals the base one
        self.client.get_week_average_deltas.return_value = MagicMock(deltas=[])

    @contextlib.contextmanager
    def patched(self):
        """Serve this client wherever the app opens a Data Platform connection."""
        targets = [
            "site_forecast_app.data_platform.get_dataplatform_client",  # loading locations
            "site_forecast_app.data.generation.get_dataplatform_client",  # reading generation
            "site_forecast_app.save.data_platform.get_dataplatform_client",  # saving forecasts
        ]
        connection = AsyncMock()
        connection.__aenter__.return_value = self.client

        with contextlib.ExitStack() as stack:
            for target in targets:
                stack.enter_context(patch(target, return_value=connection))
            yield self

    def forecasts_by_location(self) -> dict[str, list[dp.CreateForecastRequest]]:
        """Group the submitted forecasts by location name."""
        names = {location.location_uuid: location.location_name for location in self.locations}
        grouped: dict[str, list[dp.CreateForecastRequest]] = {}
        for forecast in self.forecasts:
            grouped.setdefault(names[forecast.location_uuid], []).append(forecast)
        return grouped

    def _list_locations(self, request: dp.ListLocationsRequest) -> MagicMock:
        """Apply whichever filters are set, so state and nation calls get different lists."""
        locations = self.locations
        if request.location_type_filter is not None:
            locations = [
                location
                for location in locations
                if location.location_type == request.location_type_filter
            ]
        if request.energy_source_filter is not None:
            locations = [
                location
                for location in locations
                if location.energy_source == request.energy_source_filter
            ]

        return MagicMock(locations=locations)

    def _get_location(self, request: dp.GetLocationRequest) -> MagicMock:
        return MagicMock(effective_capacity_watts=self.capacity_watts[request.location_uuid])

    def _get_observations(
        self, request: dp.GetObservationsAsTimeseriesRequest,
    ) -> dp.GetObservationsAsTimeseriesResponse:
        """Return 15 minutely generation covering the requested window."""
        capacity_watts = self.capacity_watts[request.location_uuid]
        start = request.time_window.start_timestamp_utc
        end = request.time_window.end_timestamp_utc

        values = []
        timestamp = start
        while timestamp <= end:
            values.append(
                dp.GetObservationsAsTimeseriesResponseValue(
                    timestamp_utc=timestamp,
                    value_fraction=0.2,
                    effective_capacity_watts=capacity_watts,
                ),
            )
            timestamp += dt.timedelta(minutes=15)

        return dp.GetObservationsAsTimeseriesResponse(
            location_uuid=request.location_uuid,
            values=values,
        )

    def _create_forecaster(self, request: dp.CreateForecasterRequest) -> MagicMock:
        return MagicMock(
            forecaster=dp.Forecaster(
                forecaster_name=request.name,
                forecaster_version=request.version,
            ),
        )


@freeze_time(now)
@patch("site_forecast_app.curtailment.EntsoePandasClient")
def test_app_sat_v0(
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
    n_regional_models = 4
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
@patch("site_forecast_app.curtailment.EntsoePandasClient")
def test_app_sat_v1(
    mock_entsoe_pandas_client,
    db_session,
    sites,  # noqa: ARG001
    nwp_data,
    nwp_mo_global_data_nl,
    generation_db_values,  # noqa: ARG001
    satellite_data_icechunk,
    mock_da_prices,
    monkeypatch,
):
    """Test for running app from command line"""
    monkeypatch.setenv("CLIENT_NAME", "nl")
    monkeypatch.setenv("COUNTRY", "nl")
    monkeypatch.setenv("NWP_ECMWF_ZARR_PATH", nwp_data)
    monkeypatch.setenv("NWP_MO_GLOBAL_ZARR_PATH", nwp_mo_global_data_nl)
    monkeypatch.setenv("SATELLITE_ICECHUNK_PATH_5", satellite_data_icechunk)
    monkeypatch.setenv("SATELLITE_ARCHIVE_VERSION", "v1")

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
    n_national_models = 0
    n_regional_models = 7
    n_uncurtailed_saves = 0  # nl_regional_pv_ecmwf_mo_sat saves uncurtailed forecasts too
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



def test_app_de(
    db_session,  # noqa: ARG001
    de_dp_locations,
    satellite_data_icechunk,
    init_timestamp,
    monkeypatch,
):
    """Test for running the DE app, which uses the Data Platform throughout."""
    monkeypatch.setenv("CLIENT_NAME", "de")
    monkeypatch.setenv("COUNTRY", "de")
    monkeypatch.setenv("LOAD_SITES_FROM_DATA_PLATFORM", "true")
    monkeypatch.setenv("READ_FROM_DATA_PLATFORM", "true")
    monkeypatch.setenv("SAVE_TO_DATA_PLATFORM", "true")
    monkeypatch.setenv("SATELLITE_ARCHIVE_VERSION", "v1")
    # de_sat_only takes satellite as an input
    monkeypatch.setenv("SATELLITE_ICECHUNK_PATH_5", satellite_data_icechunk)
    # DE adjusts through the Data Platform, not the database
    monkeypatch.setenv("USE_ADJUSTER_DATABASE", "false")
    # DE has no blend config
    monkeypatch.setenv("RUN_BLEND_SERVICE", "false")
    # DE saves to the Data Platform, not the database
    monkeypatch.setenv("WRITE_TO_DB", "false")

    data_platform = FakeDataPlatform(de_dp_locations)

    args = ["--date", init_timestamp.strftime("%Y-%m-%d-%H-%M")]

    with data_platform.patched():
        result = run_click_script(app, args)

    assert result.exit_code == 0

    # every location already exists, so none is created
    data_platform.client.create_location.assert_not_called()

    forecasts = data_platform.forecasts_by_location()

    # every zone is forecast separately, which needs a distinct ml_id per zone
    assert set(forecasts) == {
        "de_50hertz",
        "de_amprion",
        "de_tennet",
        "de_transnetbw",
        "de_national",
    }

    # only the national forecast is saved a second time with the adjuster applied
    assert [f.forecaster.forecaster_name for f in forecasts["de_national"]] == [
        "de_pv_only",
        "de_pv_only_adjust",
        "de_sat_only",
        "de_sat_only_adjust",
    ]
    for zone in ("de_50hertz", "de_amprion", "de_tennet", "de_transnetbw"):
        assert [f.forecaster.forecaster_name for f in forecasts[zone]] == [
            "de_pv_only",
            "de_sat_only",
        ]

    n_fv = 36 * 4  # 36 hours at 15 minute resolution
    for location_forecasts in forecasts.values():
        for forecast in location_forecasts:
            assert len(forecast.values) == n_fv
            assert all(0 <= value.p50_fraction <= 1 for value in forecast.values)
            assert all(value.other_statistics_fractions for value in forecast.values)


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

    n = 3  # 1 site, 3 wind models (2 GenCast + 1 ECMWF)
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
