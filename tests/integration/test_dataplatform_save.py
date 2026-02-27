"""Integration-style test for Data Platform save path."""
import datetime as dt

import pandas as pd
import pytest
from dp_sdk.ocf import dp

from site_forecast_app.save import save_forecast, save_forecast_to_dataplatform


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="module")
async def test_save_forecast_triggers_dataplatform(
    monkeypatch, db_session, sites, forecast_values, client,
):
    """Ensure save_forecast calls the Data Platform helper when enabled."""

    monkeypatch.setenv("SAVE_TO_DATA_PLATFORM", "true")
    # Point DP_HOST/PORT to the testcontainer started by the client fixture
    # Access private attributes as grpclib Channel doesn't expose them publicly
    monkeypatch.setenv("DP_HOST", client.channel._host)
    monkeypatch.setenv("DP_PORT", str(client.channel._port))

    # 1. Create a Location in DP so save_forecast can find it by name
    site = sites[0]
    # Ensure site has a name we can use
    site_name = site.client_location_name or "test_site_name_fixture"

    create_location_request = dp.CreateLocationRequest(
        location_name=site_name,
        energy_source=dp.EnergySource.SOLAR,
        geometry_wkt="POINT(0 0)",
        location_type=dp.LocationType.SITE,
        effective_capacity_watts=int(site.capacity_kw * 1000),
        valid_from_utc=dt.datetime(2020, 1, 1, tzinfo=dt.UTC),
    )
    resp = await client.create_location(create_location_request)
    dp_location_uuid = resp.location_uuid

    forecast = {
        "meta": {
            "location_uuid": site.location_uuid,
            "version": "0.0.0-test",
            "timestamp": forecast_values["start_utc"][0],
            "client_location_name": site_name,
        },
        "values": [
            {
                "start_utc": s,
                "end_utc": e,
                "forecast_power_kw": p,
            }
            for s, e, p in zip(
                forecast_values["start_utc"],
                forecast_values["end_utc"],
                forecast_values["forecast_power_kw"],
                strict=False,
            )
        ],
    }

    # Monkeypatch asyncio.run to capture tasks generated inside save_forecast
    # This prevents RuntimeError because asyncio.run() cannot be called when an
    # event loop is running.
    import asyncio
    captured_tasks = []

    def fake_run(coro):
        task = asyncio.create_task(coro)
        captured_tasks.append(task)
        return None

    monkeypatch.setattr("site_forecast_app.dataplatform.asyncio.run", fake_run)

    save_forecast(
        db_session,
        forecast,
        write_to_db=False,
        ml_model_name="test-model",
        ml_model_version="0.0.0-test",
        use_adjuster=False,
    )

    # Allow async tasks created by save_forecast to complete
    if captured_tasks:
        await asyncio.gather(*captured_tasks)

    # Verify that the forecast was actually created in the DP
    # We can check by listing forecasters or getting the forecast

    forecaster_name = "test_model" # save_forecast replaces '-' with '_'
    list_forecasters_request = dp.ListForecastersRequest(
        forecaster_names_filter=[forecaster_name],
    )
    list_forecasters_response = await client.list_forecasters(list_forecasters_request)

    assert len(list_forecasters_response.forecasters) > 0, "Forecaster not created in DP"
    forecaster = list_forecasters_response.forecasters[0]

    # Check for forecast values
    # We look for the first timestamp
    start_ts = forecast["values"][0]["start_utc"]
    end_ts = forecast["values"][-1]["end_utc"]

    # Widen the window slightly to avoid boundary issues
    query_start = start_ts - dt.timedelta(minutes=1)
    query_end = end_ts + dt.timedelta(minutes=1)

    get_forecast_request = dp.GetForecastAsTimeseriesRequest(
        location_uuid=dp_location_uuid,
        energy_source=dp.EnergySource.SOLAR,
        time_window=dp.TimeWindow(
            start_timestamp_utc=query_start,
            end_timestamp_utc=query_end,
        ),
        forecaster=forecaster,
    )

    forecast_resp = await client.get_forecast_as_timeseries(get_forecast_request)
    assert len(forecast_resp.values) == len(forecast["values"]), (
        f"Forecast values count mismatch in DP. Expected {len(forecast['values'])}, "
        f"got {len(forecast_resp.values)}. Response: {forecast_resp.values}"
    )


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="module")
async def test_save_forecast_sends_adjusted_forecast(client, sites):
    """Ensure adjusted forecast also saves to DP via the real client.

    With use_adjuster=True, save_forecast_to_dataplatform must write:
    - a base forecast (under 'test_adj_model')
    - an adjusted forecast (under 'test_adj_model_adjust')

    We verify both forecasters exist in the real Data Platform and each
    has at least one forecast value stored.
    """
    site = sites[0]
    # Use a fixed name that will never collide with the location created by
    # test_save_forecast_triggers_dataplatform (which uses site.client_location_name).
    site_name = "test_adj_site"

    # 1. Register the location in the real DP
    # Use a small polygon — the DP requires valid, closed WGS84 geometry for SITE type.
    geometry_wkt = "POLYGON ((0.001 0.001, 0.002 0.001, 0.002 0.002, 0.001 0.002, 0.001 0.001))"
    create_location_request = dp.CreateLocationRequest(
        location_name=site_name,
        energy_source=dp.EnergySource.SOLAR,
        geometry_wkt=geometry_wkt,
        location_type=dp.LocationType.SITE,
        effective_capacity_watts=int(site.capacity_kw * 1000),
        valid_from_utc=dt.datetime(2020, 1, 1, tzinfo=dt.UTC),
    )
    loc_resp = await client.create_location(create_location_request)
    dp_location_uuid = loc_resp.location_uuid

    # 2. Build a forecast dataframe — DP requires at least 2 values per request
    init_time = dt.datetime(2025, 6, 1, 12, 0, 0, tzinfo=dt.UTC)
    forecast_df = pd.DataFrame(
        {
            "start_utc": [
                init_time + dt.timedelta(minutes=15),
                init_time + dt.timedelta(minutes=30),
            ],
            "end_utc": [
                init_time + dt.timedelta(minutes=30),
                init_time + dt.timedelta(minutes=45),
            ],
            "forecast_power_kw": [5.0, 6.0],
            "horizon_minutes": [15, 30],
        },
    )

    # 3. Save with adjuster enabled — this should create both base + _adjust forecasts
    await save_forecast_to_dataplatform(
        forecast_df=forecast_df,
        client_location_name=site_name,
        model_tag="test-adj-model",
        init_time_utc=init_time,
        client=client,
        use_adjuster=True,
    )

    # 4. Verify base forecaster ("test_adj_model") was created and has values
    base_name = "test_adj_model"
    base_list_resp = await client.list_forecasters(
        dp.ListForecastersRequest(forecaster_names_filter=[base_name]),
    )
    assert len(base_list_resp.forecasters) > 0, f"Base forecaster '{base_name}' not found in DP"
    base_forecaster = base_list_resp.forecasters[0]

    base_forecast_resp = await client.get_forecast_as_timeseries(
        dp.GetForecastAsTimeseriesRequest(
            location_uuid=dp_location_uuid,
            energy_source=dp.EnergySource.SOLAR,
            time_window=dp.TimeWindow(
                start_timestamp_utc=init_time,
                end_timestamp_utc=init_time + dt.timedelta(hours=1),
            ),
            forecaster=base_forecaster,
        ),
    )
    assert len(base_forecast_resp.values) >= 1, "No base forecast values found in DP"

    # 5. Verify adjusted forecaster ("test_adj_model_adjust") was created and has values
    adj_name = "test_adj_model_adjust"
    adj_list_resp = await client.list_forecasters(
        dp.ListForecastersRequest(forecaster_names_filter=[adj_name]),
    )
    assert len(adj_list_resp.forecasters) > 0, f"Adjusted forecaster '{adj_name}' not found in DP"
    adj_forecaster = adj_list_resp.forecasters[0]

    adj_forecast_resp = await client.get_forecast_as_timeseries(
        dp.GetForecastAsTimeseriesRequest(
            location_uuid=dp_location_uuid,
            energy_source=dp.EnergySource.SOLAR,
            time_window=dp.TimeWindow(
                start_timestamp_utc=init_time,
                end_timestamp_utc=init_time + dt.timedelta(hours=1),
            ),
            forecaster=adj_forecaster,
        ),
    )
    assert len(adj_forecast_resp.values) >= 1, "No adjusted forecast values found in DP"


@pytest.mark.asyncio(loop_scope="module")
async def test_save_forecast_to_dataplatform_integration(client):
    """
    Test saving data to the Data Platform (Integration).
    """

    # 1. Create a Location in DP
    # We need a location UUID. Let's generate one.

    # Try creating with Site type, fallback to something else if fails
    # assuming LocationType.Site exists or is valid.
    # If not, user might need to adjust. But let's assume valid based on usage in other tests.
    # Solar consumer used LocationType.GSP.
    create_location_request = dp.CreateLocationRequest(
        location_name="test_site_integration",
        energy_source=dp.EnergySource.SOLAR,
        geometry_wkt="POINT(0 0)",
        location_type=dp.LocationType.SITE,
        effective_capacity_watts=10_000,
        valid_from_utc=dt.datetime(2020, 1, 1, tzinfo=dt.UTC),
    )

    location_response = await client.create_location(create_location_request)
    dp_location_uuid = location_response.location_uuid

    # 2. Prepare fake forecast data
    init_time = dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.UTC)

    fake_data = pd.DataFrame(
        {
            "start_utc": [
                init_time + dt.timedelta(minutes=15),
                init_time + dt.timedelta(minutes=30),
            ],
            "end_utc": [
                init_time + dt.timedelta(minutes=30),
                init_time + dt.timedelta(minutes=45),
            ],
            "forecast_power_kw": [5.0, 8.0],  # 5 kW and 8 kW
            "horizon_minutes": [15, 30],
            # probabilistic_values if needed
        },
    )

    # 3. Call save_forecast_to_dataplatform
    # We pass client_location_name="test_site_integration".
    # The function should map it to dp_location_uuid.
    await save_forecast_to_dataplatform(
        forecast_df=fake_data,
        client_location_name="test_site_integration",
        model_tag="test-integration-mn",
        init_time_utc=init_time,
        client=client,
        use_adjuster=False,
    )

    # 4. Verify data in DP using GetForecast or ListForecasts
    # We need to find the forecaster first
    forecaster_name = "test_integration_mn"
    list_forecasters_request = dp.ListForecastersRequest(
        forecaster_names_filter=[forecaster_name],
    )
    list_forecasters_response = await client.list_forecasters(list_forecasters_request)

    assert len(list_forecasters_response.forecasters) > 0
    forecaster = list_forecasters_response.forecasters[0]

    # Get Forecast values
    start_ts_dp = init_time + dt.timedelta(minutes=15)
    end_ts_dp = init_time + dt.timedelta(minutes=45)

    get_forecast_request = dp.GetForecastAsTimeseriesRequest(
        location_uuid=dp_location_uuid,
        energy_source=dp.EnergySource.SOLAR,
        time_window=dp.TimeWindow(
            start_timestamp_utc=start_ts_dp,
            end_timestamp_utc=end_ts_dp,
        ),
        forecaster=forecaster,
        # This filter might reduce result to 1 if we only want horizon 15.
        horizon_mins=15,
    )
    # Removing horizon_mins filter or explicitly checking for specific one
    # If I remove it, I should get both. But GetForecastAsTimeseriesRequest
    # definition showed optional horizon_mins.
    # Let's remove horizon_mins filter to get both.

    get_forecast_request = dp.GetForecastAsTimeseriesRequest(
        location_uuid=dp_location_uuid,
        energy_source=dp.EnergySource.SOLAR,
        time_window=dp.TimeWindow(
            start_timestamp_utc=start_ts_dp,
            end_timestamp_utc=end_ts_dp,
        ),
        forecaster=forecaster,
    )

    forecast_resp = await client.get_forecast_as_timeseries(get_forecast_request)
    assert len(forecast_resp.values) == 2

    # Check values
    # value is p50_fraction = (kW * 1000) / capacity_watts
    # capacity = 10,000 W.
    # 5 kW = 0.5 fraction (horizon 15)
    # 8 kW = 0.8 fraction (horizon 30)

    # Values might not be sorted by time, so we convert to list and sort or check by existence
    fractions = sorted([v.p50_value_fraction for v in forecast_resp.values])
    assert abs(fractions[0] - 0.5) < 1e-6
    assert abs(fractions[1] - 0.8) < 1e-6



