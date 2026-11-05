import json
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from ocf.dp.dp_data import messages_pb2

from site_forecast_app.curtailment import Curtailment


@patch("site_forecast_app.curtailment.EntsoePandasClient")
def test_get_entsoe_day_prices(mock_entsoe_pandas_client, mock_da_prices):

    mock_entsoe_pandas_client_instance = MagicMock()
    mock_entsoe_pandas_client.return_value = mock_entsoe_pandas_client_instance
    mock_entsoe_pandas_client_instance.query_day_ahead_prices.return_value = mock_da_prices

    prices = Curtailment(now=mock_da_prices.index.min()).prices_df
    assert prices is not None
    assert isinstance(prices, pd.DataFrame)
    assert not prices.empty
    assert len(prices) == 4*24+1 # 15 minute intervals in one day + 1


@patch("site_forecast_app.curtailment.EntsoePandasClient")
def test_make_potential_generation(mock_entsoe_pandas_client, mock_da_prices):
    start = pd.Timestamp("2026-05-12").tz_localize("UTC")
    end = pd.Timestamp("2026-05-13").tz_localize("UTC")

    data = pd.DataFrame({
        "target_datetime_utc": pd.date_range(start=start, end=end, freq="15min"),
        "forecast_power": [1] * 97,
        "p10": [0.8] * 97,
        "p90": [1.2] * 97,
    })

    data = data[["target_datetime_utc", "forecast_power", "p10", "p90"]]

    # add negative prices
    negative_prices_idx =\
          (mock_da_prices.index.time >= pd.Timestamp("08:15").time()) \
        & (mock_da_prices.index.time <= pd.Timestamp("14:15").time())
    mock_da_prices.loc[negative_prices_idx, "price"] = -1

    mock_entsoe_pandas_client_instance = MagicMock()
    mock_entsoe_pandas_client.return_value = mock_entsoe_pandas_client_instance
    mock_entsoe_pandas_client_instance.query_day_ahead_prices.return_value = mock_da_prices

    curtailment = Curtailment(now=start)
    forecast_values = data.to_dict("records")
    potential_generation = curtailment.apply_curtailment(forecast_values=forecast_values)
    potential_generation = pd.DataFrame(potential_generation)
    assert potential_generation is not None
    assert isinstance(potential_generation, pd.DataFrame)
    assert not potential_generation.empty
    assert len(potential_generation) == 97 # 15 minute intervals in one day + 1
    assert potential_generation.iloc[0].forecast_power == 1 # 00:00 is 1
    assert potential_generation.iloc[48].forecast_power == 1/1.11 #12:00 has been increased
    assert potential_generation.iloc[0].probabilistic_values == json.dumps({"p10": 0.8, "p90": 1.2})
    assert potential_generation.iloc[48].probabilistic_values \
        == json.dumps({"p10": 0.8/1.11, "p90": 1.2/1.11})

    # we know that the prices at 8:15 to 14:15 was negative
    # lets check these values
    negative_prices_idx =\
          (potential_generation["target_datetime_utc"].dt.time >= pd.Timestamp("08:15").time()) \
        & (data["target_datetime_utc"].dt.time <= pd.Timestamp("14:15").time())
    # check the negative prices are all zero
    assert (potential_generation.loc[negative_prices_idx, "forecast_power"]== 1/1.11).all()
    assert (potential_generation.loc[~negative_prices_idx, "forecast_power"]== 1).all()

@pytest.mark.asyncio(loop_scope="session")
@patch("site_forecast_app.curtailment.EntsoePandasClient")
async def test_apply_curtailment_on_model(mock_entsoe_pandas_client, mock_da_prices):
    start = pd.Timestamp("2026-05-12").tz_localize("UTC")

    # add negative prices
    negative_prices_idx =\
          (mock_da_prices.index.time >= pd.Timestamp("08:15").time()) \
        & (mock_da_prices.index.time <= pd.Timestamp("14:15").time())
    mock_da_prices.loc[negative_prices_idx, "price"] = -1

    mock_entsoe_pandas_client_instance = MagicMock()
    mock_entsoe_pandas_client.return_value = mock_entsoe_pandas_client_instance
    mock_entsoe_pandas_client_instance.query_day_ahead_prices.return_value = mock_da_prices

    # set up fake dp
    client = AsyncMock()
    client.get_latest_forecasts = \
        AsyncMock(
            return_value=messages_pb2.GetLatestForecastsResponse(
                forecasts=[
                    messages_pb2.GetLatestForecastsResponse.Forecast(
                        initialization_timestamp_utc=start,
                        location_uuid="location-uuid",
                        forecaster=messages_pb2.Forecaster(
                                forecaster_name="test_uncurtailed",
                                forecaster_version="1.0",
                            ),
                        ),
                    ],
                ),
            )

    client.get_forecast_as_timeseries = AsyncMock(
        return_value=messages_pb2.GetForecastAsTimeseriesResponse(
            values=[
                messages_pb2.GetForecastAsTimeseriesResponse.Value(
                    target_timestamp_utc=start + timedelta(minutes=15 * i),
                    p50_value_fraction=i/100,
                    other_statistics_fractions={"p10": 0.8, "p90": 1.2},
                )
                for i in range(97)
            ],
        ),
    )

    # run curtailment
    curtailment = Curtailment(now=start)
    await curtailment.apply_curtailment_on_model(model_name="test_uncurtailed",
                                           location_map={"nl_national": "test_location"},
                                           new_model_name="test",
                                           t0=start,
                                           dp_client=client)

    # check that the curtailment results are as expected
    curtailment_results = curtailment.curtailment_results
    assert len(curtailment_results) == 97
    for i in range(97):
        row = curtailment_results.iloc[i]
        time = row.target_time.time()
        if (time >= pd.Timestamp("08:15").time()) and (time <= pd.Timestamp("14:15").time()):
            assert np.round(row["forecast_power"], 4) == np.round(i/100 / 1.11, 4)
        else:
            assert np.round(row["forecast_power"], 4) == np.round(i/100,4)
