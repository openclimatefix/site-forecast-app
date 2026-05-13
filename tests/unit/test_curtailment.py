from unittest.mock import MagicMock, patch

import pandas as pd

from site_forecast_app.curtailment import Curtailment


@patch("site_forecast_app.curtailment.EntsoePandasClient")
def test_get_entsoe_day_prices(mock_entsoe_pandas_client):

    start = pd.Timestamp("2026-05-12").tz_localize("UTC")
    end = pd.Timestamp("2026-05-14").tz_localize("UTC")

    mock_da_prices = pd.DataFrame({
        "target_datetime_utc": pd.date_range(start=start, end=end, freq="15min"),
        "price": [10] * (4*24*2+1),   # 2 days +1
    })
    mock_da_prices.set_index("target_datetime_utc", inplace=True)
    mock_entsoe_pandas_client_instance = MagicMock()
    mock_entsoe_pandas_client.return_value = mock_entsoe_pandas_client_instance
    mock_entsoe_pandas_client_instance.query_day_ahead_prices.return_value = mock_da_prices

    prices = Curtailment(now=start).prices_df
    assert prices is not None
    assert isinstance(prices, pd.DataFrame)
    assert not prices.empty
    assert len(prices) == 4*24*2+1 # 15 minute intervals in one day + 1


@patch("site_forecast_app.curtailment.EntsoePandasClient")
def test_make_potential_generation(mock_entsoe_pandas_client):
    start = pd.Timestamp("2026-05-10").tz_localize("UTC")
    end = pd.Timestamp("2026-05-11").tz_localize("UTC")

    data = pd.DataFrame({
        "target_datetime_utc": pd.date_range(start=start, end=end, freq="15min"),
        "forecast_value": [1] * 97,
    })

    mock_da_prices = pd.DataFrame({
        "target_datetime_utc": pd.date_range(start=start, end=end, freq="15min"),
        "price": [10] * 33 + [-1] * 25 + [1]* 39,
    })
    mock_da_prices.set_index("target_datetime_utc", inplace=True)
    mock_entsoe_pandas_client_instance = MagicMock()
    mock_entsoe_pandas_client.return_value = mock_entsoe_pandas_client_instance
    mock_entsoe_pandas_client_instance.query_day_ahead_prices.return_value = mock_da_prices

    curtailment = Curtailment(now=start)
    forecast_values = data.to_dict(orient="records")
    potential_generation = curtailment.apply_curtailment(forecast_values=forecast_values)
    potential_generation = pd.DataFrame(potential_generation)
    assert potential_generation is not None
    assert isinstance(potential_generation, pd.DataFrame)
    assert not potential_generation.empty
    assert len(potential_generation) == 97 # 15 minute intervals in one day + 1
    assert potential_generation.iloc[0].forecast_value == 1 # 00:00 is 1
    assert potential_generation.iloc[48].forecast_value == 1/1.11 #12:00 has been increased

    # we know that the prices at 8:15 to 14:15 was negative
    # lets check these values
    negative_prices_idx =\
          (potential_generation["target_datetime_utc"].dt.time >= pd.Timestamp("08:15").time()) \
        & (data["target_datetime_utc"].dt.time <= pd.Timestamp("14:15").time())
    # check the negative prices are all zero
    assert (potential_generation.loc[negative_prices_idx, "forecast_value"]== 1/1.11).all()
    assert (potential_generation.loc[~negative_prices_idx, "forecast_value"]== 1).all()
