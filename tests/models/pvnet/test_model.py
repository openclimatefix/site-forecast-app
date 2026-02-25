import numpy as np
import pandas as pd
import xarray as xr

from site_forecast_app.models.pvnet.model import feather_forecast


def test_feather_forecast():
    """Test for feathering the forecast to the latest value of generation"""
    # Setup the forecast values dataframe
    start_times = pd.date_range("2024-01-01", periods=12, freq="h")
    end_times = start_times + pd.Timedelta(minutes=30)
    forecast_values = np.arange(7000, 19000, 1000)

    # 4. Create the DataFrame
    df_values = pd.DataFrame(
        {"start_utc": start_times, "end_utc": end_times, "forecast_power_kw": forecast_values},
    )

    # set up generation data
    location_ids = [1]

    generation_values_kw = xr.DataArray(
        np.array([np.arange(1000, 13000, 1000)]),
        coords={"location_id": location_ids, "time_utc": start_times},
        dims=["location_id", "time_utc"],
    )

    generation_xr = xr.Dataset({"generation_kw": generation_values_kw})

    # check that the first 8 timesteps are adjusted to the latest value of generation
    # first timestamp should be (7000 - (7000-4000)*0.8) = 4600
    feathered_preds = feather_forecast(
        df_values, t0_time=pd.Timestamp("2024-01-01 03:00:00"), generation_data=generation_xr,
    )
    assert np.allclose(
        feathered_preds.iloc[:8]["forecast_power_kw"],
        [4600.0, 5200.0, 6000.0, 7000.0, 8200.0, 9600.0, 11200.0, 13000.0],
    )


def test_feather_forecast_old_gen():
    """Test for feathering the forecast to the latest value of generation"""
    # Setup the forecast values dataframe
    start_times = pd.date_range("2024-01-01", periods=12, freq="h")
    end_times = start_times + pd.Timedelta(minutes=30)
    forecast_values = np.arange(7000, 19000, 1000)

    start_times_old = pd.date_range("2023-01-01", periods=12, freq="h")

    # 4. Create the DataFrame
    df_values = pd.DataFrame(
        {"start_utc": start_times, "end_utc": end_times, "forecast_power_kw": forecast_values},
    )

    # set up generation data
    location_ids = [1]

    generation_values_kw = xr.DataArray(
        np.array([np.arange(1000, 13000, 1000)]),
        coords={"location_id": location_ids, "time_utc": start_times_old},
        dims=["location_id", "time_utc"],
    )

    generation_xr = xr.Dataset({"generation_kw": generation_values_kw})

    # Should be no change as generation values are stale
    feathered_preds = feather_forecast(
        df_values, t0_time=pd.Timestamp("2024-01-01 03:00:00"), generation_data=generation_xr,
    )
    assert np.allclose(
        feathered_preds.iloc[:8]["forecast_power_kw"], np.arange(7000, 19000, 1000)[:8],
    )
