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
    # 7000 is first forecast value, 4000 is gen value at t0 time
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


def test_feather_forecast_dtype_corruption():
    """Regression test: .where(drop=True) can corrupt time_utc dtype from datetime64 to int64.

    Ensures feather_forecast handles forward-filled generation data without raising:
    TypeError: Cannot compare dtypes int64 and datetime64[ns]

    We explicitly simulate the corruption by encoding time_utc as int64 (nanoseconds
    since epoch), which is exactly what xarray produces internally in affected versions
    after .where(drop=True). This makes the test deterministic regardless of xarray version.
    """
    start_times = pd.date_range("2024-01-01", periods=12, freq="h")
    end_times = start_times + pd.Timedelta(minutes=30)
    forecast_values = np.arange(7000, 19000, 1000)

    df_values = pd.DataFrame(
        {"start_utc": start_times, "end_utc": end_times, "forecast_power_kw": forecast_values},
    )

    # Simulate dtype corruption: encode time_utc as int64 (nanoseconds since epoch),
    # which is what xarray produces after .where(drop=True) in affected versions.
    int64_times = start_times.astype(np.int64)
    gen_values = np.array([[1000, 1000, 1000, 1000, 2000, 2000, 3000, 4000, 4000, 4000, 4000, 4000]])  # noqa: E501
    generation_values_kw = xr.DataArray(
        gen_values,
        coords={"location_id": [1], "time_utc": int64_times},
        dims=["location_id", "time_utc"],
    )
    generation_xr = xr.Dataset({"generation_kw": generation_values_kw})

    # Without the assign_coords fix this raises:
    # TypeError: Cannot compare dtypes int64 and datetime64[ns]
    feathered_preds = feather_forecast(
        df_values, t0_time=pd.Timestamp("2024-01-01 03:00:00"), generation_data=generation_xr,
    )

    assert len(feathered_preds) == len(df_values)
    # First value should be blended away from raw forecast — confirms feathering ran
    assert feathered_preds.iloc[0]["forecast_power_kw"] != forecast_values[0]


def test_feather_forecast_all_fill_values():
    """Regression test: if all generation values are fill values, .where(drop=True) produces
    an empty array. feather_forecast should skip feathering and return the original forecast.
    """
    start_times = pd.date_range("2024-01-01", periods=12, freq="h")
    end_times = start_times + pd.Timedelta(minutes=30)
    forecast_values = np.arange(7000, 19000, 1000)

    df_values = pd.DataFrame(
        {"start_utc": start_times, "end_utc": end_times, "forecast_power_kw": forecast_values},
    )

    # All identical fill values — mask drops everything, producing an empty filtered array
    gen_values = np.full((1, 12), 0.00001)
    generation_values_kw = xr.DataArray(
        gen_values,
        coords={"location_id": [1], "time_utc": start_times},
        dims=["location_id", "time_utc"],
    )
    generation_xr = xr.Dataset({"generation_kw": generation_values_kw})

    # Must NOT crash — should return forecast unchanged
    feathered_preds = feather_forecast(
        df_values, t0_time=pd.Timestamp("2024-01-01 03:00:00"), generation_data=generation_xr,
    )

    assert np.allclose(feathered_preds["forecast_power_kw"].values, forecast_values)
