
import pandas as pd

from nl_blend.init_times import (
    calculate_model_delays,
    extract_latest_init_times,
    shift_mae_curves,
)


def test_calculate_model_delays():
    t0 = pd.Timestamp("2024-01-01 12:00", tz="UTC")

    model_init_times = {
        "model_A": pd.Timestamp("2024-01-01 11:30", tz="UTC"),
        "model_B": pd.Timestamp("2024-01-01 11:45", tz="UTC"), # floor("15min") stays 11:45 => 15min delay
        "model_C": pd.Timestamp("2024-01-01 10:00", tz="UTC"),
        "model_D": None,
    }

    delays = calculate_model_delays(model_init_times, t0)

    assert delays["model_A"] == pd.Timedelta("30min")
    assert delays["model_B"] == pd.Timedelta("15min") # floor("15min"): 11:45 -> 11:45 => 15min delay
    assert delays["model_C"] == pd.Timedelta("120min")
    assert "model_D" not in delays

def test_shift_mae_curves():
    # Construct a dummy DataFrame
    horizons = [
        pd.Timedelta("30min"),
        pd.Timedelta("60min"),
        pd.Timedelta("90min"),
        pd.Timedelta("120min"),
    ]

    df_mae = pd.DataFrame(
        {
            "model_A": [1.0, 2.0, 3.0, 4.0],
            "model_B": [1.5, 2.5, 3.5, 4.5],
        },
        index=horizons,
    )

    delays = {
        "model_A": pd.Timedelta("30min"), # shifted by 1 index
        "model_B": pd.Timedelta("60min"), # shifted by 2 indices
    }

    df_shifted = shift_mae_curves(df_mae, delays)

    # model_A originally at index 60min has value 2.0 -> now it's at index 30min
    assert df_shifted.loc[pd.Timedelta("30min"), "model_A"] == 2.0
    assert df_shifted.loc[pd.Timedelta("60min"), "model_A"] == 3.0

    # model_B originally at index 90min has value 3.5 -> now it's at index 30min
    assert df_shifted.loc[pd.Timedelta("30min"), "model_B"] == 3.5
    assert df_shifted.loc[pd.Timedelta("60min"), "model_B"] == 4.5

    # Assert earlier horizons below MIN_FORECAST_HORIZON are dropped
    assert pd.Timedelta("0min") not in df_shifted.index

def test_extract_latest_init_times():
    t0 = pd.Timestamp("2024-01-01 12:00", tz="UTC")
    max_delay = pd.Timedelta("3h") # Cutoff is 09:00 UTC

    # Mock grpc-like objects
    class MockForecaster:
        def __init__(self, name):
            self.forecaster_name = name

    class MockForecast:
        def __init__(self, name, ts):
            self.forecaster = MockForecaster(name)
            self.initialization_timestamp_utc = ts

    forecasts = [
        MockForecast("model_A", pd.Timestamp("2024-01-01 11:30", tz="UTC")),
        MockForecast("model_A", pd.Timestamp("2024-01-01 11:00", tz="UTC")), # Older
        MockForecast("model_B", pd.Timestamp("2024-01-01 08:00", tz="UTC")), # Too old
        MockForecast("model_C", pd.Timestamp("2024-01-01 10:00", tz="UTC")),
    ]

    result = extract_latest_init_times(forecasts, ["model_A", "model_B", "model_C"], t0, max_delay)

    assert result["model_A"] == pd.Timestamp("2024-01-01 11:30", tz="UTC")
    assert "model_B" not in result # Filtered out
    assert result["model_C"] == pd.Timestamp("2024-01-01 10:00", tz="UTC")
