import math
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from blend.data_platform import (
    build_forecast_value_objects,
    get_all_forecast_values_as_dataframe,
)

# ---------------------------------------------------------------------------
# Tests for get_all_forecast_values_as_dataframe
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_all_forecast_values_as_dataframe_empty():
    """Verify empty response is correctly handled."""
    mock_client = AsyncMock()

    with patch(
        "blend.data_platform.fetch_dp_forecast_values_as_timeseries",
        new_callable=AsyncMock,
    ) as fetch_mock:
        fetch_mock.return_value = []

        df = await get_all_forecast_values_as_dataframe(
            client=mock_client,
            location_uuid="some-uuid",
            model_name="test-model",
            start_datetime=None,
        )

    assert df.empty
    assert "target_time" in df.columns
    assert "expected_power_generation_megawatts" in df.columns
    assert "p10_mw" in df.columns


@pytest.mark.asyncio
async def test_get_all_forecast_values_as_dataframe_success():
    """Test standard mapping of timeseries values into a DataFrame."""
    mock_client = AsyncMock()

    # Mocking protobuf-like values
    mock_val1 = MagicMock()
    mock_val1.p50_value_fraction = 15.5
    mock_val1.other_statistics_fractions = {"p10": 10.0, "p90": 20.0}
    mock_val1.target_timestamp_utc = datetime(2024, 6, 1, 10, 0, tzinfo=UTC)

    # Value without optional p10/p90
    mock_val2 = MagicMock()
    mock_val2.p50_value_fraction = 7.0
    mock_val2.other_statistics_fractions = {}
    mock_val2.target_timestamp_utc = datetime(2024, 6, 1, 10, 30, tzinfo=UTC)

    with patch(
        "blend.data_platform.fetch_dp_forecast_values_as_timeseries",
        new_callable=AsyncMock,
    ) as fetch_mock:
        fetch_mock.return_value = [mock_val1, mock_val2]

        df = await get_all_forecast_values_as_dataframe(
            client=mock_client,
            location_uuid="some-uuid",
            model_name="test-model",
            start_datetime=None,
        )

    assert len(df) == 2
    assert df.iloc[0]["expected_power_generation_megawatts"] == 15.5
    assert df.iloc[0]["p10_mw"] == 10.0
    assert df.iloc[0]["p90_mw"] == 20.0
    assert df.iloc[0]["model_name"] == "test-model"

    assert df.iloc[1]["expected_power_generation_megawatts"] == 7.0
    assert math.isnan(df.iloc[1]["p10_mw"])
    assert math.isnan(df.iloc[1]["p90_mw"])

# ---------------------------------------------------------------------------
# Tests for build_forecast_value_objects
# ---------------------------------------------------------------------------

def test_build_forecast_value_objects():
    """Verify DataFrame to payload conversion bypasses bounding and uses float raw MW values."""
    t0 = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)

    blended_df = pd.DataFrame(
        [
            {
                "target_time": datetime(2024, 6, 1, 12, 30, tzinfo=UTC),
                "expected_power_generation_megawatts": 20.0,
                "p10_mw": 15.5,
                "p90_mw": 25.0,
            },
            {
                "target_time": datetime(2024, 6, 1, 13, 0, tzinfo=UTC),
                "expected_power_generation_megawatts": 35.0,
                "p10_mw": float("nan"), # Missing p10
                "p90_mw": float("nan"),
            },
        ],
    )

    values = build_forecast_value_objects(
        blended_df=blended_df,
        init_time_utc=t0,
    )

    assert len(values) == 2

    # Item 1: half an hour later => 30 mins
    assert values[0].horizon_mins == 30
    assert values[0].p50_fraction == 20.0
    assert values[0].other_statistics_fractions["p10"] == 15.5
    assert values[0].other_statistics_fractions["p90"] == 25.0

    # Item 2: one hour later => 60 mins
    assert values[1].horizon_mins == 60
    assert values[1].p50_fraction == 35.0
    assert "p10" not in values[1].other_statistics_fractions
    assert "p90" not in values[1].other_statistics_fractions
