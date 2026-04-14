import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

from dp_sdk.ocf import dp

from nl_blend.init_times import extract_latest_init_times

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forecast value fetching
# ---------------------------------------------------------------------------

async def fetch_dp_forecast_values_as_timeseries(
    client: dp.DataPlatformDataServiceStub,
    location_uuid: str,
    model_name: str,
    start_datetime: Optional[datetime],
) -> list:
    """
    Fetches the latest forecast timeseries from the Data Platform for a single
    model at a given location.

    The UK implementation fetches the single most-recent run for each model and
    returns up to 48 hours of values, filtered to >= start_datetime.

    Args:
        client:         Authenticated Data Platform stub.
        location_uuid:  Target location UUID.
        model_name:     Forecaster name to match (e.g. "nl_regional_48h_pv_ecmwf").
        start_datetime: If provided, only values at or after this time are returned.

    Returns:
        List of Data Platform timeseries value objects, possibly empty.
    """
    logger.debug(
        f"Fetching timeseries for model='{model_name}' location='{location_uuid}'"
    )

    # Single call to get all latest forecasts for this location, then filter
    # by model name – this is the same pattern the UK uses (one call per model
    # within a shared connection opened by the caller).
    response = await client.get_latest_forecasts(
        dp.GetLatestForecastsRequest(
            location_uuid=location_uuid,
            energy_source=dp.EnergySource.SOLAR,
        )
    )

    matching = [
        f for f in response.forecasts
        if f.forecaster.forecaster_name == model_name
    ]

    if not matching:
        logger.warning(
            f"No forecast found for model='{model_name}' at location='{location_uuid}'"
        )
        return []

    # Take the most recently initialised run
    forecast = matching[0]

    timeseries_response = await client.get_forecast_as_timeseries(
        dp.GetForecastAsTimeseriesRequest(
            location_uuid=location_uuid,
            energy_source=dp.EnergySource.SOLAR,
            forecaster=forecast.forecaster,
            time_window=dp.TimeWindow(
                start_timestamp_utc=forecast.initialization_timestamp_utc,
                end_timestamp_utc=forecast.initialization_timestamp_utc
                + timedelta(hours=48),
            ),
        )
    )

    values = timeseries_response.values

    if start_datetime is not None:
        if start_datetime.tzinfo is None:
            start_datetime = start_datetime.replace(tzinfo=timezone.utc)
        filtered = []
        for v in values:
            t = _to_aware_datetime(v.target_timestamp_utc)
            if t >= start_datetime:
                filtered.append(v)
        values = filtered

    return values


async def get_all_forecast_values_as_dataframe(
    client: dp.DataPlatformDataServiceStub,
    location_uuid: str,
    model_name: str,
    start_datetime: Optional[datetime],
) -> pd.DataFrame:
    """
    Fetches the latest forecast timeseries for a model and returns a tidy
    long-format DataFrame.

    Columns:
        target_time                          – UTC datetime of the forecast step
        expected_power_generation_megawatts  – p50 value converted to MW
        created_utc                          – wall-clock time of the fetch
        model_name                           – name of the source model

    Returns an empty DataFrame (with the correct columns) when no data is found,
    matching the UK handling for absent models.
    """
    _EMPTY = pd.DataFrame(
        columns=[
            "target_time",
            "expected_power_generation_megawatts",
            "created_utc",
            "model_name",
        ]
    )

    dp_values = await fetch_dp_forecast_values_as_timeseries(
        client=client,
        location_uuid=location_uuid,
        model_name=model_name,
        start_datetime=start_datetime,
    )

    if not dp_values:
        return _EMPTY

    now = datetime.now(timezone.utc)
    rows = []
    for v in dp_values:
        capacity_mw = v.effective_capacity_watts / 1_000_000
        p50_mw = v.p50_value_fraction * capacity_mw

        target_time = _to_aware_datetime(v.target_timestamp_utc).replace(microsecond=0)

        rows.append(
            {
                "target_time": target_time,
                "expected_power_generation_megawatts": p50_mw,
                "created_utc": now,
                "model_name": model_name,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Initialisation-time fetching (used by weights.py)
# ---------------------------------------------------------------------------

async def fetch_latest_nl_init_times(
    client: dp.DataPlatformDataServiceStub,
    location_uuid: str,
    model_names: List[str],
    t0: pd.Timestamp,
    max_delay: pd.Timedelta,
) -> Dict[str, pd.Timestamp]:
    """
    Fetches all latest forecasts from Data Platform in a single call and
    extracts the most-recent valid initialisation time for each requested model.

    A single DP call is made (matching the UK single-pass pattern) and the
    pure `extract_latest_init_times` function handles the filtering/selection
    logic so it remains fully unit-testable without a live stub.

    Args:
        client:       Authenticated Data Platform stub.
        location_uuid: Target location UUID.
        model_names:  Models we want init times for.
        t0:           Blend reference time.
        max_delay:    Forecasts older than (t0 - max_delay) are ignored.

    Returns:
        Dict mapping model name -> latest valid initialisation timestamp (UTC).
        Returns an empty dict on failure so the caller can apply fallbacks.
    """
    try:
        response = await client.get_latest_forecasts(
            dp.GetLatestForecastsRequest(
                location_uuid=location_uuid,
                energy_source=dp.EnergySource.SOLAR,
            )
        )
        return extract_latest_init_times(
            forecasts=list(response.forecasts),
            model_names=model_names,
            t0=t0,
            max_delay=max_delay,
        )
    except Exception:
        logger.warning(
            "Failed to fetch latest forecasts from Data Platform while "
            "resolving init times.",
            exc_info=True,
        )
        return {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_aware_datetime(ts) -> datetime:
    """
    Converts a protobuf Timestamp or plain datetime to a timezone-aware UTC
    datetime, normalising the various forms that come back from the DP SDK.
    """
    if hasattr(ts, "ToDatetime"):
        dt = ts.ToDatetime()
    else:
        dt = ts
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt