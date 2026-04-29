"""Data Platform I/O helpers for the NL site blending service."""
import logging
import math
from datetime import UTC, datetime, timedelta

import pandas as pd
from dp_sdk.ocf import dp

from site_forecast_app.blend.init_times import extract_latest_init_times

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forecast value fetching
# ---------------------------------------------------------------------------

async def fetch_dp_forecast_values_as_timeseries(
    client: dp.DataPlatformDataServiceStub,
    location_uuid: str,
    model_name: str,
    start_datetime: datetime | None,
) -> list:
    """Fetches latest forecast timeseries from Data Platform for a single model.

    Fetches the single most-recent run for each model and returns up to 48
    hours of values, filtered to >= start_datetime.

    Args:
        client:         Authenticated Data Platform stub.
        location_uuid:  Target location UUID.
        model_name:     Forecaster name to match (e.g. "nl_regional_48h_pv_ecmwf").
        start_datetime: If provided, only values at or after this time are returned.

    Returns:
        List of Data Platform timeseries value objects, possibly empty.
    """
    logger.debug(
        f"Fetching timeseries for model='{model_name}' location='{location_uuid}'",
    )

    # Single call to get all latest forecasts for this location, then filter
    # within the shared connection opened by the caller.
    response = await client.get_latest_forecasts(
        dp.GetLatestForecastsRequest(
            location_uuid=location_uuid,
            energy_source=dp.EnergySource.SOLAR,
        ),
    )

    matching = [
        f for f in response.forecasts
        if f.forecaster.forecaster_name == model_name
    ]

    if not matching:
        logger.warning(
            f"No forecast found for model='{model_name}' at location='{location_uuid}'",
        )
        return []

    # Take the most recently initialised run.
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
        ),
    )

    values = timeseries_response.values

    if start_datetime is not None:
        if start_datetime.tzinfo is None:
            start_datetime = start_datetime.replace(tzinfo=UTC)
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
    start_datetime: datetime | None,
) -> pd.DataFrame:
    """Fetches latest forecast timeseries for a model and returns a tidy DataFrame.

    Columns
    -------
    target_time                          - UTC datetime of the forecast step
    expected_power_generation_megawatts  - p50 value converted to MW
    p10_mw                               - p10 value converted to MW (NaN if absent)
    p90_mw                               - p90 value converted to MW (NaN if absent)
    created_utc                          - wall-clock time of the fetch
    model_name                           - name of the source model

    Returns an empty DataFrame (with the correct columns) when no data is found.
    """
    _EMPTY = pd.DataFrame(
        columns=[
            "target_time",
            "expected_power_generation_megawatts",
            "p10_mw",
            "p90_mw",
            "created_utc",
            "model_name",
        ],
    )

    dp_values = await fetch_dp_forecast_values_as_timeseries(
        client=client,
        location_uuid=location_uuid,
        model_name=model_name,
        start_datetime=start_datetime,
    )

    if not dp_values:
        return _EMPTY

    now = datetime.now(UTC)
    rows = []
    for v in dp_values:
        p50_mw = v.p50_value_fraction

        # p10/p90 are optional fields - fall back to NaN when absent or zero.
        p10_fraction = v.other_statistics_fractions.get("p10")
        p90_fraction = v.other_statistics_fractions.get("p90")
        p10_mw = p10_fraction if p10_fraction is not None else float("nan")
        p90_mw = p90_fraction if p90_fraction is not None else float("nan")

        target_time = _to_aware_datetime(v.target_timestamp_utc).replace(microsecond=0)

        rows.append(
            {
                "target_time": target_time,
                "expected_power_generation_megawatts": p50_mw,
                "p10_mw": p10_mw,
                "p90_mw": p90_mw,
                "created_utc": now,
                "model_name": model_name,
            },
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Initialisation-time fetching (used by weights.py)
# ---------------------------------------------------------------------------

async def fetch_latest_nl_init_times(
    client: dp.DataPlatformDataServiceStub,
    location_uuid: str,
    model_names: list[str],
    t0: pd.Timestamp,
    max_delay: pd.Timedelta,
) -> dict[str, pd.Timestamp]:
    """Fetches all latest forecasts and extracts valid init times.

    A single DP call is made and the pure `extract_latest_init_times` function
    handles the filtering/selection logic so it remains fully unit-testable
    without a live stub.

    Args:
        client:        Authenticated Data Platform stub.
        location_uuid: Target location UUID.
        model_names:   Models we want init times for.
        t0:            Blend reference time.
        max_delay:     Forecasts older than (t0 - max_delay) are ignored.

    Returns:
        Dict mapping model name -> latest valid initialisation timestamp (UTC).
        Returns an empty dict on failure so the caller can apply fallbacks.
    """
    try:
        response = await client.get_latest_forecasts(
            dp.GetLatestForecastsRequest(
                location_uuid=location_uuid,
                energy_source=dp.EnergySource.SOLAR,
            ),
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

def _to_aware_datetime(ts: datetime | pd.Timestamp | object) -> datetime:
    """Converts a protobuf Timestamp or plain datetime to a UTC-aware datetime.

    Normalises the various forms that come back from the DP SDK.
    """
    dt = ts.ToDatetime() if hasattr(ts, "ToDatetime") else ts
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


# ---------------------------------------------------------------------------
# Forecast saving helpers
# ---------------------------------------------------------------------------

async def fetch_location_capacity_watts(
    client: dp.DataPlatformDataServiceStub,
    location_uuid: str,
) -> int:
    """Returns the effective capacity in watts for a DP location.

    Args:
        client:        Authenticated Data Platform stub.
        location_uuid: Target location UUID.

    Returns:
        Capacity in watts (integer). Returns 0 on failure, which the caller
        should treat as a save-blocking condition.
    """
    location = await client.get_location(
        dp.GetLocationRequest(
            location_uuid=location_uuid,
            energy_source=dp.EnergySource.SOLAR,
            include_geometry=False,
        ),
    )
    return location.effective_capacity_watts


def build_forecast_value_objects(
    blended_df: pd.DataFrame,
    init_time_utc: datetime,
) -> list[dp.CreateForecastRequestForecastValue]:
    """Converts a blended forecast DataFrame into DP CreateForecastRequestForecastValue objects.

    horizon_mins  = (target_time - init_time_utc) in whole minutes.
    p50_fraction  = expected_power_generation_megawatts (passed without scaling).
    p10 / p90 are placed in other_statistics_fractions under keys "p10" / "p90"
    only when their MW column is present and not NaN.

    Args:
        blended_df:     DataFrame with columns [target_time,
                        expected_power_generation_megawatts, p10_mw (opt),
                        p90_mw (opt)].
        init_time_utc:  Forecast init time (UTC); used to compute horizon_mins.

    Returns:
        List of DP forecast value objects, one per row in blended_df.
    """
    if init_time_utc.tzinfo is None:
        init_time_utc = init_time_utc.replace(tzinfo=UTC)

    has_p10 = "p10_mw" in blended_df.columns
    has_p90 = "p90_mw" in blended_df.columns

    values: list[dp.CreateForecastRequestForecastValue] = []
    for row in blended_df.itertuples(index=False):
        target_time = row.target_time
        if hasattr(target_time, "to_pydatetime"):
            target_time = target_time.to_pydatetime()
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=UTC)

        horizon_mins = int((target_time - init_time_utc).total_seconds() / 60)

        p50_fraction = float(row.expected_power_generation_megawatts)

        other_stats: dict[str, float] = {}
        if has_p10:
            p10_mw = row.p10_mw
            if not math.isnan(p10_mw):
                other_stats["p10"] = float(p10_mw)
        if has_p90:
            p90_mw = row.p90_mw
            if not math.isnan(p90_mw):
                other_stats["p90"] = float(p90_mw)

        values.append(
            dp.CreateForecastRequestForecastValue(
                horizon_mins=horizon_mins,
                p50_fraction=p50_fraction,
                other_statistics_fractions=other_stats,
            ),
        )

    return values
