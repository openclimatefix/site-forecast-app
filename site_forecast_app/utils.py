"""Shared utility helpers for the site forecast app."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

MAX_DELTA_ABSOLUTE = 0.1


def add_or_convert_to_utc(timestamp: object) -> pd.Timestamp:
    """Ensure a timestamp is a timezone-aware UTC pd.Timestamp."""
    ts = pd.Timestamp(timestamp)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    return ts


def ensure_timezone_aware(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware and always in UTC."""
    if isinstance(dt, pd.Timestamp):
        return dt.tz_localize("UTC") if dt.tz is None else dt.tz_convert("UTC")
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def limit_adjuster(delta_fraction: float, value_fraction: float) -> float:
    """Limit adjuster magnitude to a fraction of forecast and absolute cap."""
    max_delta = 0.1 * value_fraction
    delta_fraction = min(max(delta_fraction, -max_delta), max_delta)
    delta_fraction = min(max(delta_fraction, -MAX_DELTA_ABSOLUTE), MAX_DELTA_ABSOLUTE)
    return delta_fraction
