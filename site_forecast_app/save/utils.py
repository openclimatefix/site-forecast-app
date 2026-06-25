"""Shared utility helpers used across the save subpackage."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pandas as pd
from dp_sdk.ocf import dp
from pvsite_datamodel.sqlmodels import LocationSQL  # noqa: TC002

adjuster_limit_fraction = os.environ.get("ADJUSTER_LIMIT_FRACTION", 0.1)
adjuster_limit_mw = os.environ.get("ADJUSTER_LIMIT_MW", 1000.0)

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


def limit_adjuster(delta_fraction: float, value_fraction: float, capacity_mw: float) -> float:
    """Limit the adjuster to 10% of forecast and max 1000 MW."""
    # limit adjusted fractions to 10% of fv.p50_fraction
    max_delta = adjuster_limit_fraction * value_fraction
    if delta_fraction > max_delta:
        delta_fraction = max_delta
    elif delta_fraction < -max_delta:
        delta_fraction = -max_delta

    # limit adjust to 1000 MW
    max_delta_absolute = adjuster_limit_mw / capacity_mw
    if delta_fraction > max_delta_absolute:
        delta_fraction = max_delta_absolute
    elif delta_fraction < -max_delta_absolute:
        delta_fraction = -max_delta_absolute

    return delta_fraction


def determine_energy_source(site: LocationSQL) -> dp.EnergySource:
    """Determine the Data Platform EnergySource based on site asset type."""
    asset_type = site.asset_type.name if hasattr(site.asset_type, "name") else str(site.asset_type)
    if asset_type.lower() == "wind":
        return dp.EnergySource.WIND
    return dp.EnergySource.SOLAR
