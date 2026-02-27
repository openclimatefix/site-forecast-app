"""Orchestrates saving forecasts to the database and/or Data Platform.

Sub-modules:
  - site_forecast_app.db          - Database persistence helpers
  - site_forecast_app.dataplatform - Data Platform client operations
  - site_forecast_app.utils        - Shared utility functions
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from sqlalchemy.orm import Session  # noqa: TC002

from site_forecast_app.dataplatform import (
    DataPlatformClient,
    build_dp_location_map,
    fetch_dp_location_map,
    save_forecast_to_dataplatform,
    save_to_dataplatform,
)
from site_forecast_app.db import adjust_and_save_forecast, write_forecast_to_db

log = logging.getLogger(__name__)

# Re-export for backwards compatibility so callers that import from save still work.
__all__ = [
    "DataPlatformClient",
    "build_dp_location_map",
    "fetch_dp_location_map",
    "save_forecast",
    "save_forecast_to_dataplatform",
]


def save_forecast(
    db_session: Session,
    forecast: dict,
    write_to_db: bool = False,
    ml_model_name: str | None = None,
    ml_model_version: str | None = None,
    use_adjuster: bool = True,
    adjuster_average_minutes: int | None = 60,
    location_map: dict[str, str] | None = None,
) -> None:
    """Save a forecast for a given site & timestamp.

    Args:
        db_session: A SQLAlchemy session
        forecast: A forecast dict containing forecast meta and predicted values
        write_to_db: If true, forecast values are written to the DB, otherwise to stdout
        ml_model_name: Name of the ML model used for the forecast
        ml_model_version: Version of the ML model used for the forecast
        use_adjuster: Make a new model adjusted by last 7 days of ME values
        adjuster_average_minutes: Minutes to average over when calculating adjuster values
        location_map: Optional pre-fetched mapping of DP location name to UUID.
            When provided, avoids a list_locations gRPC call per site.

    Raises:
        IOError: An error if the database save fails
    """
    log.info(f"Saving forecast for location_id={forecast['meta']['location_uuid']}...")

    forecast_meta = {
        "location_uuid": forecast["meta"]["location_uuid"],
        "timestamp_utc": forecast["meta"]["timestamp"],
        "forecast_version": forecast["meta"]["version"],
        "client_location_name": forecast["meta"].get("client_location_name"),
        "capacity_kw": forecast["meta"].get("capacity_kw"),
        "latitude": forecast["meta"].get("latitude"),
        "longitude": forecast["meta"].get("longitude"),
    }

    forecast_values_df = pd.DataFrame(forecast["values"])
    forecast_values_df["horizon_minutes"] = (
        (forecast_values_df["start_utc"] - forecast_meta["timestamp_utc"]) / pd.Timedelta("60s")
    ).astype("int")

    # Persist base forecast to DB
    write_forecast_to_db(
        db_session,
        forecast_meta,
        forecast_values_df,
        write_to_db=bool(write_to_db),
        ml_model_name=ml_model_name,
        ml_model_version=ml_model_version,
    )

    # Persist adjuster forecast to DB
    if use_adjuster and ml_model_name is not None:
        adjust_and_save_forecast(
            db_session,
            forecast_meta,
            forecast_values_df,
            ml_model_name=ml_model_name,
            ml_model_version=ml_model_version,
            adjuster_average_minutes=adjuster_average_minutes,
            write_to_db=bool(write_to_db),
        )

    output = (
        f"Forecast for location_id={forecast_meta['location_uuid']},"
        f"timestamp={forecast_meta['timestamp_utc']},"
        f"version={forecast_meta['forecast_version']}:"
    )
    log.info(output)
    log.info(f"\n{forecast_values_df.to_string()}\n")

    # Optionally push to the Data Platform
    if os.getenv("SAVE_TO_DATA_PLATFORM", "false").lower() == "true":
        log.info("Saving to Data Platform...")
        save_to_dataplatform(
            forecast_df=forecast_values_df,
            forecast_meta=forecast_meta,
            ml_model_name=ml_model_name,
            use_adjuster=use_adjuster,
            location_map=location_map,
        )
