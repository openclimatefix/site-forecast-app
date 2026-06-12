"""Orchestrates saving forecasts to the database and/or Data Platform."""

from __future__ import annotations

import asyncio
import logging
import os

import pandas as pd
from dp_sdk.ocf import dp
from pvsite_datamodel.sqlmodels import LocationSQL  # noqa: TC002
from sqlalchemy.orm import Session  # noqa: TC002

from site_forecast_app.models.pydantic_models import Model  # noqa: TC001
from site_forecast_app.save.data_platform import (
    save_forecast_to_dataplatform,  # noqa: F401  (re-exported via __init__)
    save_to_dataplatform,
)
from site_forecast_app.save.database import write_forecast_to_db

log = logging.getLogger(__name__)


def determine_location_type(site: LocationSQL, model_config: Model) -> dp.LocationType:
    """Determine the Data Platform LocationType based on site and model properties."""
    if site.ml_id == 0:
        loc_type = model_config.summation_location_type or "nation"
    else:
        loc_type = model_config.location_type

    if loc_type == "nation":
        return dp.LocationType.NATION
    if loc_type == "state":
        return dp.LocationType.STATE

    return dp.LocationType.SITE


def save_forecast_for_site_group(
    db_session: Session,
    forecast_values: dict,
    timestamp: pd.Timestamp,
    site_group: list,
    write_to_db: bool = False,
    model_config: Model | None = None,
    version: str | None = None,
    use_adjuster_database: bool = True,
    location_map: dict[str, str] | None = None,
    observer_name: str | None = None,
    model_name: str | None = None,
) -> None:
    """Saves forecasts for a group of sites.

    Args:
        db_session: A SQLAlchemy session
        forecast_values: dict (by site_id) of forecast values
        timestamp: The timestamp for which the forecast is valid
        site_group: The group of sites to save forecasts for
        write_to_db: If true, forecast values are written to the DB, otherwise to stdout
        model_config: Config of the ML model used for the forecast
        version: Version of the ML model used for the forecast
        use_adjuster_database: Make a new model adjusted by last 7 days of ME values.
            Also controls whether an adjusted forecast is sent to the Data Platform.
        location_map: Optional pre-fetched mapping of DP location name to UUID.
            When provided, avoids a list_locations gRPC call per site.
        observer_name: The name of the observer to use for the adjuster
        model_name: Name of the ML model used for the forecast

    """
    for site in site_group:
        # Write forecast for one site at a time
        forecast = {
            "meta": {
                "location_uuid": site.location_uuid,
                "version": version,
                "timestamp": timestamp,
                "client_location_name": site.client_location_name,
                "capacity_kw": site.capacity_kw,
                "latitude": site.latitude,
                "longitude": site.longitude,
                "location_type": determine_location_type(site, model_config),
            },
            "values": forecast_values[site.ml_id],
        }
        save_forecast(
            db_session,
            forecast=forecast,
            write_to_db=write_to_db,
            ml_model_name=model_name,
            ml_model_version=version,
            location_map=location_map,
            use_adjuster_database=use_adjuster_database,
            use_adjuster=site.ml_id == 0,
            observer_name=observer_name,
        )


def save_forecast(
    db_session: Session,
    forecast: dict,
    write_to_db: bool = False,
    ml_model_name: str | None = None,
    ml_model_version: str | None = None,
    use_adjuster_database: bool = True,
    adjuster_average_minutes: int | None = 60,
    location_map: dict[str, str] | None = None,
    observer_name: str | None = None,
    use_adjuster: bool = True,
) -> None:
    """Save a forecast for a given site & timestamp.

    Args:
        db_session: A SQLAlchemy session
        forecast: A forecast dict containing forecast meta and predicted values
        write_to_db: If true, forecast values are written to the DB, otherwise to stdout
        ml_model_name: Name of the ML model used for the forecast
        ml_model_version: Version of the ML model used for the forecast
        use_adjuster_database: Make a new model adjusted by last 7 days of ME values.
            Also controls whether an adjusted forecast is sent to the Data Platform.
        adjuster_average_minutes: Minutes to average over when calculating adjuster values
        location_map: Optional pre-fetched mapping of DP location name to UUID.
            When provided, avoids a list_locations gRPC call per site.
        observer_name: The name of the observer to use for the adjuster
        use_adjuster: Whether to save an extra model with the adjusted forecast

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
        "location_type": forecast["meta"].get("location_type"),
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
    if use_adjuster_database and ml_model_name is not None:
        from site_forecast_app.save.database import adjust_and_save_forecast
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

    # Optionally push to the Data Platform
    if os.getenv("SAVE_TO_DATA_PLATFORM", "false").lower() == "true":
        log.info("Saving to Data Platform...")
        asyncio.run(
            save_to_dataplatform(
                forecast_df=forecast_values_df,
                forecast_meta=forecast_meta,
                ml_model_name=ml_model_name,
                location_map=location_map,
                use_adjuster=use_adjuster,
                observer_name=observer_name,
            ),
        )
