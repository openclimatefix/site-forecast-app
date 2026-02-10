"""Functions for saving forecasts."""

import logging

import pandas as pd
from pvsite_datamodel.write import insert_forecast_values
from sqlalchemy.orm import Session

from site_forecast_app.adjuster import adjust_forecast_with_adjuster

log = logging.getLogger(__name__)

def save_forecast(
    db_session: Session,
    forecast: dict,
    write_to_db: bool,
    ml_model_name: str | None = None,
    ml_model_version: str | None = None,
    use_adjuster: bool = True,
    adjuster_average_minutes: int | None = 60,
) -> None:
    """Saves a forecast for a given site & timestamp.

    Args:
            db_session: A SQLAlchemy session
            forecast: a forecast dict containing forecast meta and predicted values
            write_to_db: If true, forecast values are written to db, otherwise to stdout
            ml_model_name: Name of the ML model used for the forecast
            ml_model_version: Version of the ML model used for the forecast
            use_adjuster: Make new model, adjusted by last 7 days of ME values
            adjuster_average_minutes: The number of minutes that results are average over
                when calculating adjuster values

    Raises:
            IOError: An error if database save fails
    """
    log.info(f"Saving forecast for location_id={forecast['meta']['location_uuid']}...")

    forecast_meta = {
        "location_uuid": forecast["meta"]["location_uuid"],
        "timestamp_utc": forecast["meta"]["timestamp"],
        "forecast_version": forecast["meta"]["version"],
    }
    forecast_values_df = pd.DataFrame(forecast["values"])
    forecast_values_df["horizon_minutes"] = (
        (forecast_values_df["start_utc"] - forecast_meta["timestamp_utc"]) / pd.Timedelta("60s")
    ).astype("int")

    if write_to_db:
        insert_forecast_values(
            db_session,
            forecast_meta,
            forecast_values_df,
            ml_model_name=ml_model_name,
            ml_model_version=ml_model_version,
        )

    if use_adjuster:
        log.info(f"Adjusting forecast for location_id={forecast_meta['location_uuid']}...")
        forecast_values_df_adjust = adjust_forecast_with_adjuster(
            db_session,
            forecast_meta,
            forecast_values_df,
            ml_model_name=ml_model_name,
            average_minutes=adjuster_average_minutes,
        )

        if write_to_db:
            insert_forecast_values(
                db_session,
                forecast_meta,
                forecast_values_df_adjust,
                ml_model_name=f"{ml_model_name}_adjust",
                ml_model_version=ml_model_version,
            )

    output = f"Forecast for location_id={forecast_meta['location_uuid']},\
               timestamp={forecast_meta['timestamp_utc']},\
               version={forecast_meta['forecast_version']}:"
    log.info(output.replace("  ", ""))
    log.info(f"\n{forecast_values_df.to_string()}\n")
