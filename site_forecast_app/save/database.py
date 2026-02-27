"""Database operations for persisting forecasts."""

from __future__ import annotations

import json
import logging
import traceback
from typing import TYPE_CHECKING
from uuid import uuid4

from pvsite_datamodel.read.model import get_or_create_model
from pvsite_datamodel.sqlmodels import ForecastSQL, ForecastValueSQL
from sqlalchemy.orm import Session  # noqa: TC002

from site_forecast_app.adjuster import adjust_forecast_with_adjuster

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)


def insert_forecast_values(
    db_session: Session,
    forecast_meta: dict,
    forecast_values_df: pd.DataFrame,
    *,
    ml_model_name: str | None,
    ml_model_version: str | None,
) -> None:
    """Insert a forecast + its values into the DB.

    This mirrors how fixtures insert data in tests and is intentionally minimal.
    """
    model_name = ml_model_name or "default"
    model_version = ml_model_version or "0.0.0"
    ml_model = get_or_create_model(db_session, model_name, model_version)

    forecast_uuid = uuid4()
    created_utc = forecast_meta["timestamp_utc"]

    forecast = ForecastSQL(
        location_uuid=str(forecast_meta["location_uuid"]),
        timestamp_utc=forecast_meta["timestamp_utc"],
        forecast_version=str(forecast_meta["forecast_version"]),
        created_utc=created_utc,
        forecast_uuid=forecast_uuid,
    )
    db_session.add(forecast)

    values_to_add: list[ForecastValueSQL] = []
    for _, row in forecast_values_df.iterrows():
        probabilistic_values = row.get("probabilistic_values")
        if isinstance(probabilistic_values, dict):
            probabilistic_values = json.dumps(probabilistic_values)
        elif probabilistic_values is not None and not isinstance(probabilistic_values, str):
            # Best-effort: coerce unknown types to JSON.
            probabilistic_values = json.dumps(probabilistic_values)

        values_to_add.append(
            ForecastValueSQL(
                horizon_minutes=int(row["horizon_minutes"]),
                forecast_power_kw=float(row["forecast_power_kw"]),
                start_utc=row["start_utc"],
                end_utc=row["end_utc"],
                ml_model_uuid=ml_model.model_uuid,
                forecast_uuid=forecast_uuid,
                created_utc=created_utc,
                probabilistic_values=probabilistic_values,
            ),
        )

    db_session.add_all(values_to_add)
    # Ensure queries in the same session can see newly added rows.
    db_session.commit()


def write_forecast_to_db(
    db_session: Session,
    forecast_meta: dict,
    forecast_values_df: pd.DataFrame,
    *,
    write_to_db: bool,
    ml_model_name: str | None,
    ml_model_version: str | None,
) -> None:
    """Write a forecast dataframe to DB when enabled."""
    if not write_to_db:
        return

    insert_forecast_values(
        db_session,
        forecast_meta,
        forecast_values_df,
        ml_model_name=ml_model_name,
        ml_model_version=ml_model_version,
    )


def adjust_and_save_forecast(
    db_session: Session,
    forecast_meta: dict,
    forecast_values_df: pd.DataFrame,
    ml_model_name: str,
    ml_model_version: str | None,
    adjuster_average_minutes: int | None,
    write_to_db: bool,
) -> None:
    """Adjust forecast using the adjuster and save to DB."""
    log.info(f"Adjusting forecast for location_id={forecast_meta['location_uuid']}...")
    try:
        forecast_values_df_adjust = adjust_forecast_with_adjuster(
            db_session,
            forecast_meta,
            forecast_values_df,
            ml_model_name=ml_model_name,
            average_minutes=adjuster_average_minutes,
        )
        log.info(f"Adjusted forecast shape: {forecast_values_df_adjust.shape}")

        write_forecast_to_db(
            db_session,
            forecast_meta,
            forecast_values_df_adjust,
            write_to_db=write_to_db,
            ml_model_name=f"{ml_model_name}_adjust",
            ml_model_version=ml_model_version,
        )
    except Exception as e:
        log.error(f"Failed to adjust/save forecast for {ml_model_name}: {e}")
        log.error(traceback.format_exc())
