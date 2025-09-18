"""Main forecast app entrypoint."""

import datetime as dt
import logging
import os
import sys

import click
import pandas as pd
import sentry_sdk
from pvsite_datamodel import DatabaseConnection
from pvsite_datamodel.read import get_sites_by_country
from pvsite_datamodel.sqlmodels import LocationSQL
from pvsite_datamodel.write import insert_forecast_values
from sqlalchemy.orm import Session

import site_forecast_app
from site_forecast_app import __version__
from site_forecast_app.adjuster import adjust_forecast_with_adjuster
from site_forecast_app.data.generation import get_generation_data
from site_forecast_app.models import PVNetModel, get_all_models

log = logging.getLogger(__name__)
version = site_forecast_app.__version__


sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    environment=os.getenv("ENVIRONMENT", "local"),
)

sentry_sdk.set_tag("app_name", "site_forecast_app")
sentry_sdk.set_tag("version", __version__)


def get_sites(db_session: Session, country: str = "nl") -> list[LocationSQL]:
    """Gets all available sites.

    Args:
            db_session: A SQLAlchemy session
            country: The country to get sites for

    Returns:
            A list of LocationSQL objects
    """
    client = os.getenv("CLIENT_NAME", "nl")
    log.info(f"Getting sites for client: {client}")

    sites = get_sites_by_country(db_session, country=country, client_name=client)

    log.info(f"Found {len(sites)} sites for {client} in {country}")
    return sites


def get_model(
    timestamp: dt.datetime,
    generation_data: dict,
    hf_repo: str,
    hf_version: str,
    name: str,
    satellite_scaling_method: str = "constant",
) -> PVNetModel:
    """Instantiates and returns the forecast model ready for running inference.

    Args:
            timestamp: Datetime at which the forecast will be made
            generation_data: Latest historic generation data
            hf_repo: ID of the ML model used for the forecast
            hf_version: Version of the ML model used for the forecast
            name: Name of the ML model used for the forecast
            satellite_scaling_method: Method to scale the satellite data

    Returns:
            A forecasting model
    """
    # Only PVNet (and WindNet) based models are used currently
    model = PVNetModel(timestamp, generation_data, hf_repo=hf_repo, hf_version=hf_version,
                      name=name, satellite_scaling_method=satellite_scaling_method)
    return model


def run_model(model: PVNetModel, site_uuid: str, timestamp: dt.datetime) -> dict | None:
    """Runs inference on model for the given site & timestamp.

    Args:
            model: A forecasting model
            site_uuid: A specific site uuid
            timestamp: timestamp to run a forecast for

    Returns:
            A forecast or None if model inference fails
    """
    try:
        forecast = model.predict(site_uuid=site_uuid, timestamp=timestamp)
    except Exception:
        log.error(
            f"Error while running model.predict for site_uuid={site_uuid}. Skipping",
            exc_info=True,
        )
        return None

    return forecast


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
    log.info(f"Saving forecast for site_id={forecast['meta']['location_uuid']}...")

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
        log.info(f"Adjusting forecast for site_id={forecast_meta['location_uuid']}...")
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

    output = f'Forecast for site_id={forecast_meta["location_uuid"]},\
               timestamp={forecast_meta["timestamp_utc"]},\
               version={forecast_meta["forecast_version"]}:'
    log.info(output.replace("  ", ""))
    log.info(f"\n{forecast_values_df.to_string()}\n")


@click.command()
@click.option(
    "--date",
    "-d",
    "timestamp",
    type=click.DateTime(formats=["%Y-%m-%d-%H-%M"]),
    default=None,
    help='Date-time (UTC) at which we make the prediction. \
Format should be YYYY-MM-DD-HH-mm. Defaults to "now".',
)
@click.option(
    "--write-to-db",
    is_flag=True,
    default=False,
    help="Set this flag to actually write the results to the database.",
)
@click.option(
    "--log-level",
    default="info",
    help="Set the python logging log level",
    show_default=True,
)
def app(timestamp: dt.datetime | None, write_to_db: bool, log_level: str) -> None:
    """Main click function for running forecasts for sites."""
    app_run(timestamp=timestamp, write_to_db=write_to_db, log_level=log_level)


def app_run(timestamp: dt.datetime | None, write_to_db: bool = False, log_level: str = "info") \
        -> None:
    """Main function for running forecasts for sites."""
    logging.basicConfig(stream=sys.stdout, level=getattr(logging, log_level.upper()))

    log.info(f"Running India forecast app:{version}")

    if timestamp is None:
        # get the timestamp now rounded down the nearest 15 minutes
        # TODO better to have explicity UTC time here?
        timestamp = pd.Timestamp.now(tz="UTC").replace(tzinfo=None).floor("15min")
        log.info(f'Timestamp omitted - will generate forecasts for "now" ({timestamp})')
    else:
        timestamp = pd.Timestamp(timestamp).floor("15min")

    # 0. Initialise DB connection
    url = os.environ["DB_URL"]
    db_conn = DatabaseConnection(url, echo=False)
    country = os.environ.get("COUNTRY", "nl")
    log.info(f"Country {country}...")
    log.info(f"write_to_db {write_to_db}...")

    with db_conn.get_session() as session:

        # 1. Get sites
        log.info("Getting sites...")
        sites = get_sites(db_session=session, country=country)
        log.info(f"Found {len(sites)} sites")

        # 2. Load data/models
        all_model_configs = get_all_models(client_abbreviation=os.getenv("CLIENT_NAME", "nl"))
        successful_runs = 0
        runs = 0
        for model_config in all_model_configs.models:

            # reduce to only pv or wind, depending on the model
            sites_for_model = [
                site for site in sites if site.asset_type.name == model_config.asset_type
            ]

            for site in sites_for_model:
                runs += 1

                log.info(f"Reading latest historic {site} generation data...")
                generation_data = get_generation_data(session, [site], timestamp)

                log.debug(f"{generation_data['data']=}")
                log.debug(f"{generation_data['metadata']=}")

                log.info(f"Loading {site} model {model_config.name}...")
                ml_model = get_model(
                    timestamp,
                    generation_data,
                    hf_repo=model_config.id,
                    hf_version=model_config.version,
                    name=model_config.name,
                    satellite_scaling_method=model_config.satellite_scaling_method,
                )
                ml_model.location_uuid = site.location_uuid

                log.info(f"{site} model loaded")

                # 3. Run model for each site
                site_uuid = ml_model.location_uuid
                asset_type = model_config.asset_type
                log.info(f"Running {asset_type} model for site={site_uuid}...")
                forecast_values = run_model(
                    model=ml_model,
                    site_uuid=site_uuid,
                    timestamp=timestamp,
                )

                if forecast_values is None:
                    log.info(f"No forecast values for site_uuid={site_uuid}")
                else:
                    # 4. Write forecast to DB or stdout
                    log.info(f"Writing forecast for site_uuid={site_uuid}")
                    forecast = {
                        "meta": {
                            "location_uuid": site_uuid,
                            "version": version,
                            "timestamp": timestamp,
                        },
                        "values": forecast_values,
                    }
                    save_forecast(
                        session,
                        forecast=forecast,
                        write_to_db=write_to_db,
                        ml_model_name=ml_model.name,
                        ml_model_version=version,
                        adjuster_average_minutes=model_config.adjuster_average_minutes,
                    )
                    successful_runs += 1

        log.info(
            f"Completed forecasts for {successful_runs} runs for "
            f"{runs} model runs. This was for {len(sites)} sites",
        )
        if successful_runs == runs:
            log.info("All forecasts completed successfully")
        elif 0 < successful_runs < runs:
            raise Exception("Some forecasts failed")
        else:
            raise Exception("All forecasts failed")

        log.info("Forecast finished")


if __name__ == "__main__":
    app()
