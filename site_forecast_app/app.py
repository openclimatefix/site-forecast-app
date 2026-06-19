"""Main forecast app entrypoint."""

import asyncio
import datetime as dt
import logging
import os
import shutil
import sys
from functools import partial

import click
import pandas as pd
import sentry_sdk
from dp_sdk.ocf import dp
from pvsite_datamodel import DatabaseConnection
from pvsite_datamodel.read import get_sites_by_country
from pvsite_datamodel.sqlmodels import LocationGroupSQL, LocationSQL
from sqlalchemy.orm import Session

import site_forecast_app
from site_forecast_app import __version__
from site_forecast_app.blend.app import run_blend_app
from site_forecast_app.blend.config import load_blend_config
from site_forecast_app.curtailment import Curtailment
from site_forecast_app.data.generation import get_generation_data
from site_forecast_app.data.satellite import (
    check_model_satellite_inputs_available,
    get_valid_satellite_times,
)
from site_forecast_app.models import PVNetModel, get_all_models
from site_forecast_app.models.pvnet.consts import root_data_path, satellite_path
from site_forecast_app.models.pydantic_models import Model
from site_forecast_app.save import (
    build_dp_location_map,
    save_forecast_for_site_group,
)

log = logging.getLogger(__name__)
version = site_forecast_app.__version__


sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    environment=os.getenv("ENVIRONMENT", "local"),
)

sentry_sdk.set_tag("app_name", "site_forecast_app")
sentry_sdk.set_tag("version", __version__)


def get_sites(
    db_session: Session,
    model_config: Model | None = None,
    country: str = "nl",
    client_name: str = "nl",
) -> list[LocationSQL]:
    """Gets all available sites.

    Args:
            db_session: A SQLAlchemy session
            model_config: The model configuration to use
            country: The country to get sites for
            client_name: The client name to get sites for

    Returns:
            A list of LocationSQL objects
    """
    # if model.site_group_uuid is provided, filter sites by that too
    if model_config is not None and model_config.site_group_uuid is not None:
        log.info(f"Getting sites for site_group_uuid: {model_config.site_group_uuid}")
        site_group = (
            db_session.query(LocationGroupSQL)
            .filter(
                LocationGroupSQL.location_group_uuid == model_config.site_group_uuid,
            )
            .one()
        )  # check it exists
        # note if this uuid doesn't exist, an exception will be raised
        sites = site_group.locations
    else:
        # get sites and filter by client
        log.info(f"Getting sites for client: {client_name}")
        sites = get_sites_by_country(
            db_session,
            country=country,
            client_name=client_name,
        )

    log.info(f"Found {len(sites)} sites in {country}")
    return sites


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


def run_model(model: PVNetModel, timestamp: pd.Timestamp) -> dict | None:
    """Runs inference on model for the given site & timestamp.

    Args:
            model: A forecasting model
            timestamp: timestamp to run a forecast for

    Returns:
            A forecast or None if model inference fails
    """
    try:
        forecast = model.predict(timestamp=timestamp)
    except Exception:
        log.error(
            f"Error while running model.predict for site_uuid={model.site_uuid}. Skipping",
            exc_info=True,
        )
        return None

    return forecast


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
@click.option(
    "--use-adjuster-database",
    is_flag=True,
    default=True,
    help="Set this flag to use the adjuster.",
    envvar="USE_ADJUSTER_DATABASE",
)
def app(timestamp: dt.datetime | None,
        write_to_db: bool,
        log_level: str,
        use_adjuster_database: bool) -> None:
    """Main click function for running forecasts for sites."""
    app_run(timestamp=timestamp,
            write_to_db=write_to_db,
            log_level=log_level,
            use_adjuster_database=use_adjuster_database)



def app_run(
    timestamp: dt.datetime | None,
    write_to_db: bool = False,
    log_level: str = "info",
    use_adjuster_database: bool = True,
) -> None:
    """Main function for running forecasts for sites."""
    logging.basicConfig(stream=sys.stdout, level=getattr(logging, log_level.upper()))

    log.info(f"Running site forecast app:{version}")
    client_name = os.getenv("CLIENT_NAME", "nl")
    save_to_data_platform = os.getenv("SAVE_TO_DATA_PLATFORM", "false").lower() == "true"

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
    country = os.environ.get("COUNTRY", None)
    if country is None:
        raise ValueError("COUNTRY environment variable must be set")

    log.info(f"Country {country}...")
    log.info(f"write_to_db {write_to_db}...")

    with db_conn.get_session() as session:
        # 1. Load data/models
        run_critical_only = os.getenv("RUN_CRITICAL_MODELS_ONLY", "false").lower() == "true"
        all_model_configs = get_all_models(
            client_abbreviation=os.getenv("CLIENT_NAME", "nl"),
            get_critical_only=run_critical_only,
        )
        successful_runs = 0
        runs = 0
        failed_runs = []

        # Pre-fetch the DP location map once so _resolve_target_uuid doesn't call
        # list_locations on every individual forecast save.
        dp_location_map: dict[str, str] | None = None
        if save_to_data_platform:
            try:
                dp_location_map = asyncio.run(build_dp_location_map())
                log.info(f"Pre-fetched {len(dp_location_map)} DP site locations.")
            except Exception:
                log.warning(
                    "Failed to pre-fetch DP location map — will fall back to per-site lookup.",
                    exc_info=True,
                )

        if any(m.curtailment for m in all_model_configs.models):
            log.info("Curtailment is enabled for at least one model.")
            curtailment = Curtailment(timestamp)

        for model_config in all_model_configs.models:
            # 2. Get sites
            log.info("Getting sites...")
            sites = get_sites(db_session=session,
                              country=country,
                              model_config=model_config,
                              client_name=client_name)
            log.info(f"Found {len(sites)} sites")

            # reduce to only pv or wind, depending on the model
            sites_for_model = [
                site for site in sites if site.asset_type.name == model_config.asset_type
            ]

            if model_config.summation_version:
                # Summation model provided, run the model concurrently on all sites
                site_groups_for_running_model = [sites_for_model]
                log.info("Running the model concurrently")
            else:
                site_groups_for_running_model = [[s] for s in sites_for_model]
                log.info("Running the model for each site one by one")


            for site_group in site_groups_for_running_model:
                runs += 1

                log.info("Reading latest historic generation data for all sites...")
                generation_data = get_generation_data(
                    session, site_group, timestamp, observer_name=model_config.observer_name,
                )

                log.info(f"Loading concurrent model {model_config.name}...")
                ml_model = PVNetModel(
                    timestamp,
                    generation_data,
                    hf_repo=model_config.id,
                    hf_version=model_config.version,
                    name=model_config.name,
                    site_uuid=str(model_config.site_group_uuid),
                    satellite_scaling_method=model_config.satellite_scaling_method,
                    summation_version=model_config.summation_version,
                    summation_repo = model_config.summation_id,
                    asset_type=model_config.asset_type,
                )

                log.info(f"{ml_model.site_uuid} model loaded")

                # Validate satellite data is sufficient for this model's requirements.
                # If not, skip to the next model rather than producing odd outputs.
                if not check_model_satellite_inputs_available(
                    data_config_filename=ml_model.populated_data_config_filename,
                    t0=timestamp,
                    sat_datetimes=get_valid_satellite_times(satellite_path),
                    country=country,
                ):
                    log.warning(
                        f"Skipping model {model_config.name} for site_group_uuid="
                        f"{model_config.site_group_uuid}: satellite data is too delayed.",
                    )
                    failed_runs.append(f"model={model_config.name}, "
                                       f"site_group_uuid={model_config.site_group_uuid}")
                    continue

                # 3. Run model for all sites
                asset_type = model_config.asset_type
                log.info(
                    f"Running {asset_type} model for site group={model_config.site_group_uuid}...",
                )
                forecast_values = run_model(
                    model=ml_model,
                    timestamp=timestamp,
                )

                # when not running the summation model, we get a list back,
                # lets make sure its a dictionary like we get back if its a summation model
                if isinstance(forecast_values, list):
                    forecast_values = {site_group[0].ml_id: forecast_values}

                if forecast_values is None:
                    log.info(
                        f"No forecast values for site_group_uuid={model_config.site_group_uuid}",
                    )
                    failed_runs.append(f"model={model_config.name}, "
                                       f"site_group_uuid={model_config.site_group_uuid}")
                else:

                    # 4. Write forecast to DB or stdout
                    log.info(
                        f"Writing forecast for site_group_uuid={model_config.site_group_uuid}",
                    )

                    # make a partial function for saving forecast, still need to set
                    # forecast_values and model_name
                    save_forecast_for_site_group_partial = partial(save_forecast_for_site_group,
                        db_session=session,
                        timestamp=timestamp,
                        site_group=site_group,
                        write_to_db=write_to_db,
                        model_config=model_config,
                        version=version,
                        use_adjuster_database=use_adjuster_database,
                        location_map=dp_location_map,
                    )

                    if not model_config.curtailment:
                        # if there is no curtailment, we can save the forecast normally
                        save_forecast_for_site_group_partial(
                            forecast_values=forecast_values,
                            model_name=model_config.name,
                            observer_name=model_config.observer_name_adjuster,
                        )

                    elif model_config.save_uncurtailed & model_config.curtailment:
                        # if there is curtailment, and we want to save the uncurtailed values
                        log.info("Saving uncurtailed forecast values...")

                        save_forecast_for_site_group_partial(
                            forecast_values=forecast_values,
                            model_name=model_config.name + "_uncurtailed",
                            observer_name=model_config.observer_name_uncurtailed_adjuster,
                        )

                    if model_config.curtailment:
                        # now saving the curtailed forecast values
                        log.info("Applying curtailment to forecast values...")
                        forecast_values = {k: curtailment.apply_curtailment(v) \
                                        for k, v in forecast_values.items()}


                        save_forecast_for_site_group_partial(
                            forecast_values=forecast_values,
                            model_name=model_config.name,
                            observer_name=model_config.observer_name_adjuster,
                        )
                    successful_runs += 1

        log.info(
            f"Completed forecasts for {successful_runs} runs for "
            f"{runs} model runs.",
        )
        if save_to_data_platform and client_name == "nl":
            # Run the generic blend pipeline automatically after site forecasts complete.
            # Blend writes to the Data Platform, so only run when DP saves are enabled.
            # The config is loaded here (where country context lives) and passed into
            # run_blend_app so the blend app remains country-agnostic.
            log.info("Checking for blend pipeline configuration...")
            app_config = load_blend_config(client_name=client_name)
            if app_config and app_config.client_name == client_name:
                log.info(f"Starting {app_config.client_name} blend pipeline...")
                asyncio.run(run_blend_app(config=app_config))
                log.info(f"{app_config.client_name} blend pipeline completed.")
        if successful_runs == runs:
            log.info("All forecasts completed successfully")
        elif 0 < successful_runs < runs:
            raise Exception(f"Some forecasts failed {failed_runs}")
        else:
            raise Exception("All forecasts failed")


        # lets remove any temporary files created during forecasting
        if os.path.exists(root_data_path):
            log.info("Removing temporary files...")
            shutil.rmtree(root_data_path)

        log.info("Forecast finished")


if __name__ == "__main__":
    app()
