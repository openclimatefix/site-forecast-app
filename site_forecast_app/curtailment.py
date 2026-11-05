"""Class for managing curtailment forecasts."""

import json
import logging
import os
import time
from datetime import UTC, datetime
from venv import logger

import pandas as pd
from entsoe import EntsoePandasClient
from ocf import dp

from site_forecast_app.blend.app import save_forecasts
from site_forecast_app.blend.data_platform import get_all_forecast_values_as_dataframe
from site_forecast_app.data_platform import get_dataplatform_client

log = logging.getLogger(__name__)

api_key = os.getenv("APIKEY_ENTSOE")


class Curtailment:
    """Curtailment class for managing curtailment forecasts."""
    def __init__(self, now: pd.Timestamp) -> None:
        """Initialize the Curtailment class by getting NL DA prices."""
        self.now = now
        self.get_prices()

    def get_prices(self) -> None:
        """Fetch the latest market prices for curtailment."""
        client = EntsoePandasClient(api_key=api_key)
        country_code = "NL"  # Netherlands
        start = self.now
        # make sure start has a timezone
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        end = start + pd.Timedelta(days=2)  # fetch a 2 days of data

        # methods that return Pandas Series
        log.info(f"Fetching day-ahead prices from ENTSOE API for {country_code} \
                 from {start} to {end}")

        max_retries = 3

        for attempt in range(max_retries):
            log.info(f"Attempt {attempt + 1}/{max_retries} to fetch data from ENTSOE API.")
            try:
                data = client.query_day_ahead_prices(
                    country_code,
                    start=start,
                    end=end,
                )
                break

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")

                if attempt == max_retries - 1:
                    # ENTSOE being unavailable should not fail the whole forecast run,
                    # we carry on with no prices, which means no curtailment is applied.
                    log.error(
                        f"Error fetching data: {e}. "
                        f"Carrying on without prices, no curtailment will be applied.",
                    )
                    data = pd.DataFrame(columns=["NL_day_ahead_prices_euros_per_mwh"])

                time.sleep(2**attempt)

        # validate data
        if data.empty:
            log.warning("No data returned from ENTSOE API.")

        # check there are not nans
        if data.isnull().values.any():
            log.warning("Data contains NaNs.")

        # make sure timezone is utc
        if not data.empty:
            data.index = data.index.tz_convert("UTC")

        # convert to dataframe with columns ['target_datetime_utc', 'price']
        data = data.reset_index()
        data.columns = ["target_datetime_utc", "NL_day_ahead_prices_euros_per_mwh"]
        data["target_datetime_utc"] = pd.to_datetime(data["target_datetime_utc"])

        self.prices_df = data

    def apply_curtailment(self, forecast_values: dict | None) -> dict | None:
        """Apply curtailment to the forecast values.

        This is v1 curtailment for NL.
        If the prices are negative, then we reduce the values by 9.1%.
        (This is done by divided by 1.11 - this is what the analysis showed)
        """
        # for forecast values is None, then also return None
        if forecast_values is None:
            return forecast_values

        if len(forecast_values) == 0:
            return forecast_values

        # make into dataframe and merge with prices
        forecast_values_df = pd.DataFrame(forecast_values)
        forecast_values_df = forecast_values_df.merge(
            self.prices_df, on="target_datetime_utc", how="left",
        )

        # apply curtailment
        # method as of 2026-05-15 from NedNL analysis
        forecast_values_df["curtailed"] = forecast_values_df[
            "NL_day_ahead_prices_euros_per_mwh"
        ].apply(lambda x: x <= 0)
        forecast_values_df["forecast_power"] = forecast_values_df.apply(
            lambda row: row["forecast_power"] / 1.11 if row["curtailed"] \
                else row["forecast_power"],
            axis=1,
        )

        # apply curtailament on p10 and p90
        for plevel in ["p10", "p90"]:
            forecast_values_df[plevel] = forecast_values_df.apply(
                lambda x: x[plevel] / 1.11 if x["curtailed"] else x[plevel], # noqa B023
                axis=1,
            )

        # convert back to json
        forecast_values_df["probabilistic_values"] = forecast_values_df[["p10", "p90"]].apply(
            lambda row: json.dumps(row.to_dict()),
            axis=1,
        )
        forecast_values_df = forecast_values_df.drop(columns=["p10", "p90"])

        # convert back to list of dicts
        return forecast_values_df.to_dict("records")

    async def apply_curtailment_on_model(
            self,
            model_name: str,
            location_map: dict[str, str],
            new_model_name:str,
            t0: datetime,
            dp_client: dp.DataPlatformDataServiceStub | None = None) -> None:
        """Run the curtailment process for a specific model."""
        async with get_dataplatform_client(dp_client) as dp_client:


            # load the forecast values
            values = await get_all_forecast_values_as_dataframe(
                client=dp_client,
                location_uuid=location_map["nl_national"],
                model_name=model_name,
                start_datetime=t0,
            )

            # Format the data, ready for curtailment
            values["target_datetime_utc"] = values["target_time"]
            values["forecast_power"] = values["expected_power_generation"]

            # Apply curtailment
            self.curtailment_results = pd.DataFrame(self.apply_curtailment(forecast_values=values))
            self.curtailment_results["expected_power_generation"] \
                = self.curtailment_results["forecast_power"]

            # Save the results
            await save_forecasts(
                client=dp_client,
                blended_df=self.curtailment_results,
                t0=t0.replace(tzinfo=UTC),
                location_uuid=location_map["nl_national"],
                location_key=new_model_name,
                forecaster_name=new_model_name)

