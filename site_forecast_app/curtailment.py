"""Class for managing curtailment forecasts."""

import json
import logging
import os

import pandas as pd
from entsoe import EntsoePandasClient

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
        data = client.query_day_ahead_prices(country_code, start=start, end=end)

        # validate data
        if data.empty:
            log.warning("No data returned from ENTSOE API.")

        # check there are not nans
        if data.isnull().values.any():
            log.warning("Data contains NaNs.")

        # make sure timezone is utc
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
        forecast_values_df["forecast_power_kw"] = forecast_values_df.apply(
            lambda row: row["forecast_power_kw"] / 1.11 if row["curtailed"] \
                else row["forecast_power_kw"],
            axis=1,
        )

        # Lets now also apply curtailment to the probabilistic_values
        # Convert to probabilistic_values column of dict to columns
        forecast_values_df["probabilistic_values"] \
            = forecast_values_df["probabilistic_values"].apply(json.loads)
        forecast_values_df = forecast_values_df.pipe(
            lambda df: df.join(pd.json_normalize(df["probabilistic_values"])),
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
        ).drop(columns=["p10", "p90"])

        # convert back to list of dicts
        return forecast_values_df.to_dict("records")
