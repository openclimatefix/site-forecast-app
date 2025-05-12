"""Functions for getting site generation data."""
import datetime as dt
import logging

import numpy as np
import pandas as pd
import pvlib
from pvsite_datamodel import SiteSQL
from pvsite_datamodel.read import get_pv_generation_by_sites
from pvsite_datamodel.sqlmodels import SiteAssetType
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)


def get_generation_data(
    db_session: Session, sites: list[SiteSQL], timestamp: dt.datetime,
) -> dict[str, pd.DataFrame]:
    """Gets generation data values for given sites.

    Args:
            db_session: A SQLAlchemy session
            sites: A list of SiteSQL objects
            timestamp: The end time from which to retrieve data

    Returns:
            A Dict containing:
            - "data": Dataframe containing 15-minutely generation data
            - "metadata": Dataframe containing information about the site
    """
    site_uuids = [s.site_uuid for s in sites]
    start = timestamp - dt.timedelta(hours=48)
    # pad by 1 second to ensure get_pv_generation_by_sites returns correct data
    end = timestamp + dt.timedelta(seconds=1)

    log.info(f"Getting generation data for sites: {site_uuids}, from {start=} to {end=}")
    generation_data = get_pv_generation_by_sites(
        session=db_session, site_uuids=site_uuids, start_utc=start, end_utc=end,
    )
    # get the ml id, this only works for one site right now
    system_id = sites[0].ml_id
    system_id = 0  # TODO

    if len(generation_data) == 0:
        log.warning("No generation found for the specified sites/period")
        # created empty data frame with dimesion of time_utc
        generation_xr = pd.DataFrame(columns=["generation_kw"]).to_xarray()
        # add dimension of site_id
        generation_xr = generation_xr.expand_dims("site_id")
        generation_xr = generation_xr.assign_coords(
            site_id=(["site_id"], [system_id]),
        )

        # rename index to time_utc
        generation_xr = generation_xr.rename({"index": "time_utc"})

    else:
        # Convert to dataframe
        generation_df = pd.DataFrame(
            [(g.start_utc, g.generation_power_kw, system_id) for g in generation_data],
            columns=["time_utc", "power_kw", "ml_id"],
        ).pivot(index="time_utc", columns="ml_id", values="power_kw")

        log.info(generation_df)

        # Filter out any 0 values when the sun is up
        if sites[0].asset_type == SiteAssetType.pv:
            generation_df = filter_on_sun_elevation(generation_df, sites[0])

        # Ensure timestamps line up with 3min intervals
        generation_df.index = generation_df.index.round("3min")

        # Drop any duplicated timestamps
        generation_df = generation_df[~generation_df.index.duplicated()]

        # xarray (used later) expects columns with string names
        generation_df.columns = generation_df.columns.astype(str)

        # Handle any missing timestamps
        contiguous_dt_idx = pd.date_range(start=start, end=end, freq="3min")[:-1]
        generation_df = generation_df.reindex(contiguous_dt_idx, fill_value=None)

        # Interpolate NaNs
        generation_df = generation_df.interpolate(method="linear", limit_direction="both")

        # Down-sample from 3 min to 15 min intervals
        generation_df = generation_df.resample("15min").mean()

        # Add a final row for t0, and interpolate this row
        generation_df.loc[timestamp] = np.nan
        generation_df = generation_df.interpolate(method="quadratic", fill_value="extrapolate")

        # rename column to be generation_kw
        generation_df.columns = ["generation_kw"]

        # change to xarray
        generation_xr = generation_df.to_xarray()

        # add dimension of site_id
        generation_xr = generation_xr.expand_dims("site_id")

        # add coordinates of site_id
        generation_xr = generation_xr.assign_coords(
            site_id=(["site_id"], [system_id]),
        )

        # rename index to time_utc
        generation_xr = generation_xr.rename({"index": "time_utc"})

        log.info(generation_xr)

    # Site metadata dataframe
    sites_df = pd.DataFrame(
        [(system_id, s.latitude, s.longitude, s.capacity_kw, 0) for s in sites],
        columns=["system_id", "latitude", "longitude", "capacity_kwp", "site_id"],
    )

    return {"data": generation_xr, "metadata": sites_df}


def filter_on_sun_elevation(generation_df: pd.DataFrame, site: SiteSQL) -> pd.DataFrame:
    """Filter the data on sun elevation.

    If the sun is up, the generation values should be above zero
    param:
        generation_df: A dataframe containing generation data,
            with a column "power_kw", and index of datetimes
        site: A SiteSQL object

    return: dataframe with generation data
    """
    # using pvlib, calculate the sun elevations
    solpos = pvlib.solarposition.get_solarposition(
        time=generation_df.index,
        longitude=site.longitude,
        latitude=site.latitude,
        method="nrel_numpy",
    )
    elevation = solpos["elevation"].values

    # find the values that are <=0 and elevation >5
    mask = (elevation > 5) & (generation_df[generation_df.columns[0]] <= 0)

    dropping_datetimes = generation_df.index[mask]
    if len(dropping_datetimes) > 0:
        log.warning(
            f"Will be dropping {len(dropping_datetimes)} rows "
            f"from generation data: {dropping_datetimes.values} "
            f"due to sun elevation > 5 degrees and generation <= 0.0 kW. "
            f"This is likely due error in the generation data)",
        )

    generation_df = generation_df[~mask]
    return generation_df
