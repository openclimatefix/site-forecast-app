"""Functions for getting site generation data."""
import datetime as dt
import logging

import numpy as np
import pandas as pd
import pvlib
import xarray as xr
from pvsite_datamodel import LocationSQL
from pvsite_datamodel.read import get_pv_generation_by_sites
from pvsite_datamodel.sqlmodels import LocationAssetType
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)


def get_generation_data(
        db_session: Session, sites: list[LocationSQL], timestamp: pd.Timestamp,
) -> dict[str, pd.DataFrame | xr.Dataset]:
    """Load generation data from Database.

    Loads the data one site at a time and return compiled data and metadata
    for all sites.

    Args:
            db_session: A SQLAlchemy session
            sites: A list of LocationSQL objects
            timestamp: The end time from which to retrieve data

    Returns:
            A Dict containing:
            - "data": xr.Dataset containing 15-minutely generation data
            - "metadata": Dataframe containing information about the sites.
    """
    if len(sites) == 1:
        return _get_site_generation_data(db_session, sites[0], timestamp)
    else:
        log.info("Multiple sites requested. Loading data for one site at a time...")
        data_list: list[xr.Dataset] = []
        for site in sites:
            generation_ds = _get_site_generation_data(db_session, site, timestamp)
            data_list.append(generation_ds)
        log.debug("Generation data loaded for all sites. Compiling...")
        data = xr.concat(data_list, dim="location_id")
        log.debug("Data for all sites retrieved successfully.")
        data = data.sortby(data.location_id)
        return data


def _get_site_generation_data(
    db_session: Session, site: LocationSQL, timestamp: pd.Timestamp,
) -> dict[str, pd.DataFrame | xr.Dataset]:
    """Gets generation data values for a single site.

    Args:
            db_session: A SQLAlchemy session
            site: A LocationSQL object
            timestamp: The end time from which to retrieve data

    Returns:
            A Dict containing:
            - "data": xr.Dataset containing 15-minutely generation data
            - "metadata": Dataframe containing information about the site
    """
    start = timestamp - dt.timedelta(hours=48)
    # pad by 1 second to ensure get_pv_generation_by_sites returns correct data
    end = timestamp + dt.timedelta(seconds=1)

    log.info(f"Getting generation data for site {site.location_uuid}, from {start=} to {end=}")
    generation_data = get_pv_generation_by_sites(
        session=db_session, site_uuids=[site.location_uuid], start_utc=start, end_utc=end,
    )
    # get the ml id
    system_id = site.ml_id

    if len(generation_data) == 0:
        log.warning(f"No generation found for site {site.location_uuid}")
        # created empty data frame with dimesion of time_utc
        generation_xr = pd.DataFrame(columns=["generation_mw"]).to_xarray()
        # add dimension of location_id
        generation_xr = generation_xr.expand_dims("location_id")
        generation_xr = generation_xr.assign_coords(
            location_id=(["location_id"], [system_id]),
        )

        # rename index to time_utc
        generation_xr = generation_xr.rename({"index": "time_utc"})

    else:
        # Convert to dataframe
        generation_df = pd.DataFrame(
            [(g.start_utc, g.generation_power_kw/1000, system_id) for g in generation_data],
            columns=["time_utc", "generation_mw", "ml_id"],
        ).pivot(index="time_utc", columns="ml_id", values="generation_mw")

        log.info(generation_df)

        # Filter out any 0 values when the sun is up
        if site.asset_type == LocationAssetType.pv:
            generation_df = filter_on_sun_elevation(generation_df, site)

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

        # rename column to be generation_mw
        generation_df.columns = ["generation_mw"]

        # change to xarray
        generation_xr = generation_df.to_xarray()

        # add dimension of location_id
        generation_xr = generation_xr.expand_dims("location_id")

        # add coordinates of location_id
        generation_xr = generation_xr.assign_coords(
            location=(["location_id"], [system_id]),
        )

        # rename index to time_utc
        generation_xr = generation_xr.rename({"index": "time_utc"})

        log.info(generation_xr)

    # Add location metadata to generation xr

    capacity = xr.DataArray(
        site.capacity_kw / 1000,
        dims=["location_id", "time_utc"],
        coords={
            "location_id": [system_id],
            "time_utc": generation_xr.time_utc,
            "longitude": ("location_id", [site.longitude]),
            "latitude": ("location_id", [site.latitude]),
        },
        name="capacity_mwp",
    )

    generation_xr = generation_xr.assign_coords(longitude=(["location_id"], [site.longitude]),
                                latitude=(["location_id"], [site.latitude]))
    
    generation_xr["capacity_mwp"] = capacity
    
    return generation_xr


def filter_on_sun_elevation(generation_df: pd.DataFrame, site: LocationSQL) -> pd.DataFrame:
    """Filter the data on sun elevation.

    If the sun is up, the generation values should be above zero
    param:
        generation_df: A dataframe containing generation data,
            with a column "power_kw", and index of datetimes
        site: A LocationSQL object

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
