"""Functions for working with site generation data."""
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
        metadata_list: list[pd.DataFrame] = []
        data_list: list[xr.Dataset] = []
        for site in sites:
            site_dict = _get_site_generation_data(db_session, site, timestamp)
            metadata_list.append(site_dict["metadata"])
            data_list.append(site_dict["data"])
        log.debug("Generation data loaded for all sites. Compiling...")
        metadata = pd.concat(metadata_list)
        data = xr.concat(data_list, dim="location_id")
        log.debug("Data for all sites retrieved successfully.")
        metadata = metadata.sort_values(by="system_id")
        data = data.sortby(data.location_id)
        return {"data": data, "metadata": metadata.set_index("system_id")}


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
        generation_xr = pd.DataFrame(columns=["generation_kw"]).to_xarray()
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
            [(g.start_utc, g.generation_power_kw, system_id) for g in generation_data],
            columns=["time_utc", "power_kw", "ml_id"],
        ).pivot(index="time_utc", columns="ml_id", values="power_kw")

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

        # rename column to be generation_kw
        generation_df.columns = ["generation_kw"]

        # change to xarray
        generation_xr = generation_df.to_xarray()

        # add dimension of location_id
        generation_xr = generation_xr.expand_dims("location_id")

        # add coordinates of location_id
        generation_xr = generation_xr.assign_coords(
            location_id=(["location_id"], [system_id]),
        )

        # rename index to time_utc
        generation_xr = generation_xr.rename({"index": "time_utc"})

        log.info(generation_xr)

    # Site metadata dataframe
    site_df = pd.DataFrame(
        [(system_id, site.latitude, site.longitude, site.capacity_kw, system_id)],
        columns=["system_id", "latitude", "longitude", "capacity_kwp", "location_id"],
    )

    return {"data": generation_xr, "metadata": site_df}


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

def format_generation_data(generation_xr: xr.Dataset,
                            metadata_df: pd.DataFrame) -> xr.Dataset:
    """Format generation data to schema pvnet expects.

    Args:
        generation_xr: xr.Dataset containing generation data
        metadata_df: pd.DataFrame containing location_id, capacity_kwp, latitude, longitude

    Returns:
        Generation data schema formatted to is:
        Dimensions: (time_utc, location_id)
        Data Variables:
            generation_mw (time_utc, location_id): float32 representing the generation in MW
            capacity_mwp (time_utc, location_id): float32 representing the capacity in MW peak
        Coordinates:
            time_utc (time_utc): datetime64[ns] representing the time in utc
            location_id (location_id): int representing the location IDs
            longitute (location_id): float representing the longitudes of the locations
            latitude (location_id): float representing the latitudes of the locations
    """
    # Clean and prepare metadata
    metadata = (
        metadata_df
        .assign(
            capacity_mwp=lambda df: df["capacity_kwp"] / 1000,
        )
        .drop(columns=["capacity_kwp"])
        .set_index("location_id")
    )

    # Prepare generation data, convert to MW
    generation = (
        generation_xr
        .rename({"generation_kw": "generation_mw"})
        / 1000
    )

    # Capacity: align to generation_xr
    capacity = metadata["capacity_mwp"].to_xarray().broadcast_like(generation)

    # Attach capacity & coordinates
    generation = generation.assign(capacity_mwp=capacity).assign_coords(
        latitude=("location_id", metadata.loc[generation.location_id.values, "latitude"]),
        longitude=("location_id", metadata.loc[generation.location_id.values, "longitude"]),
    )

    return generation
