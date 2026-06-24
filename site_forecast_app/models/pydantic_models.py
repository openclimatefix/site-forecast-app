"""A pydantic model for the ML models."""


from typing import Literal

import fsspec
from pyaml_env import parse_config
from pydantic import BaseModel, Field


class Model(BaseModel):
    """One ML Model."""

    name: str = Field(..., title="Model Name", description="The name of the model")
    type: str | None = Field("pvnet", title="Model Type", description="The type of model")
    id: str = Field(
        ..., title="Model ID", description="The ID of the model, "
                                           "this what repo to load from in HF ",
    )
    version: str = Field(
        ...,
        title="Model Version",
        description="The version of the model, this is what git version to load from in HF",
    )

    asset_type: str = Field(
        "pv", title="Asset Type", description="The type of asset the model is for (pv or wind)",
    )
    adjuster_average_minutes: int = Field(
        60,
        title="Average Minutes",
        description="The number of minutes that results are average over when "
        "calculating adjuster values. "
        "For solar site with regular data, 15 should be used. "
        "For wind sites, 60 minutes should be used.",
    )
    satellite_scaling_method: Literal["constant", "minmax"] = Field(
        "constant",
        title="Satellite Scaling Method",
        description="The scaling method to use for the satellite data. ",
    )

    client: str = Field(
        "ruvnl",
        title="Client Abbreviation",
        description="The name of the client that the model is for.",
    )

    site_group_uuid: str | None = Field(
        None,
        title="Site Group UUID",
        description="The UUID of the site group that the model is for.",
    )

    observer_name: str | None = Field(
        None,
        title="Observer Name",
        description="The name of the observer to fetch from the DP generation data from",
    )

    # Summation model requires a National site, which by convention has ml_id=0 and should be
    # added to the site_group alongside the regional sites for the model
    summation_id: str | None = Field(
        None,
        title="HuggingFace repo for Summation Model",
        description="HuggingFace repo for the summation model",
    )

    summation_version: str | None = Field(
        None,
        title="Model Version of Summation Model",
        description="HuggingFace hash for the summation model",
    )

    location_type: Literal["site", "state", "nation"] = Field(
        "site",
        title="Location Type",
        description="The type of location the model is for (site, state, or nation)",
    )

    summation_location_type: Literal["site", "state", "nation"] | None = Field(
        None,
        title="Summation Location Type",
        description="The type of location for the summation outcome (site, state, or nation)",
    )

    is_critical: bool = Field(
        True,
        title="Is Critical",
        description="If this model must always be part of the critical set of models which should "
        "always be run. Non-critical models are skipped when RUN_CRITICAL_MODELS_ONLY is true.",
    )

    observer_name_adjuster: str | None = Field(
        None,
        title="Observer Name for the adjuster",
        description="The name of the observer to fetch from the DP use for the adjuster",
    )

    # curtailment options
    curtailment: bool = Field(
        False,
        title="Curtailment",
        description="Whether the model should apply curtailment to the forecasts.",
    )

    save_uncurtailed: bool = Field(
        False,
        title="Save Uncurtailed",
        description="Whether to save the uncurtailed forecasts. " \
        "This will only be used if curtailment is enabled.",
    )

    observer_name_uncurtailed_adjuster: str | None = Field(
        None,
        title="Observer Name for the uncurtailed adjuster",
        description="The name of the observer to " \
        "fetch from the DP use for the uncurtailed adjuster",
    )

    # We have v0 and v1 satellite
    # most models work off v0 satellite archive, but one works on v1
    satellite_archive_version: Literal["v0", "v1"] = Field(
        "v0",
        title="Satellite Archive",
        description="The version of the satellite archive to use.",
    )


class Models(BaseModel):
    """A group of ml models."""

    models: list[Model] = Field(
        ..., title="Models", description="A list of models to use for the forecast",
    )


def get_all_models(
    client_abbreviation: str | None = None,
    get_critical_only: bool = False,
    satellite_archive_version:str = "v0",
) -> Models:
    """Returns all the models for a given client."""
    import os

    filename = os.path.dirname(os.path.abspath(__file__)) + "/all_models.yaml"

    with fsspec.open(filename, mode="r") as stream:
        models = parse_config(data=stream)
        models = Models(**models)

    if client_abbreviation:
        models.models = [model for model in models.models if model.client == client_abbreviation]

    if get_critical_only:
        models.models = [model for model in models.models if model.is_critical]

    # filter by satellite archive version
    models.models = [model for model in models.models
                     if model.satellite_archive_version == satellite_archive_version]

    return models
