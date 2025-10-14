"""A pydantic model for the ML models."""


from typing import Literal

import fsspec
from pyaml_env import parse_config
from pydantic import BaseModel, Field, model_validator


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

    client: str | None = Field(
        "ruvnl",
        title="Client Abbreviation",
        description="The name of the client that the model is for." \
        "Note that either client or site_group_uuid must be provided.",
    )

    site_group_uuid: str | None = Field(
        None,
        title="Site Group UUID",
        description="The UUID of the site group that the model is for." \
        "Note that either client or site_group_uuid must be provided.",
    )

    # validate that either site_group_uuid or client is provided
    @model_validator(mode="after")
    def validate_client_or_site_group_uuid(self) -> "Model":
        """Make sure that either client or site_group_uuid is provided."""
        if not self.client and not self.site_group_uuid:
            raise ValueError("Either client or site_group_uuid must be provided.")
        if self.client and self.site_group_uuid:
            raise ValueError("Only one of client or site_group_uuid must be provided.")
        return self



class Models(BaseModel):
    """A group of ml models."""

    models: list[Model] = Field(
        ..., title="Models", description="A list of models to use for the forecast",
    )


def get_all_models(client_abbreviation: str | None = None) -> [Model]:
    """Returns all the models for a given client."""
    # load models from yaml file
    import os

    filename = os.path.dirname(os.path.abspath(__file__)) + "/all_models.yaml"

    with fsspec.open(filename, mode="r") as stream:
        models = parse_config(data=stream)
        models = Models(**models)

    if client_abbreviation:
        models.models = [model for model in models.models if model.client == client_abbreviation]

    return models
