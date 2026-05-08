"""Pydantic config model for the NL Blend application.

- YAML is loaded via fsspec + pyaml_env (supports !ENV tags).
- A Pydantic BaseModel validates every field.
- A single loader function is the public API.
"""
import os

import fsspec
import pandas as pd
from pyaml_env import parse_config
from pydantic import BaseModel, Field


class BlendConfig(BaseModel):
    """Configuration for a client-specific blend pipeline."""

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------
    backup_model: str = Field(
        ...,
        title="Backup Model",
        description="The absolute fallback model name; always used as the base.",
    )
    national_candidate_models: list[str] = Field(
        ...,
        title="National Candidate Models",
        description=(
            "Models evaluated as candidates for the national blend. "
            "The optimiser picks the single best one to blend against backup_model."
        ),
    )
    regional_candidate_models: list[str] = Field(
        ...,
        title="Regional Candidate Models",
        description=(
            "Models evaluated as candidates for regional blends. "
            "Typically a subset of national_candidate_models."
        ),
    )

    # ------------------------------------------------------------------
    # Blending parameters
    # ------------------------------------------------------------------
    blend_kernel: list[float] = Field(
        ...,
        title="Blend Kernel",
        description=(
            "Taper kernel weights applied at the transition zone between two models. "
            "Must be strictly between 0 and 1 and non-increasing."
        ),
    )
    min_forecast_horizon_minutes: int = Field(
        15,
        title="Minimum Forecast Horizon (minutes)",
        description="Minimum forecast horizon emitted in any blended forecast.",
    )
    t0_frequency: str = Field(
        "15min",
        title="t0 Floor Frequency",
        description="The frequency to floor the current time to for the blend reference time (t0).",
    )

    # ------------------------------------------------------------------
    # Adjuster
    # ------------------------------------------------------------------
    use_adjuster: bool = Field(
        False,
        title="Use Adjuster",
        description=(
            "Whether to run a second blend pass using adjuster model forecasts "
            "({model_name}_adjust) and save the result under {forecaster_name}_adjust. "
            "The full blend pipeline runs unchanged on the adjuster model names. "
            "Set to false to skip the adjuster blend entirely."
        ),
    )

    # ------------------------------------------------------------------
    # Infrastructure / naming
    # ------------------------------------------------------------------
    scorecard_path: str = Field(
        "data/nl_backtest_nmae_comparison.csv",
        title="Scorecard Path",
        description="Path to the MAE scorecard, relative to the blend package directory.",
    )
    forecaster_name: str = Field(
        "nl_blend",
        title="Forecaster Name",
        description="Forecaster name written to the Data Platform.",
    )
    national_location_key: str = Field(
        "nl_national",
        title="National Location Key",
        description=(
            "Location name key that identifies the national location in the "
            "DP location map."
        ),
    )
    regional_location_type: str = Field(
        "STATE",
        title="Regional Location Type",
        description=(
            "Data Platform LocationType enum name used to filter regional "
            "locations (e.g. 'STATE')."
        ),
    )

    # ------------------------------------------------------------------
    # Computed helpers (plain properties — not serialised by Pydantic)
    # ------------------------------------------------------------------
    @property
    def min_forecast_horizon(self) -> pd.Timedelta:
        """Minimum forecast horizon as a pd.Timedelta."""
        return pd.Timedelta(minutes=self.min_forecast_horizon_minutes)

    @property
    def adjuster_forecaster_name(self) -> str:
        """Forecaster name for the adjusted blend."""
        return f"{self.forecaster_name}_adjust"

    @property
    def t0(self) -> pd.Timestamp:
        """The blend reference time (t0), floored to the configured frequency."""
        return pd.Timestamp.utcnow().floor(self.t0_frequency)


class BlendAppConfig(BaseModel):
    """Global configuration for the blend application.

    Provides a generic trigger mechanism: the blend only runs if
    the environment's CLIENT_NAME matches ``client_name``.
    """

    client_name: str = Field(
        "nl",
        title="Client Name",
        description="The name of the client to process (e.g. 'nl').",
    )
    blend: BlendConfig


def load_blend_config() -> BlendAppConfig:
    """Load and validate the blend configuration from ``config.yaml``.

    The file is resolved relative to this module so it is always found
    regardless of the working directory.

    Returns:
        A validated :class:`BlendAppConfig` instance.
    """
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

    with fsspec.open(filename, mode="r") as stream:
        raw = parse_config(data=stream)
        config = BlendAppConfig(**raw)

    return config
