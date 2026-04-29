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


class NlBlendConfig(BaseModel):
    """Configuration for the NL Blend blending pipeline."""

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

    # ------------------------------------------------------------------
    # Infrastructure / naming
    # ------------------------------------------------------------------
    scorecard_path: str = Field(
        "data/nl_backtest_nmae_comparison.csv",
        title="Scorecard Path",
        description="Path to the MAE scorecard, relative to the nl_blend package directory.",
    )
    forecaster_name: str = Field(
        "nl_blend",
        title="Forecaster Name",
        description="Forecaster name written to the Data Platform.",
    )

    # ------------------------------------------------------------------
    # Computed helpers (plain properties — not serialised by Pydantic)
    # ------------------------------------------------------------------
    @property
    def min_forecast_horizon(self) -> pd.Timedelta:
        """Minimum forecast horizon as a pd.Timedelta."""
        return pd.Timedelta(minutes=self.min_forecast_horizon_minutes)


class NlBlendConfigWrapper(BaseModel):
    """Wrapper for the NL Blend configuration."""

    nl_blend: NlBlendConfig


def load_nl_blend_config() -> NlBlendConfig:
    """Load and validate the NL Blend configuration from ``config.yaml``.

    The file is resolved relative to this module so it is always found
    regardless of the working directory — identical to the approach used
    by ``site_forecast_app/models/pydantic_models.py``.

    Returns:
        A validated :class:`NlBlendConfig` instance.
    """
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

    with fsspec.open(filename, mode="r") as stream:
        raw = parse_config(data=stream)
        wrapper = NlBlendConfigWrapper(**raw)

    return wrapper.nl_blend
