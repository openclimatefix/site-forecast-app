"""Adjuster helpers for the NL blend pipeline.

The adjuster blend is identical to the main blend, except it operates on
adjuster model forecasts ({model_name}_adjust) instead of the main model
forecasts. No subtraction or custom calculation is needed - the full blend
pipeline (weight calculation, blending, saving) runs unchanged on the
adjuster model names.

This module provides a single helper to derive adjuster model names from
the standard candidate and backup model names.
"""
import logging

logger = logging.getLogger(__name__)


def get_adjuster_model_name(model_name: str) -> str:
    """Returns the adjuster forecaster name for a given model name.

    Args:
        model_name: Standard model name (e.g. "nl_regional_pv_ecmwf_mo_sat").

    Returns:
        Adjuster model name (e.g. "nl_regional_pv_ecmwf_mo_sat_adjust").
    """
    return f"{model_name}_adjust"


def get_adjuster_model_names(
    backup_model: str,
    candidate_models: list[str],
) -> tuple[str, list[str]]:
    """Derives adjuster model names for the backup and all candidates.

    Args:
        backup_model:      Standard backup model name.
        candidate_models:  Standard candidate model names.

    Returns:
        Tuple of (adjuster_backup_model, adjuster_candidate_models).
    """
    adjuster_backup = get_adjuster_model_name(backup_model)
    adjuster_candidates = [get_adjuster_model_name(m) for m in candidate_models]

    logger.debug(
        f"Adjuster models derived - backup: '{adjuster_backup}', "
        f"candidates: {adjuster_candidates}",
    )

    return adjuster_backup, adjuster_candidates
