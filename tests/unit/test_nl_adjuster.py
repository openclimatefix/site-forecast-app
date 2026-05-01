"""Unit tests for site_forecast_app.blend.adjuster.

Covers:
  - get_adjuster_model_name: suffix appended correctly in all cases.
  - get_adjuster_model_names: correct derivation for backup and all candidates.
  - Logging behaviour: debug message emitted by get_adjuster_model_names.
"""
import logging
from typing import ClassVar

import pytest

from site_forecast_app.blend.adjuster import (
    get_adjuster_model_name,
    get_adjuster_model_names,
)

# ---------------------------------------------------------------------------
# Tests: get_adjuster_model_name
# ---------------------------------------------------------------------------


class TestGetAdjusterModelName:
    """Tests for the single-model name helper."""

    def test_appends_adjust_suffix(self):
        """Standard model name gets '_adjust' appended."""
        assert get_adjuster_model_name("nl_regional_pv_ecmwf_mo_sat") == (
            "nl_regional_pv_ecmwf_mo_sat_adjust"
        )

    def test_backup_model_name(self):
        """Backup model name is also correctly suffixed."""
        assert get_adjuster_model_name("nl_regional_2h_pv_ecmwf") == (
            "nl_regional_2h_pv_ecmwf_adjust"
        )

    def test_already_suffixed_name_gets_double_suffix(self):
        """Calling twice (i.e. on an already-adjusted name) appends a second suffix.
        This documents current behaviour — callers should not pass adjusted names.
        """
        assert get_adjuster_model_name("some_model_adjust") == "some_model_adjust_adjust"

    def test_empty_string(self):
        """Empty string input returns '_adjust' (edge case)."""
        assert get_adjuster_model_name("") == "_adjust"

    def test_single_word_model(self):
        """Simple single-word model name is handled."""
        assert get_adjuster_model_name("ecmwf") == "ecmwf_adjust"

    @pytest.mark.parametrize(
        ("model_name", "expected"),
        [
            ("nl_regional_48h_pv_ecmwf", "nl_regional_48h_pv_ecmwf_adjust"),
            ("nl_regional_pv_ecmwf_mo_sat", "nl_regional_pv_ecmwf_mo_sat_adjust"),
            ("nl_regional_pv_ecmwf_sat", "nl_regional_pv_ecmwf_sat_adjust"),
            ("nl_national_pv_ecmwf_sat_small", "nl_national_pv_ecmwf_sat_small_adjust"),
        ],
    )
    def test_all_candidate_models_from_config(self, model_name, expected):
        """Every candidate model listed in config.yaml gets the correct adjuster name."""
        assert get_adjuster_model_name(model_name) == expected


# ---------------------------------------------------------------------------
# Tests: get_adjuster_model_names
# ---------------------------------------------------------------------------


class TestGetAdjusterModelNames:
    """Tests for the multi-model name helper."""

    BACKUP = "nl_regional_2h_pv_ecmwf"
    CANDIDATES: ClassVar[list[str]] = [
        "nl_regional_48h_pv_ecmwf",
        "nl_regional_pv_ecmwf_mo_sat",
        "nl_regional_pv_ecmwf_sat",
        "nl_national_pv_ecmwf_sat_small",
    ]

    def test_backup_suffixed(self):
        """Returned backup model name has '_adjust' appended."""
        adj_backup, _ = get_adjuster_model_names(self.BACKUP, self.CANDIDATES)
        assert adj_backup == f"{self.BACKUP}_adjust"

    def test_all_candidates_suffixed(self):
        """Every candidate model name has '_adjust' appended."""
        _, adj_candidates = get_adjuster_model_names(self.BACKUP, self.CANDIDATES)
        assert adj_candidates == [f"{m}_adjust" for m in self.CANDIDATES]

    def test_return_type_is_tuple(self):
        """Return value is a tuple of (str, list)."""
        result = get_adjuster_model_names(self.BACKUP, self.CANDIDATES)
        assert isinstance(result, tuple)
        assert len(result) == 2
        adj_backup, adj_candidates = result
        assert isinstance(adj_backup, str)
        assert isinstance(adj_candidates, list)

    def test_candidate_order_preserved(self):
        """Order of candidate models is preserved in the output."""
        ordered_candidates = ["model_z", "model_a", "model_m"]
        _, adj_candidates = get_adjuster_model_names(self.BACKUP, ordered_candidates)
        assert adj_candidates == ["model_z_adjust", "model_a_adjust", "model_m_adjust"]

    def test_empty_candidates_list(self):
        """Empty candidates list returns an empty adjuster candidates list."""
        adj_backup, adj_candidates = get_adjuster_model_names(self.BACKUP, [])
        assert adj_backup == f"{self.BACKUP}_adjust"
        assert adj_candidates == []

    def test_single_candidate(self):
        """Single candidate list is handled correctly."""
        _, adj_candidates = get_adjuster_model_names(self.BACKUP, ["nl_regional_pv_ecmwf_sat"])
        assert adj_candidates == ["nl_regional_pv_ecmwf_sat_adjust"]

    def test_debug_log_emitted(self, caplog):
        """A DEBUG log message is emitted containing the adjuster model names."""
        with caplog.at_level(logging.DEBUG, logger="site_forecast_app.blend.adjuster"):
            adj_backup, _adj_candidates = get_adjuster_model_names(self.BACKUP, self.CANDIDATES)

        assert any(
            adj_backup in record.message and "candidates" in record.message
            for record in caplog.records
        ), "Expected a debug log containing the adjuster backup and candidates"

    def test_debug_log_contains_all_candidate_names(self, caplog):
        """The debug log mentions every derived adjuster candidate name."""
        with caplog.at_level(logging.DEBUG, logger="site_forecast_app.blend.adjuster"):
            _, adj_candidates = get_adjuster_model_names(self.BACKUP, self.CANDIDATES)

        log_text = " ".join(r.message for r in caplog.records)
        for name in adj_candidates:
            assert name in log_text, f"Expected '{name}' to appear in debug log"

    def test_idempotent_with_single_call(self):
        """Calling get_adjuster_model_names twice with same args returns equal results."""
        result_1 = get_adjuster_model_names(self.BACKUP, self.CANDIDATES)
        result_2 = get_adjuster_model_names(self.BACKUP, self.CANDIDATES)
        assert result_1 == result_2

    @pytest.mark.parametrize(
        "backup",
        [
            "nl_regional_2h_pv_ecmwf",
            "ecmwf_backup",
            "simple",
        ],
    )
    def test_various_backup_models(self, backup):
        """Any backup model name is correctly suffixed."""
        adj_backup, _ = get_adjuster_model_names(backup, [])
        assert adj_backup == f"{backup}_adjust"
