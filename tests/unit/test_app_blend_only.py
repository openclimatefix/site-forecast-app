from unittest.mock import AsyncMock, patch

import pandas as pd

from site_forecast_app.app import app_run, run_blend_and_curtailment


def test_app_run_without_models(monkeypatch):
    """With RUN_MODELS=false, only blend and curtailment run, and no DB is needed."""
    monkeypatch.delenv("DB_URL", raising=False)
    monkeypatch.setenv("CLIENT_NAME", "nl")
    monkeypatch.setenv("COUNTRY", "nl")
    monkeypatch.setenv("RUN_MODELS", "false")

    with (
        patch("site_forecast_app.app.run_blend_and_curtailment") as mock_blend,
        patch("site_forecast_app.app.get_all_models") as mock_models,
    ):
        app_run(timestamp=None)

    mock_blend.assert_called_once()
    assert mock_blend.call_args.kwargs["client_name"] == "nl"
    mock_models.assert_not_called()


def test_curtailment_uses_model_suffix(monkeypatch):
    """Curtailment reads and writes the suffixed blend names."""
    monkeypatch.setenv("BLEND_MODEL_SUFFIX", "_pia")

    with patch("site_forecast_app.app.Curtailment") as mock_curtailment:
        mock_curtailment.return_value.apply_curtailment_on_model = AsyncMock()
        run_blend_and_curtailment(
            client_name="nl",
            country="nl",
            timestamp=pd.Timestamp("2026-01-01"),
            dp_location_map={"nl_national": "uuid"},
            run_blend=False,
        )

    calls = mock_curtailment.return_value.apply_curtailment_on_model.call_args_list
    names = [(c.kwargs["model_name"], c.kwargs["new_model_name"]) for c in calls]
    assert names == [
        ("nl_blend_uncurtailed_pia", "nl_blend_pia"),
        ("nl_blend_uncurtailed_pia_adjust", "nl_blend_pia_adjust"),
    ]
