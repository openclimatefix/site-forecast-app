""" Test for getting all ml models"""

from site_forecast_app.models.pydantic_models import Model, get_all_models


def test_get_all_models():
    """Test for getting all models"""
    models = get_all_models()
    assert len(models.models) == 20

def test_get_all_models_satellite_v1():
    """Test for getting all models"""
    models = get_all_models(satellite_archive_version="v1")
    assert len(models.models) == 1

def test_site_group_uuid():
    """Test for setting site_group_uuid"""
    model = get_all_models().models[0]
    model.site_group_uuid = "some-uuid"

    model = Model(**model.model_dump())

def test_get_all_models_critical_only():
    """Test that get_critical_only filters to only critical models."""
    from site_forecast_app.models.pydantic_models import get_all_models

    all_models = get_all_models(client_abbreviation="nl")
    critical_models = get_all_models(client_abbreviation="nl", get_critical_only=True)

    assert len(critical_models.models) > 0
    assert len(critical_models.models) < len(all_models.models)
    assert all(m.is_critical for m in critical_models.models)
    assert not all(m.is_critical for m in all_models.models)
