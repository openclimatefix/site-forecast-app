""" Test for getting all ml models"""
import pytest

from site_forecast_app.models.pydantic_models import Model, get_all_models


def test_get_all_models():
    """Test for getting all models"""
    models = get_all_models()
    assert len(models.models) == 7


def test_site_group_uuid():
    """Test for getting all models for a given client"""
    model = get_all_models().models[0]
    model.site_group_uuid = "some-uuid"
    model.client = None

    model = Model(**model.model_dump())

def test_site_group_uuid_error_both():
    """Test for getting all models for a given client"""
    model = get_all_models().models[0]
    model.site_group_uuid = "some-uuid"

    with pytest.raises(ValueError):
        model = Model(**model.model_dump())

def test_site_group_uuid_error_neither():
    """Test for getting all models for a given client"""
    model = get_all_models().models[0]
    model.site_group_uuid = None
    model.client = None

    with pytest.raises(ValueError):
        model = Model(**model.model_dump())

