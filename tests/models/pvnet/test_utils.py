""" Tests for utils for pvnet"""
import os
import tempfile

import numpy as np
import xarray as xr

from site_forecast_app.models.pvnet.utils import (
    satellite_scale_minmax,
    save_batch,
    set_night_time_zeros,
)


def test_set_night_time_zeros():
    """Test for setting night time zeros"""
    # set up preds (1,5,7) {example, time, plevels}
    preds = np.random.rand(1, 5, 7)

    # check that all values are positive
    assert np.all(preds > 0)

    # set up batch, last 3 sun elevations are <0.5, so should set these to zero
    # set_night_time_zeros changes eleveations from [0,1] to [-90, 90]
    batch = {
        "solar_elevation": np.array([[0, 1, 2, 3, 4, 0.62, 0.51, 0.25, 0.26, 0.1]]),
        "t0_idx": 4,
    }

    # test function
    preds = set_night_time_zeros(batch, preds, t0_idx=4)

    # check that all values are zero
    assert np.all(preds[:, 2:, :] == 0)
    # check that all values are positive
    assert np.all(preds[:, :2, :] > 0)


def test_save_batch():
    """Test to check batches are saved"""

    # set up batch
    batch = {"key": "value"}
    i = 1
    model_name = "test_model_name"

    # create temp folder
    with tempfile.TemporaryDirectory() as temp_dir:
        save_batch(batch, i, model_name, save_batches_dir=temp_dir, site_uuid="fff-fff")

        # check that batch is saved
        assert os.path.exists(f"{temp_dir}/batch_{i}_{model_name}_fff-fff.pt")

def test_satellite_scale_minmax(small_satellite_data, # noqa: ARG001
                                ) -> None:
    """Test for scaling satellite data using min-max scaling."""

    ds = xr.open_zarr(os.getenv("SATELLITE_ZARR_PATH"))

    ds_scaled = satellite_scale_minmax(ds)

    assert ds_scaled.data.min() >= 0
    assert ds_scaled.data.max() <= 1
    assert ds_scaled.data.shape == ds.data.shape
