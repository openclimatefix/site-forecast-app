"""
PVNet model class
"""

import datetime as dt
import logging
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

from ocf_data_sampler.torch_datasets.datasets.site import SitesDataset
from ocf_data_sampler.config.save import save_yaml_configuration
from ocf_data_sampler.torch_datasets.datasets.site import convert_netcdf_to_numpy_sample
from ocf_data_sampler.numpy_sample.collate import stack_np_samples_into_batch
from ocf_data_sampler.torch_datasets.sample.base import (
    batch_to_tensor,
)
from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvsite_datamodel.sqlmodels import SiteAssetType
from torch.utils.data import DataLoader


from .consts import (
    nwp_ecmwf_path,
    pv_metadata_path,
    pv_netcdf_path,
    pv_path,
    root_data_path,
    satellite_path,
)
from .utils import (
    NWPProcessAndCacheConfig,
    download_satellite_data,
    populate_data_config_sources,
    process_and_cache_nwp,
    save_batch,
    set_night_time_zeros,
    worker_init_fn,
)

# Global settings for running the model

# Model will use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log = logging.getLogger(__name__)


class PVNetModel:
    """
    Instantiates a PVNet model for inference
    """

    def __init__(
        self,
        asset_type: str,
        timestamp: dt.datetime,
        generation_data: dict[str, pd.DataFrame],
        hf_repo: str,
        hf_version: str,
        name: str,
    ):
        """Initializer for the model"""

        self.asset_type = asset_type
        self.id = hf_repo
        self.version = hf_version
        self.name = name
        self.site_uuid = None
        self.t0 = timestamp
        log.info(f"Model initialised at t0={self.t0}")

        self.client = os.getenv("CLIENT_NAME", "nl")
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")

        # Setup the data, dataloader, and model
        self.generation_data = generation_data
        self._get_config()
        self._prepare_data_sources()
        self._create_dataloader()
        self.model = self._load_model()

    def predict(self, site_id: str, timestamp: dt.datetime):
        """Make a prediction for the model"""

        capacity_kw = self.generation_data["metadata"].iloc[0]["capacity_kwp"]

        normed_preds = []
        with torch.no_grad():

            batch = self.dataset[0]
            i = 0

            # for i, batch in enumerate(self.dataloader):
            log.info(f"Predicting for batch: {i}")

            log.info(batch)
            batch = convert_netcdf_to_numpy_sample(batch)
            batch = stack_np_samples_into_batch([batch])
            batch = batch_to_tensor(batch)

            # save batch
            save_batch(batch=batch, i=i, model_name=self.name, site_uuid=self.site_uuid)

            # Run batch through model
            preds = self.model(batch).detach().cpu().numpy()

            preds = set_night_time_zeros(batch, preds, t0_idx=192)

            # Store predictions
            normed_preds += [preds]

            # log max prediction
            log.info(f"Max prediction: {np.max(preds, axis=1)}")
            log.info(f"Completed batch: {i}")

        normed_preds = np.concatenate(normed_preds)
        n_times = normed_preds.shape[1]
        valid_times = pd.to_datetime(
            [self.t0 + dt.timedelta(minutes=15 * (i + 1)) for i in range(n_times)]
        )

        # index of the 50th percentile, assumed number of p values odd and in order
        middle_plevel_index = normed_preds.shape[2] // 2

        # TODO add 10th and 90th percentage

        values_df = pd.DataFrame(
            [
                {
                    "start_utc": valid_times[i],
                    "end_utc": valid_times[i] + dt.timedelta(minutes=15),
                    "forecast_power_kw": int(v * capacity_kw),
                }
                for i, v in enumerate(normed_preds[0, :, middle_plevel_index])
            ]
        )
        # remove any negative values
        values_df["forecast_power_kw"] = values_df["forecast_power_kw"].clip(lower=0.0)

        return values_df.to_dict("records")

    def _prepare_data_sources(self):
        """Pull and prepare data sources required for inference"""

        log.info("Preparing data sources")

        # Create root data directory if not exists
        try:
            os.mkdir(root_data_path)
        except FileExistsError:
            pass
        # Load remote zarr source
        use_satellite = os.getenv("USE_SATELLITE", "true").lower() == "true"
        satellite_source_file_path = os.getenv("SATELLITE_ZARR_PATH", None)

        # only load nwp that we need
        nwp_configs = []
        nwp_keys = self.config["input_data"]["nwp"].keys()
        if "ecmwf" in nwp_keys:

            nwp_configs.append(
                NWPProcessAndCacheConfig(
                    source_nwp_path=os.environ["NWP_ECMWF_ZARR_PATH"],
                    dest_nwp_path=nwp_ecmwf_path,
                    source="ecmwf",
                )
            )

        # Remove local cached zarr if already exists
        for nwp_config in nwp_configs:
            # Process/cache remote zarr locally
            process_and_cache_nwp(nwp_config)
        if use_satellite and "satellite" in self.config["input_data"].keys():
            # pass # TODO
            shutil.rmtree(satellite_path, ignore_errors=True)
            download_satellite_data(satellite_source_file_path)

        log.info("Preparing PV data sources")
        # Clear local cached wind data if already exists
        shutil.rmtree(pv_path, ignore_errors=True)
        os.mkdir(pv_path)

        # Save generation data as netcdf file
        generation_xr = self.generation_data["data"]

        forecast_timesteps = pd.date_range(
            start=self.t0 - pd.Timedelta("52H"), periods=4*24*4.5, freq="15min"
        )

        generation_xr = generation_xr.reindex(time_utc=forecast_timesteps, fill_value=0.00001)
        log.info(forecast_timesteps)

        generation_xr.to_netcdf(pv_netcdf_path, engine="h5netcdf")

        # Save metadata as csv
        self.generation_data["metadata"].to_csv(pv_metadata_path, index=False)

    def _get_config(self):
        """Setup dataloader with prepared data sources"""

        log.info("Creating configuration")

        # Pull the data config from huggingface

        data_config_filename = PVNetBaseModel.get_data_config(
            self.id, revision=self.version, token=self.hf_token
        )

        # Populate the data config with production data paths
        populated_data_config_filename = f"data/data_config.yaml"
        log.info(populated_data_config_filename)
        # if the file already exists, remove it
        if os.path.exists(populated_data_config_filename):
            os.remove(populated_data_config_filename)

        self.config = populate_data_config_sources(
            data_config_filename, populated_data_config_filename
        )
        self.populated_data_config_filename = populated_data_config_filename

    def _create_dataloader(self):

        if not os.path.exists(self.populated_data_config_filename):
            raise FileNotFoundError(
                f"Data config file not found: {self.populated_data_config_filename}"
            )

        # Location and time datapipes
        self.dataset = SitesDataset(config_filename=self.populated_data_config_filename)

        # n_workers = 0
        #
        # # Set up dataloader for parallel loading
        # dataloader_kwargs = dict(
        #     shuffle=False,
        #     batch_size=None,
        #     sampler=None,
        #     batch_sampler=None,
        #     num_workers=n_workers,
        #     pin_memory=False,
        #     drop_last=False,
        #     timeout=0,
        #     worker_init_fn=worker_init_fn,
        #     prefetch_factor=None,
        #     persistent_workers=False,
        #     collate_fn=None,
        # )
        #
        # dataloader = DataLoader(self.dataset, **dataloader_kwargs)
        #
        # return dataloader

    def _load_model(self):
        """Load model"""
        log.info(f"Loading model: {self.id} - {self.version} ({self.name})")

        return PVNetBaseModel.from_pretrained(
            model_id=self.id, revision=self.version, token=self.hf_token
        ).to(DEVICE)
