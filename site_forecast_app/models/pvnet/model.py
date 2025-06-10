"""PVNet model class."""

import contextlib
import datetime as dt
import json
import logging
import os
import shutil

import numpy as np
import pandas as pd
import torch
from ocf_data_sampler.numpy_sample.collate import stack_np_samples_into_batch
from ocf_data_sampler.torch_datasets.datasets.site import (
    SitesDataset,
    convert_netcdf_to_numpy_sample,
)
from ocf_data_sampler.torch_datasets.sample.base import (
    batch_to_tensor,
)
from pvnet.models.base_model import BaseModel as PVNetBaseModel

from .consts import (
    nwp_ecmwf_path,
    nwp_mo_global_path,
    root_data_path,
    site_metadata_path,
    site_netcdf_path,
    site_path,
)
from .utils import (
    NWPProcessAndCacheConfig,
    download_satellite_data,
    populate_data_config_sources,
    process_and_cache_nwp,
    save_batch,
    set_night_time_zeros,
)

# Global settings for running the model

# Model will use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log = logging.getLogger(__name__)


class PVNetModel:
    """Instantiates a PVNet model for inference."""

    def __init__(
        self,
        timestamp: dt.datetime,
        generation_data: dict[str, pd.DataFrame],
        hf_repo: str,
        hf_version: str,
        name: str,
        asset_type: str = "pv",
    ) -> None:
        """Initializer for the model."""
        self.asset_type = asset_type
        self.id = hf_repo
        self.version = hf_version
        self.name = name
        self.site_uuid = None
        self.t0 = timestamp
        log.info(f"Model initialised at t0={self.t0}")

        self.client = os.getenv("CLIENT_NAME", "nl")
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN", None)

        if self.hf_token is not None:
            log.info("We are using a Hugging Face token for authentication.")
        else:
            log.warning("No Hugging Face token provided, using anonymous access.")

        # Setup the data, dataloader, and model
        self.generation_data = generation_data
        self._get_config()
        self._prepare_data_sources()
        self._create_dataloader()
        self.model = self._load_model()

    def predict(self, site_uuid: str, timestamp: dt.datetime) -> dict:
        """Make a prediction for the model."""
        capacity_kw = self.generation_data["metadata"].iloc[0]["capacity_kwp"]

        normed_preds = []
        with torch.no_grad():

            # note this only running ones site, and the latest timestamp
            samples = self.dataset.valid_t0_and_site_ids
            sample_t0 = samples.iloc[-1].t0
            sample_site_id = samples.iloc[-1].site_id

            batch = self.dataset.get_sample(t0=sample_t0, site_id=sample_site_id)
            i = 0

            if site_uuid != sample_site_id:
                log.warning(
                    f"Site id different from the one in the batch: {site_uuid} != {sample_site_id}",
                )

            if timestamp != sample_t0:
                log.warning(
                    f"Timestamp different from the one in the batch: {timestamp} != {sample_t0}",
                )

            # for i, batch in enumerate(self.dataloader):
            log.info(f"Predicting for batch: {i}, for {sample_t0=}, {sample_site_id=}")

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
            [sample_t0 + dt.timedelta(minutes=15 * i) for i in range(n_times)],
        )

        # index of the 50th percentile, assumed number of p values odd and in order
        middle_plevel_index = normed_preds.shape[2] // 2

        values_df = pd.DataFrame(
            [
                {
                    "start_utc": valid_times[i],
                    "end_utc": valid_times[i] + dt.timedelta(minutes=15),
                    "forecast_power_kw": int(v * capacity_kw),
                }
                for i, v in enumerate(normed_preds[0, :, middle_plevel_index])
            ],
        )
        # remove any negative values
        values_df["forecast_power_kw"] = values_df["forecast_power_kw"].clip(lower=0.0)

        values_df = self.add_probabilistic_values(capacity_kw, normed_preds, values_df)

        return values_df.to_dict("records")

    def add_probabilistic_values(
        self, capacity_kw: int, normed_preds: np.array, values_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add probabilistic values to the dataframe."""
        if hasattr(self.model, "output_quantiles") and self.model.output_quantiles is not None:
            output_quantiles = self.model.output_quantiles
            if 0.1 in output_quantiles and 0.9 in output_quantiles:
                idx_10 = output_quantiles.index(0.1)
                idx_90 = output_quantiles.index(0.9)
            else:
                log.warning(
                    f"Model output quantiles ({output_quantiles}) do not contain ",
                    "10th and 90th percentiles, using first and last indices.",
                )
                idx_10 = 0
                idx_90 = -1
        else:
            log.warning(
                "Model does not contain output quantiles, ",
                "going to try with using second and penultimate indices.",
            )
            idx_10 = 1
            idx_90 = 5

        # add 10th and 90th percentage
        values_df["p10"] = normed_preds[0, :, idx_10] * capacity_kw
        values_df["p90"] = normed_preds[0, :, idx_90] * capacity_kw
        # change to intergers
        values_df["p10"] = values_df["p10"].astype(int)
        values_df["p90"] = values_df["p90"].astype(int)
        values_df["probabilistic_values"] = values_df[["p10", "p90"]].apply(
            lambda row: json.dumps(row.to_dict()),
            axis=1,
        )
        values_df.drop(columns=["p10", "p90"], inplace=True)
        return values_df

    def _prepare_data_sources(self) -> None:
        """Pull and prepare data sources required for inference."""
        log.info("Preparing data sources")

        # Create root data directory if not exists
        with contextlib.suppress(FileExistsError):
            os.mkdir(root_data_path)
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
                ),
            )
        if "mo_global" in nwp_keys:
            nwp_configs.append(
                NWPProcessAndCacheConfig(
                    source_nwp_path=os.environ["NWP_MO_GLOBAL_ZARR_PATH"],
                    dest_nwp_path=nwp_mo_global_path,
                    source="mo_global",
                ),
            )

        # Remove local cached zarr if already exists
        for nwp_config in nwp_configs:
            # Process/cache remote zarr locally
            process_and_cache_nwp(nwp_config)
        if use_satellite and "satellite" in self.config["input_data"]:
            download_satellite_data(satellite_source_file_path)

        log.info("Preparing Site data sources")
        # Clear local cached site data if already exists
        shutil.rmtree(site_path, ignore_errors=True)
        os.mkdir(site_path)

        # Save generation data as netcdf file
        generation_xr = self.generation_data["data"]

        forecast_timesteps = pd.date_range(
            start=self.t0 - pd.Timedelta("52h"),
            periods=4 * 24 * 4.5,
            freq="15min",
        )

        generation_xr = generation_xr.reindex(time_utc=forecast_timesteps, fill_value=0.00001)
        log.info(forecast_timesteps)

        generation_xr.to_netcdf(site_netcdf_path, engine="h5netcdf")

        # Save metadata as csv
        self.generation_data["metadata"].to_csv(site_metadata_path, index=False)

    def _get_config(self) -> None:
        """Setup dataloader with prepared data sources."""
        log.info("Creating configuration")

        # Pull the data config from huggingface

        data_config_filename = PVNetBaseModel.get_data_config(
            self.id,
            revision=self.version,
            token=self.hf_token,
        )

        # Populate the data config with production data paths
        populated_data_config_filename = "data/data_config.yaml"
        log.info(populated_data_config_filename)
        # if the file already exists, remove it
        if os.path.exists(populated_data_config_filename):
            os.remove(populated_data_config_filename)

        self.config = populate_data_config_sources(
            data_config_filename,
            populated_data_config_filename,
        )
        self.populated_data_config_filename = populated_data_config_filename

    def _create_dataloader(self) -> None:

        if not os.path.exists(self.populated_data_config_filename):
            raise FileNotFoundError(
                f"Data config file not found: {self.populated_data_config_filename}",
            )

        # Location and time datapipes
        self.dataset = SitesDataset(config_filename=self.populated_data_config_filename)

    def _load_model(self) -> PVNetBaseModel:
        """Load model."""
        log.info(f"Loading model: {self.id} - {self.version} ({self.name})")

        return PVNetBaseModel.from_pretrained(
            model_id=self.id,
            revision=self.version,
            token=self.hf_token,
        ).to(DEVICE)
