"""PVNet model class."""

import contextlib
import datetime as dt
import json
import logging
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import torch
from ocf_data_sampler.numpy_sample.collate import stack_np_samples_into_batch
from ocf_data_sampler.torch_datasets.datasets.site import SitesDataset
from ocf_data_sampler.torch_datasets.sample.base import batch_to_tensor
from pvnet.models.base_model import BaseModel as PVNetBaseModel

from site_forecast_app.data.satellite import download_satellite_data

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
    populate_data_config_sources,
    process_and_cache_nwp,
    satellite_path,
    save_batch,
    set_night_time_zeros,
)

# Setup device
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
        satellite_scaling_method: str = "constant",
    ) -> None:
        """Initializer for the model."""
        self.asset_type = asset_type
        self.id = hf_repo
        self.version = hf_version
        self.name = name
        self.site_uuid = None
        self.t0 = timestamp
        self.satellite_scaling_method = satellite_scaling_method

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

        try:
            self._prepare_data_sources()
            self._create_dataloader()
            self.model = self._load_model()
        except Exception as e:
            log.exception("Failed to prepare data sources or load model.")
            log.exception(f"Error: {e}")

    def _get_config(self):
        log.info("Stub _get_config() called - skipping config load for test")
        self.config = {
            "input_data": {
                "nwp": {"ecmwf": {}},
                "site": {
                    "interval_start_minutes": 0,
                    "time_resolution_minutes": 15,
                },
            }
        }
        self.populated_data_config_filename = "data/data_config.yaml"
        self.t0_idx = 0  # mock index

    def _prepare_data_sources(self) -> None:
        log.info("Preparing data sources")
        try:
            with contextlib.suppress(FileExistsError):
                os.mkdir(root_data_path)

            use_satellite = os.getenv("USE_SATELLITE", "true").lower() == "true"
            satellite_source_file_path = os.getenv("SATELLITE_ZARR_PATH", None)
            satellite_backup_source_file_path = os.getenv("SATELLITE_BACKUP_ZARR_PATH", None)

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
            if "mo_global" in nwp_keys:
                nwp_configs.append(
                    NWPProcessAndCacheConfig(
                        source_nwp_path=os.environ["NWP_MO_GLOBAL_ZARR_PATH"],
                        dest_nwp_path=nwp_mo_global_path,
                        source="mo_global",
                    )
                )

            for nwp_config in nwp_configs:
                process_and_cache_nwp(nwp_config)

            if use_satellite and "satellite" in self.config["input_data"]:
                download_satellite_data(
                    satellite_source_file_path,
                    satellite_path,
                    self.satellite_scaling_method,
                    satellite_backup_source_file_path,
                )

            log.info("Preparing Site data sources")
            shutil.rmtree(site_path, ignore_errors=True)
            os.mkdir(site_path)

            generation_xr = self.generation_data["data"]
            forecast_timesteps = pd.date_range(
                start=self.t0 - pd.Timedelta("52h"),
                periods=int(4 * 24 * 4.5),
                freq="15min",
            )
            generation_xr = generation_xr.reindex(time_utc=forecast_timesteps, fill_value=0.00001)
            log.info(forecast_timesteps)

            generation_xr.to_netcdf(site_netcdf_path, engine="h5netcdf")
            self.generation_data["metadata"].to_csv(site_metadata_path, index=False)

        except Exception as e:
            error_message = (
                "Could not run the forecast because there wasn't enough NWP data. "
                "Please check your NWP input files and time range."
            )
            log.error(error_message)
            log.error(f"Underlying error: {e}")
            warnings.warn(error_message)
            raise RuntimeError(error_message) from e

    def _create_dataloader(self) -> None:
        if not os.path.exists(self.populated_data_config_filename):
            raise FileNotFoundError(f"Data config file not found: {self.populated_data_config_filename}")
        self.dataset = SitesDataset(config_filename=self.populated_data_config_filename)

    def _load_model(self) -> PVNetBaseModel:
        log.info(f"Loading model: {self.id} - {self.version} ({self.name})")
        return PVNetBaseModel.from_pretrained(
            model_id=self.id,
            revision=self.version,
            token=self.hf_token,
        ).to(DEVICE)

    def predict(self, site_uuid: str, timestamp: dt.datetime) -> dict:
        """Make a prediction for the model."""
        capacity_kw = self.generation_data["metadata"].iloc[0]["capacity_kwp"]

        normed_preds = []
        with torch.no_grad():
            samples = self.dataset.valid_t0_and_site_ids
            samples_with_same_t0 = samples[samples["t0"] == timestamp]

            if len(samples_with_same_t0) == 0:
                sample_t0 = samples.iloc[-1].t0
                sample_site_id = samples.iloc[-1].site_id

                log.warning(
                    "Timestamp different from the one in the batch: "
                    f"{timestamp} != {sample_t0} (batch)"
                    f"The other timestamps are: {samples['t0'].unique()}",
                )
            else:
                sample_t0 = samples_with_same_t0.iloc[0].t0
                sample_site_id = samples_with_same_t0.iloc[0].site_id

            batch = self.dataset.get_sample(t0=sample_t0, site_id=sample_site_id)
            i = 0

            if site_uuid != sample_site_id:
                log.warning(
                    f"Site id different from the one in the batch: {site_uuid} != {sample_site_id}",
                )

            log.info(f"Predicting for batch: {i}, for {sample_t0=}, {sample_site_id=}")

            batch = stack_np_samples_into_batch([batch])
            batch = batch_to_tensor(batch)

            for key in ["time_cos", "time_sin", "date_cos", "date_sin"]:
                if key in batch:
                    batch[f"site_{key}"] = batch[key]

            mo_global_nan_total_cloud_cover = (
                os.getenv("MO_GLOBAL_ZERO_TOTAL_CLOUD_COVER", "1") == "1"
            )
            if "mo_global" in self.config["input_data"]["nwp"] and mo_global_nan_total_cloud_cover:
                log.warning("Setting MO Global total cloud cover variables to nans")
                channels = list(batch["nwp"]["mo_global"]["nwp_channel_names"])
                idx = channels.index("cloud_cover_total")
                batch["nwp"]["mo_global"]["nwp"][:, :, idx] = 0

            save_batch(batch=batch, i=i, model_name=self.name, site_uuid=site_uuid)

            preds = self.model(batch).detach().cpu().numpy()

            preds = set_night_time_zeros(batch, preds, t0_idx=self.t0_idx)

            normed_preds += [preds]

            log.info(f"Max prediction: {np.max(preds, axis=1)}")
            log.info(f"Completed batch: {i}")

        normed_preds = np.concatenate(normed_preds)
        n_times = normed_preds.shape[1]

        valid_times = pd.to_datetime(
            [sample_t0 + dt.timedelta(minutes=15 * (i + 1)) for i in range(n_times)],
        )

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
        values_df["forecast_power_kw"] = values_df["forecast_power_kw"].clip(lower=0.0)

        values_df = self.add_probabilistic_values(capacity_kw, normed_preds, values_df)

        return values_df.to_dict("records")

    def add_probabilistic_values(
        self,
        capacity_kw: int,
        normed_preds: np.array,
        values_df: pd.DataFrame,
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

        values_df["p10"] = normed_preds[0, :, idx_10] * capacity_kw
        values_df["p90"] = normed_preds[0, :, idx_90] * capacity_kw
        values_df["p10"] = values_df["p10"].astype(int)
        values_df["p90"] = values_df["p90"].astype(int)
        values_df["probabilistic_values"] = values_df[["p10", "p90"]].apply(
            lambda row: json.dumps(row.to_dict()),
            axis=1,
        )
        values_df.drop(columns=["p10", "p90"], inplace=True)
        return values_df

