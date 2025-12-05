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
from ocf_data_sampler.numpy_sample.common_types import TensorBatch
from ocf_data_sampler.numpy_sample.sun_position import calculate_azimuth_and_elevation
from ocf_data_sampler.torch_datasets.pvnet_dataset import PVNetConcurrentDataset
from ocf_data_sampler.torch_datasets.utils.torch_batch_utils import (
    batch_to_tensor,
)
from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvnet_summation.models.base_model import BaseModel as SummationBaseModel

from site_forecast_app.data.satellite import download_satellite_data

from .consts import (
    generation_path,
    nwp_ecmwf_path,
    nwp_mo_global_path,
    root_data_path,
)
from .utils import (
    NWPProcessAndCacheConfig,
    populate_data_config_sources,
    process_and_cache_nwp,
    satellite_path,
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
        timestamp: pd.Timestamp,
        generation_data: dict[str, pd.DataFrame],
        hf_repo: str,
        hf_version: str,
        name: str,
        site_uuid: str,
        satellite_scaling_method: str = "constant",
        summation_repo: str | None = None,
        summation_version: str | None = None,
    ) -> None:
        """Initializer for the model."""
        self.id = hf_repo
        self.version = hf_version
        self.name = name
        self.site_uuid = site_uuid
        self.t0 = timestamp
        self.satellite_scaling_method = satellite_scaling_method
        self.summation_repo = summation_repo
        self.summation_version = summation_version

        log.info(f"Model initialised at t0={self.t0}")

        self.client = os.getenv("CLIENT_NAME", "nl")
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN", None)

        if self.hf_token is not None:
            log.info("We are using a Hugging Face token for authentication.")
        else:
            log.warning("No Hugging Face token provided, using anonymous access.")

        # Setup the data, dataloader, and model
        if len(generation_data["generation_mw"].location_id) == 1:
            self.generation_data = generation_data["generation_mw"]
        else:
            # Cutting off National generation data (location_id=0) to avoid it being sampled
            self.generation_data = generation_data["generation_mw"].sel(location_id=slice(1, None))
        self._get_config()

        try:
            self._prepare_data_sources()
            self._create_dataloader()
            self.model = self._load_model()
            if summation_repo and summation_version:
                self.summation_model = self._load_summation_model()
            else:
                self.summation_model = None
        except Exception as e:
            log.exception("Failed to prepare data sources or load model.")
            log.exception(f"Error: {e}")

    def predict(self, timestamp: pd.Timestamp) -> list[dict] | dict[int, list[dict]]:
        """Make a prediction for the model."""
        batch = self.construct_batch(timestamp)

        # Run batch through model
        with torch.no_grad():
            normed_preds = self.model(batch).detach().cpu().numpy()

        normed_preds = set_night_time_zeros(batch, normed_preds, t0_idx=self.t0_idx)

        # log max prediction
        log.info(f"Max prediction: {np.max(normed_preds, axis=1)}")
        log.info("Completed batch")

        # calculate target times
        n_times = normed_preds.shape[1]
        sample_t0 = batch["t0"].numpy().astype("datetime64[s]")[0]
        valid_times = pd.to_datetime(
            [sample_t0 + np.timedelta64(15 * (i + 1), "m") for i in range(n_times)],
        )

        self.valid_times = valid_times

        if not self.summation_model:
            # No summation model, saving forecast for single site
            site_values = self._prepare_values_for_saving(
                capacity_kw=self.generation_data["capacity_mwp"][0][0].item() * 1000,
                normed_values=normed_preds[0],
            )
            return site_values

        else:
            # Run summation model

            inputs = self._construct_sum_sample(pvnet_outputs=normed_preds)

            # Expand for batch dimension and convert to tensors
            inputs = {k: torch.from_numpy(v[None, ...]).to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                normed_national = self.summation_model(inputs).detach().squeeze().cpu().numpy()

            log.info(f"Max national prediction: {np.max(normed_national, axis=0)}")

            # Construct forecast for saving one site at a time and store
            # in a dictionary with ml_id (location_id) as keys
            all_values = {}

            for i, location_id in enumerate(batch["location_id"].numpy()):
                all_values[location_id] = self._prepare_values_for_saving(
                    capacity_kw=self.generation_data.sel(location_id=location_id)["capacity_mwp"][
                        0
                    ].item()
                    * 1000,
                    normed_values=normed_preds[i],
                )

            # Construct and store forecast for National prediction (location_id=0)
            all_values[0] = self._prepare_values_for_saving(
                capacity_kw=self.generation_data["capacity_mwp"][0][0].item() * 1000,
                normed_values=normed_national,
            )

            return all_values

    def add_probabilistic_values(
        self,
        capacity_kw: int,
        normed_preds: np.ndarray,
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

        # add 10th and 90th percentage
        values_df["p10"] = normed_preds[:, idx_10] * capacity_kw
        values_df["p90"] = normed_preds[:, idx_90] * capacity_kw
        # change to intergers
        values_df["p10"] = values_df["p10"].astype(int)
        values_df["p90"] = values_df["p90"].astype(int)
        values_df["probabilistic_values"] = values_df[["p10", "p90"]].apply(
            lambda row: json.dumps(row.to_dict()),
            axis=1,
        )
        values_df.drop(columns=["p10", "p90"], inplace=True)
        return values_df

    def construct_batch(self, timestamp: pd.Timestamp) -> TensorBatch:
        """Prepare batch for model prediction and save it.

        Args:
                timestamp: pandas.Timestamp of when the model is run
        Returns:
                TensorBatch for t0=timestamp if available, latest t0
                if not available.
        """
        try:
            batch = self.dataset._get_batch(t0=timestamp)
            sample_t0 = timestamp
        except Exception:
            sample_t0 = self.dataset.valid_t0s[-1]
            batch = self.dataset._get_batch(t0=sample_t0)
            log.warning(
                "Timestamp different from the one in the batch: "
                f"{timestamp} != {sample_t0} (batch)"
                f"The other timestamps are: {self.dataset.valid_t0s}",
            )

        sample_location_id = batch["location_id"]

        batch = batch_to_tensor(batch)

        # to cover both site_cos_time and cos_time we duplicate some keys
        # this should get removed in an upgrade of pvnet
        for key in ["time_cos", "time_sin", "date_cos", "date_sin"]:
            if key in batch:
                batch[f"site_{key}"] = batch[key]

        # set MO GLOBAL cloud_cover_total to 0
        mo_global_nan_total_cloud_cover = os.getenv("MO_GLOBAL_ZERO_TOTAL_CLOUD_COVER", "1") == "1"
        if "mo_global" in self.config["input_data"]["nwp"] and mo_global_nan_total_cloud_cover:
            log.warning("Setting MO Global total cloud cover variables to nans")
            # In training cloud_cover_total were 0, lets do the same here
            channels = list(batch["nwp"]["mo_global"]["nwp_channel_names"])
            idx = channels.index("cloud_cover_total")

            batch["nwp"]["mo_global"]["nwp"][:, :, idx] = 0

        # save batch
        save_batch(batch=batch, model_name=self.name, site_uuid=self.site_uuid)

        log.info(f"Predicting for {sample_t0=}, {(sample_location_id)=}")

        return batch

    def _prepare_values_for_saving(self, capacity_kw: int, normed_values: np.ndarray) -> list[dict]:
        """Helper function to prepare predictions for saving.

        Construct rows for database, including start_utc, end_utc,
        and values for p10, p50, p90. Unnormalises predictions and clips values to >= 0.

        Args:
                capacity_kw: capacity of the site to unnormalise values
                normed_values: np.ndarray of predictions, shape [target times, quantiles]

        Returns:
                list of row dictionaries for saving to database.
        """
        # index of the 50th percentile, assumes number of p values odd and in order
        middle_plevel_index = normed_values.shape[1] // 2
        values_df = pd.DataFrame(
            [
                {
                    "start_utc": self.valid_times[j],
                    "end_utc": self.valid_times[j] + dt.timedelta(minutes=15),
                    "forecast_power_kw": int(v * capacity_kw),
                }
                for j, v in enumerate(normed_values[:, middle_plevel_index])
            ],
        )
        # Remove negative values
        values_df["forecast_power_kw"] = values_df["forecast_power_kw"].clip(lower=0.0)

        values_df = self.add_probabilistic_values(
            capacity_kw,
            normed_values,
            values_df,
        )

        return values_df.to_dict("records")

    def _prepare_data_sources(self) -> None:
        """Pull and prepare data sources required for inference."""
        log.info("Preparing data sources")

        # Create root data directory if not exists
        with contextlib.suppress(FileExistsError):
            os.mkdir(root_data_path)
        # Load remote zarr source
        use_satellite = os.getenv("USE_SATELLITE", "true").lower() == "true"
        satellite_source_file_path = os.getenv("SATELLITE_ZARR_PATH", None)
        satellite_backup_source_file_path = os.getenv("SATELLITE_BACKUP_ZARR_PATH", None)

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
            download_satellite_data(
                satellite_source_file_path,
                satellite_path,
                self.satellite_scaling_method,
                satellite_backup_source_file_path,
            )

        log.info("Preparing Site data sources")
        # Clear local cached site data if already exists
        shutil.rmtree(generation_path, ignore_errors=True)
        os.mkdir(generation_path)

        # Save generation data as netcdf file
        generation_xr = self.generation_data

        forecast_timesteps = pd.date_range(
            start=self.t0 - pd.Timedelta("52h"),
            periods=int(4 * 24 * 4.5),
            freq="15min",
        )

        generation_xr = generation_xr.reindex(time_utc=forecast_timesteps, fill_value=0.00001)
        log.info(forecast_timesteps)

        # Save to zarr
        generation_xr.to_zarr(generation_path, mode="w")

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

        # set t0_idx
        generation_config = self.config["input_data"]["generation"]
        self.t0_idx = int(
            -generation_config["interval_start_minutes"]
            / generation_config["time_resolution_minutes"],
        )

    def _create_dataloader(self) -> None:
        if not os.path.exists(self.populated_data_config_filename):
            raise FileNotFoundError(
                f"Data config file not found: {self.populated_data_config_filename}",
            )

        # Location and time datapipes
        self.dataset = PVNetConcurrentDataset(config_filename=self.populated_data_config_filename)

    def _load_model(self) -> PVNetBaseModel:
        """Load model."""
        log.info(f"Loading model: {self.id} - {self.version} ({self.name})")

        return PVNetBaseModel.from_pretrained(
            model_id=self.id,
            revision=self.version,
            token=self.hf_token,
        ).to(DEVICE)

    def _load_summation_model(self) -> SummationBaseModel:
        """Load summation model."""
        log.info(
            f"Loading model: {self.summation_repo} - {self.summation_version} "
            f"({self.name + '_summation'})",
        )

        return SummationBaseModel.from_pretrained(
            model_id=self.summation_repo,
            revision=self.summation_version,
        ).to(DEVICE)

    def _construct_sum_sample(self, pvnet_outputs: np.ndarray) -> dict:
        """Create a sample for the summation model.

        Args:
                pvnet_outputs: normalised outputs of the site model
        Returns:
                batch for summation model as a dictionary.
        """
        # National data has location_id=0, regional location_ids start from 1:
        # relative_capacities = regional_capacities / national_capacity
        relative_capacities = (
            self.generation_data.loc[1:]["capacity_mwp"].values
            / self.generation_data.loc[0]["capacity_mwp"]
        )

        # Getting sun position for National location (National index is always 0)
        azimuth, elevation = calculate_azimuth_and_elevation(
            datetimes=self.valid_times,
            lon=self.generation_data.loc[0]["longitude"],
            lat=self.generation_data.loc[0]["latitude"],
        )

        sample = {
            # Numpy array with batch size = num_locations
            "pvnet_outputs": pvnet_outputs,
            # Shape: [time]
            "valid_times": self.valid_times.values.astype(int),
            # Shape: [num_locations]
            "relative_capacity": relative_capacities,
            # Shape: [time]
            "azimuth": azimuth.astype(np.float32) / 360,
            # Shape: [time]
            "elevation": elevation.astype(np.float32) / 180 + 0.5,
        }

        return sample
