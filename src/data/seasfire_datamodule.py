"""
SeasFire Lightning DataModule — wraps the dataset for training / validation / testing.

Adapted from the ORORA project (Orion AI Lab, National Observatory of Athens).
"""

from typing import Optional
import time
import numpy as np
import xarray as xr
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset

from .seasfire_dataset import load_datasets, create_dataset_for_years


class SeasFireDataModule(LightningDataModule):
    """
    Lightning DataModule for the SeasFire datacube.

    Loads local (0.25°) and global (1°) datacubes, creates patch-based
    datasets with configurable lag and lead time, and exposes standard
    train / val / test DataLoaders.
    """

    def __init__(
        self,
        ds_path: str = None,
        ds_path_global: str = None,
        input_vars: list = None,
        positional_vars: list = None,
        oci_vars: list = None,
        clip_vars: dict = None,
        local_lag: int = 1,
        global_lag: int = 1,
        oci_lag: int = 10,
        log_transform_vars: list = None,
        target: str = "gwis_ba",
        task: str = "classification",
        target_shift: int = 1,
        input_local_shape: list = None,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = False,
        debug: bool = False,
        patch_local_shape: list = None,
        patch_global_shape: list = None,
        min_lead_time: int = 1,
        max_lead_time: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.ds_path = ds_path
        self.ds_path_global = ds_path_global
        self.input_vars = list(input_vars) if input_vars else []
        self.positional_vars = list(positional_vars) if positional_vars else []
        self.oci_vars = list(oci_vars) if oci_vars else []
        self.clip_vars = clip_vars or {}
        self.log_transform_vars = log_transform_vars or []
        self.target = target
        self.task = task
        self.target_shift = target_shift
        self.local_lag = local_lag
        self.global_lag = global_lag
        self.oci_lag = oci_lag
        self.input_local_shape = tuple(input_local_shape or [1, 80, 80])
        self.patch_local_shape = patch_local_shape
        self.patch_global_shape = patch_global_shape
        self.min_lead_time = min_lead_time
        self.max_lead_time = max_lead_time
        self.debug = debug

        # Year splits
        if self.debug:
            self.training_years = [2002, 2003]
            self.validation_years = [2002, 2003]
            self.test_years = [2002, 2003]
            self.selected_years = [2002, 2003]
        else:
            self.training_years = list(range(2003, 2019))
            self.validation_years = [2019]
            self.test_years = [2020, 2021]
            self.selected_years = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data and create train / val / test datasets."""
        if not self.data_train and not self.data_val and not self.data_test:
            start = time.time()
            print("Loading datasets ...")
            ds, global_ds, oci_ds = load_datasets(
                self.ds_path,
                self.ds_path_global,
                self.input_vars,
                self.log_transform_vars,
                self.oci_vars,
                self.clip_vars,
                keep_vars=["area", "gwis_ba", "gfed_region", "ndvi", "pop_dens", "lsm"],
                selected_years=self.selected_years,
            )
            print(f"Datasets loaded in {time.time() - start:.1f}s")

            common_kwargs = dict(
                input_vars=self.input_vars,
                oci_vars=self.oci_vars,
                positional_vars=self.positional_vars,
                target=self.target,
                min_lead_time=self.target_shift,
                max_lead_time=self.target_shift,
                local_lag=self.local_lag,
                oci_lag=self.oci_lag,
                global_lag=self.global_lag,
                patch_size=self.input_local_shape,
                task=self.task,
                patch_local_shape=self.patch_local_shape,
                patch_global_shape=self.patch_global_shape,
            )

            print("Creating training dataset ...")
            self.data_train = create_dataset_for_years(
                ds, global_ds, oci_ds, years=self.training_years, **common_kwargs
            )

            print("Creating validation dataset ...")
            self.data_val = create_dataset_for_years(
                ds, global_ds, oci_ds, years=self.validation_years, **common_kwargs
            )
            # Shuffle validation indices
            val_indices = list(range(len(self.data_val)))
            np.random.shuffle(val_indices)
            self.data_val = Subset(self.data_val, val_indices)

            print("Creating test dataset ...")
            self.data_test = create_dataset_for_years(
                ds, global_ds, oci_ds, years=self.test_years, **common_kwargs
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
