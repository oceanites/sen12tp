import random
from pathlib import Path
from typing import List, Optional
import xarray as xr

import torch
import pytorch_lightning as pl

from sen12tp.dataset import Patchsize, SEN12TP, FilteredSEN12TP
from sen12tp.constants import (
    MIN_VV_VALUE,
    MIN_VH_VALUE,
    MIN_DEM_VALUE,
    MAX_DEM_VALUE,
    cgls_simplified_mapping,
)
from sen12tp.constants import BandNames
from sen12tp import default_clipping_transform


class SEN12TPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str = "path/to/dir",
        batch_size: int = 32,
        patch_size: Patchsize = Patchsize(256, 256),
        stride: int = 249,
        model_inputs: List[str] = None,
        model_targets: List[str] = None,
        transform=None,
        end_transform=None,
        num_workers: int = 1,
        pin_memory: bool = True,  # pinning the memory can increase the performance
        **kwargs,  # ignore additional keyword arguments
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.clipping_method = default_clipping_transform
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.model_inputs = (
            model_inputs if model_inputs else ["VV_corrected", "VH_corrected"]
        )
        self.model_targets = model_targets if model_targets else ["NDVI"]
        self.sen12tp_train: FilteredSEN12TP
        self.sen12tp_val: FilteredSEN12TP
        self.sen12tp_test: FilteredSEN12TP
        self.num_workers = num_workers
        self.end_transform = end_transform
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        sen12tp_kwargs = {
            "patch_size": self.patch_size,
            "transform": self.transform,
            "model_targets": self.model_targets,
            "clip_transform": self.clipping_method,
            "model_inputs": self.model_inputs,
            "end_transform": self.end_transform,
            "stride": self.stride,
        }
        sen12tp_train_ds = SEN12TP(self.dataset_dir / "train", **sen12tp_kwargs)
        sen12tp_val_ds = SEN12TP(self.dataset_dir / "val", **sen12tp_kwargs)
        sen12tp_test_ds = SEN12TP(self.dataset_dir / "test", **sen12tp_kwargs)
        random.shuffle(sen12tp_train_ds.patches)
        random.shuffle(sen12tp_val_ds.patches)
        self.sen12tp_train = FilteredSEN12TP(sen12tp_train_ds)
        self.sen12tp_val = FilteredSEN12TP(sen12tp_val_ds)
        self.sen12tp_test = FilteredSEN12TP(sen12tp_test_ds)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.sen12tp_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.sen12tp_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.sen12tp_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
