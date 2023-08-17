import warnings
import math
import os
import random
from collections import namedtuple
from itertools import product
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union, Callable

import numpy as np
import rasterio
import rasterio.crs
import torch
import xarray as xr
from rasterio.windows import Window
from torch.utils.data import Dataset, IterableDataset

from sen12tp import utils
from sen12tp.constants import BandNames

Patchsize = namedtuple("Patchsize", ["width", "height"])

DimensionNames = ["band", "width", "height"]

# The TrainingTarget named tuple encodes information about a training target.
# Thereby its name, a method for calculating it taking an xr.DataArray and returning
# one and the bands, which are required for calculating it.
TrainingTarget = namedtuple(
    "TrainingTarget", ["name", "calculation_func", "required_bands"]
)

TrainingTargets = [
    TrainingTarget("NDVI", utils.calculate_ndvi, ["B4", "B8"]),
    TrainingTarget("NDWI_11", utils.calculate_ndwi11, ["B8A", "B11"]),
    TrainingTarget("NDWI_12", utils.calculate_ndwi12, ["B8A", "B12"]),
    TrainingTarget("NDRE", utils.calculate_ndre, ["B8A", "B5"]),
    TrainingTarget("GNDVI", utils.calculate_gndvi, ["B8A", "B3"]),
    TrainingTarget("worldcover", utils.calculate_worldcover, ["worldcover"]),
    TrainingTarget("B4", utils.get_b4_target_band, ["B4"]),
    TrainingTarget("B3", utils.get_b3_target_band, ["B3"]),
    TrainingTarget("B2", utils.get_b2_target_band, ["B2"]),
]


def load_patch(
    path: Path,
    col: int,
    row: int,
    patch_size: Patchsize,
    stride: int = 249,
) -> Tuple[
    np.ndarray, rasterio.coords.BoundingBox, rasterio.crs.CRS, rasterio.transform.Affine
]:
    """
    Read a patch from the image at path.

    Windowed reading of GDAL/rasterio is used for performant reading:
    https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html

    In some cases, not all bands are needed for training. However, the performance is the same whether loading one or
    eleven bands. Therefore, all bands are loaded every time.

    Parameters
    ----------
    path : Path
        Path to image
    col : int
        the column from which the patch should be taken
    row : int
        the row from which the patch should be taken
    patch_size : Patchsize
        a tuple of the width and height of the patch
    stride : int, optional
        the width of the rows/columns (in px), by default 250

    Returns
    -------
    Tuple[ np.ndarray, rasterio.coords.BoundingBox, rasterio.crs.CRS, rasterio.transform.Affine ]
        a Tuple of a patch and all the relevant info from the original image.
    """

    col_start_index = col * stride
    row_start_index = row * stride
    window = Window(
        col_off=col_start_index,
        row_off=row_start_index,
        width=patch_size.width,
        height=patch_size.height,
    )
    with rasterio.open(path) as img_fd:
        img_fd: rasterio.DatasetReader
        image_data: np.ndarray = img_fd.read(window=window)
        bbox = img_fd.window_bounds(window=window)
        bbox = rasterio.coords.BoundingBox(*bbox)
        crs: rasterio.crs.CRS = img_fd.crs
        window_transform = rasterio.windows.transform(window, img_fd.transform)

    debug_data = (
        image_data.shape,
        patch_size,
        stride,
        col,
        row,
    )
    assert image_data.shape[1] == patch_size.height, ("Size wrong!",) + debug_data
    assert image_data.shape[2] == patch_size.width, ("Size wrong!",) + debug_data

    return image_data, bbox, crs, window_transform


def get_tifs_in_roidir(roi_dir: Path, modalities: List[str] = None) -> Dict[str, Path]:
    """
    From a path to a ROI directory return a dictionary with the image paths for each modality.

    Parameters
    ----------
    roi_dir : Path
        Path to the ROI Directory
    modalities : List[str], optional
        List describing the modalities included in the dataset, by default None

    Returns
    -------
    Dict[str, Path]
        A dictionary with a path pointing to each modality included in the dataset.
    """

    if modalities is None:
        modalities = list(BandNames.keys())
    tif_dict = dict.fromkeys(modalities)
    tifs = roi_dir.glob("*.tif")
    for tif in tifs:
        for modality in tif_dict.keys():
            if "_" + modality in tif.name.lower():
                tif_dict[modality] = tif
    assert None not in tif_dict.values(), (
        "For some modalities no tif file could be found. Is the dataset path correct? ('tif_dict wrong')",
        roi_dir,
        tif_dict,
        list(tifs),
    )
    return tif_dict


def get_col_row_count(image_path: Path, stride: int) -> Tuple[int, int]:
    """
    Return the number of usable rows and columns of an image given the stride.
    """

    assert image_path.is_file()
    with rasterio.open(image_path) as img:
        width = img.width
        height = img.height
    columns = width // stride
    rows = height // stride
    assert columns >= 1, ("Not at least one column", image_path, stride)
    assert rows >= 1, ("Not at least one row", image_path, stride)

    return columns, rows


def sample_dict_from_sample(
    sample: xr.DataArray, bands: List[str], indexes: List[str]
) -> Dict[str, np.ndarray]:
    if len(indexes) >= 1:
        # get a list with xr.DataArray with the shape of (1, N, M)
        targets = [get_training_target_by_name(index_name) for index_name in indexes]
        label_data = [t.calculation_func(sample) for t in targets]
        # convert to list with N * M arrays
        label_data = [data.squeeze() for data in label_data]
        label_data = np.stack(label_data)  # get one array with indexes, N, M
        label_data = 0.5 * label_data + 0.5  # normalize from [-1, 1] to [0, 1]
        # enforce float32, as some data might also be float64
        label = label_data.astype(np.float32)
    else:
        label = None

    sample = sample.loc[bands]
    sample = sample.values  # get the numpy array
    if sample.shape[1] != sample.shape[2]:
        warnings.warn(
            "Rectangular patches are not tested. x and y dimension might be swapped!"
        )
    if label is None:
        label = np.zeros_like(sample)
    sample = {
        "image": sample,
        "label": label,
    }
    return sample


def get_training_target_by_name(name: str) -> TrainingTarget:
    targets: List[TrainingTarget] = list(
        filter(lambda t: t.name == name, TrainingTargets)
    )
    if len(targets) == 0:
        raise ValueError(f"No TrainingTarget found with name {name}")
    else:
        return targets[0]


class SEN12TP(Dataset):
    """PyTorch dataset class for the dataset
    the data is in subdirectories defined by the ROIs
    """

    def __init__(
        self,
        path: Union[str, Path],
        patch_size: Patchsize = Patchsize(128, 128),
        transform: Callable[[xr.DataArray], xr.DataArray] = None,
        model_inputs: List[str] = None,
        clip_transform: Callable[[xr.DataArray], xr.DataArray] = None,
        end_transform: Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]] = None,
        model_targets: List[str] = None,
        data_bands: Optional[List[str]] = None,
        stride: int = 249,
    ):
        """Initialize the dataset

        data_bands defines, which bands the loaded data has for each modality.
        This is needed, if for example we want to get the VV, VH
        channels but the tif file additionally includes the inc_Angle band. Then we load all bands, set the band names
        as data_bands and can select only the needed VV, VH bands.
        """
        super(SEN12TP, self).__init__()
        assert os.path.exists(path), ("Directory does not exist!", path)

        self.path = Path(path)
        assert (
            len(list(self.path.iterdir())) > 0
        ), "Dataset does not contain any directories!"
        self.patch_size = patch_size
        self.stride = stride
        self.patches = list()
        self.modalities = list()
        self.model_inputs = model_inputs
        self.model_targets = ["NDVI"] if model_targets is None else model_targets

        self.transform = transform
        self.clip_transform = clip_transform
        self.end_transform = end_transform
        self.data_bands = data_bands

        # add modalities to access self.used_bands. modalities, that are not needed are not loaded during training.
        for used_band in self.model_inputs:
            # add all modalities that are required for the model input
            for modality, bandnames in BandNames.items():
                if used_band in bandnames and modality not in self.modalities:
                    self.modalities.append(modality)

            # add all modalities that are required for the model targets
            for index in self.model_targets:
                training_target = get_training_target_by_name(index)

                def _get_modality(_target: TrainingTarget):
                    required_bands = _target.required_bands
                    for modality, bands in BandNames.items():
                        shared_bands = set(required_bands).intersection(set(bands))
                        if len(shared_bands) >= 1:
                            yield modality
                    return

                target_modalities = set(_get_modality(training_target))
                for target_modality in target_modalities:
                    if target_modality not in self.modalities:
                        self.modalities.append(target_modality)

        roi_dirs = sorted([d for d in self.path.glob("*") if d.is_dir()])
        assert len(roi_dirs) >= 1, ("Not one roi dir present!", self.path)
        for roi_dir in roi_dirs:
            roi_tifs = get_tifs_in_roidir(roi_dir, modalities=self.modalities)
            size_distingishing_modality = "s2" if "s2" in roi_tifs else "s1"
            col_count, row_count = get_col_row_count(
                roi_tifs[size_distingishing_modality], stride=self.stride
            )
            columns = range(col_count)
            rows = range(row_count)
            for c, r in product(columns, rows):
                patch = (roi_tifs, c, r)
                self.patches.append(patch)

    def get_image_dataarray(self, index: int) -> xr.DataArray:
        """get and load sample from index file"""

        roi_tifs, c, r = self.patches[index]
        sample = self.load_sample(roi_tifs, c, r, band_names_dict=self.data_bands)
        if self.clip_transform:
            sample = self.clip_transform(sample)

        if self.transform:
            sample = self.transform(sample)
        error_msg = f"Sample is not of type xr.DataArray!"
        assert isinstance(sample, xr.DataArray), (error_msg, type(sample))
        return sample

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Get a single example from the dataset"""
        sample = self.get_image_dataarray(index)
        sample_dict = sample_dict_from_sample(
            sample, bands=self.model_inputs, indexes=self.model_targets
        )
        if self.end_transform:
            sample_dict = self.end_transform(sample_dict)
        return sample_dict

    def __len__(self) -> int:
        """Get number of samples in the dataset"""
        return len(self.patches)

    def load_sample(
        self,
        tif_dict: dict,
        img_column: int,
        img_row: int,
        band_names_dict: Optional[Dict[str, List[str]]] = None,
    ) -> xr.DataArray:
        """
        Read a multi-modal patch from a column and row of the tif. For each modality an array is returned.

        This is already a slightly optimized version: creating a xr.DataArray for each modality and then concating it
        had a low performance, because some internal methods were run frequently which wasted 15% performance.
        Therefore, first all modalities are used as numpy, then stack and only at the end converted to a xr.DataArray.

        Parameters
        ----------
        tif_dict : dict
            The Tif image from which the sample should be loaded.
        img_column : int
            the column of the image from which to load the sample
        img_row : int
            the row of the image from which to load the sample
        band_names_dict :
            A list of band names to include in the sample, by default None

        Returns
        -------
        xr.DataArray
            A dataArray containing a patch for each different modality out of original image
        """

        def _get_scene_id():
            """Get the scene id of a patch. Checks, that it is identical for all modalities."""
            scene_id_roi: Optional[int] = None
            for modality in self.modalities:
                modality_tif_path = tif_dict[modality]
                scene_id_modality = int(modality_tif_path.stem.split("_")[0])
                if scene_id_roi is None:
                    scene_id_roi = scene_id_modality
                else:
                    assert scene_id_roi == scene_id_modality
            return scene_id_roi

        band_names_dict = band_names_dict if band_names_dict else BandNames
        assert all(
            [isinstance(t, list) for t in band_names_dict.values()]
        ), "Wrong type, not list!"
        imgs: Dict[str, np.ndarray] = dict()  # store the image array for each modality
        band_names_modalities: Dict[str, List[str]] = dict()

        for modality in self.modalities:
            modality_tif_path = tif_dict[modality]

            modality_img, bbox, crs, transform = load_patch(
                modality_tif_path,
                col=img_column,
                row=img_row,
                patch_size=self.patch_size,
                stride=self.stride,
            )
            imgs[modality] = modality_img

            modality_channel_count = modality_img.shape[0]
            band_names_modality = band_names_dict[modality]
            error_msg = (
                "Number of band names does not match the number of image channels.",
                band_names_modality,
                len(band_names_modality),
                modality_channel_count,
            )
            assert len(band_names_modality) == modality_channel_count, error_msg
            band_names_modalities[modality] = band_names_modality

        sample_data = np.concatenate(list(imgs.values()), axis=0)

        band_names_list = list()
        [band_names_list.extend(names) for names in band_names_modalities.values()]

        assert_msg = "Number of band names not correct! Adjust the data_bands argument of SEN12TP."
        assert len(band_names_list) == sample_data.shape[0], (
            assert_msg,
            sample_data.shape,
            band_names_list,
        )

        # construct output xr.DataArray
        sample_data = sample_data.astype(np.float32)
        scene_id = _get_scene_id()
        modality_img_xr = xr.DataArray(
            sample_data,
            dims=DimensionNames,
            coords={"band": band_names_list},
            attrs={
                "bbox": bbox,
                "crs": crs,
                "transform": transform,
                "scene_id": scene_id,
            },
        )
        return modality_img_xr


class FilteredSEN12TP(IterableDataset):
    """
    FilteredSEN12TP is the dataset, which should be typically used for training models.
    As some patches can contain clouds or no data areas, the patches have to be filtered again, which is done
    on-the-fly when loading the patches from the larger tif images.

    The start_index and end_index variables are needed to support proper data loading when multiple workers
    are used, see: https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """

    def __init__(
        self,
        dataset: SEN12TP,
        cloud_prob_threshold: int = 40,
        shuffle: bool = False,
        start_index: int = 0,
        end_index: int = None,
    ):
        self.ds = dataset
        error_msg = "cloud_prob_threshold not between 0 and 100!"
        assert 0 < cloud_prob_threshold < 100, (error_msg, cloud_prob_threshold)
        self.cloud_probability_threshold = cloud_prob_threshold
        total_pixel_count = np.product(self.ds.patch_size)
        self.cloud_pixel_count_threshold = (
            total_pixel_count / 1e3
        )  # allow only 1 per thousands cloud pixels
        self.shuffle = shuffle
        end_index = end_index if end_index else len(self.ds)
        assert start_index < end_index
        assert start_index >= 0
        assert end_index <= len(self.ds)
        self.start_index = start_index
        self.end_index = end_index
        self.current_index = self.start_index

    @property
    def used_bands(self):
        return self.ds.model_inputs

    def __iter__(self):
        # When using multiple workers for dataloading, care must be taken to get different samples from each worker.
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            # shuffling has to happen here, otherwise, the patches of the underlying dataset could be shuffled twice
            # resulting in duplicate elements returned
            if self.shuffle:
                random.shuffle(self.ds.patches)
            iter_start = self.start_index
            iter_end = self.end_index
        else:  # in a worker process
            # split workload
            per_worker = int(
                math.ceil(
                    (self.end_index - self.start_index) / float(worker_info.num_workers)
                )
            )
            worker_id = worker_info.id
            iter_start = self.start_index + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end_index)
            if self.shuffle:
                random.seed(worker_info.seed)

                head = self.ds.patches[:iter_start]
                center = self.ds.patches[iter_start:iter_end]
                tail = self.ds.patches[iter_end:]

                random.shuffle(center)

                self.ds.patches = head + center + tail

        return FilteredSEN12TP(
            self.ds,
            cloud_prob_threshold=self.cloud_probability_threshold,
            shuffle=self.shuffle,
            start_index=iter_start,
            end_index=iter_end,
        )

    def __next__(self) -> dict:
        """
        Returns the next good (=cloud free, no NaN) patch of the dataset.
        """
        if self.current_index < self.end_index:
            img = self.ds.get_image_dataarray(self.current_index)

            self.current_index += 1

            # if 'cloud_probability' not in img.band set cloud mask to False
            # this is the case for non-optical data

            if "cloud_probability" not in img.band:
                cloud_mask = np.zeros(img.shape[1:], dtype=np.bool)
            else:
                cloud_mask = (
                    img.loc[["cloud_probability"]] > self.cloud_probability_threshold
                )

            # skip patches containing NaN
            if np.any(np.isnan(img)):
                return next(self)
            # skip patches containing clouds
            if np.sum(cloud_mask) > self.cloud_pixel_count_threshold:
                return next(self)

            # skip patches containing no data areas
            # no data areas are areas, where a pixel in all bands is zero
            avail_s1_bands = list(filter(lambda b: b in img.band, BandNames["s1"]))
            if avail_s1_bands:
                if (img.loc[avail_s1_bands].sum("band") == 0).any():
                    return next(self)
            avail_s2_bands = list(filter(lambda b: b in img.band, BandNames["s2"]))
            if avail_s2_bands:
                if (img.loc[avail_s2_bands].sum("band") == 0).any():
                    return next(self)

            sample_dict = sample_dict_from_sample(
                img, bands=self.ds.model_inputs, indexes=self.ds.model_targets
            )
            if self.ds.end_transform:
                sample_dict = self.ds.end_transform(sample_dict)
            return sample_dict
        else:
            raise StopIteration
