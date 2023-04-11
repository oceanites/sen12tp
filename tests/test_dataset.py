import pytest
import torch.utils.data

import sen12tp.dataset
from pathlib import Path
import xarray as xr
import numpy as np
from pytest import fixture
import rasterio.coords
import rasterio.transform

import sen12tp.utils

SAMPLE_SIZE = 7
DATASET_IMG_SIZE = 6


@fixture
def arr_s2():
    bands = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
        "cloud_probability",
    ]
    s2_arr = np.random.random((len(bands), SAMPLE_SIZE, SAMPLE_SIZE))
    s2 = xr.DataArray(
        s2_arr,
        dims=["band", "w", "h"],
        coords={"band": bands},
        attrs={
            "bbox": rasterio.coords.BoundingBox(0, 1, 1, 0),
            "crs": "example_crs",
            "transform": rasterio.transform.Affine(0, 0, 0, 0, 0, 0),
        },
    )
    return s2


@fixture
def dataset_fixture_dir() -> Path:
    """This test creates a dataset with stub data to test the sen12tp dataset and dataloader logic.

    Creates 20 ROIs with all modalities defined in sen12tp.dataset.BandNames.
    """
    dataset_fixture_dir = Path(__file__).parent / "fixtures" / "dataset"
    # dataset_fixture_dir = Path(f"fixtures/dataset/")
    if not dataset_fixture_dir.is_dir():
        for mod_i, (modality, bandnames) in enumerate(sen12tp.utils.BandNames.items()):
            for roi_i in range(20):
                fill_value = (
                    mod_i + 1
                ) * roi_i + 1  # set different values for each roi and modality
                channel_count = len(bandnames)
                data = np.arange(
                    fill_value,
                    fill_value + channel_count * DATASET_IMG_SIZE * DATASET_IMG_SIZE,
                    1,
                    dtype=np.int32,
                )
                data = data.reshape((channel_count, DATASET_IMG_SIZE, DATASET_IMG_SIZE))
                # set the cloud mask channel
                data[-1, :, :] = np.full(
                    (DATASET_IMG_SIZE, DATASET_IMG_SIZE),
                    fill_value=fill_value,
                    dtype=np.int32,
                )
                # data = np.full((channel_count, img_size, img_size), fill_value=fill_value, dtype=np.int32)
                roi_dir = dataset_fixture_dir / f"{roi_i}/"
                roi_dir.mkdir(parents=True, exist_ok=True)

                with rasterio.open(
                    roi_dir / f"{roi_i}_{modality}.tif",
                    "w",
                    driver="GTiff",
                    height=data.shape[1],
                    width=data.shape[2],
                    count=data.shape[0],
                    dtype=data.dtype,
                    crs="EPSG:3857",
                ) as dst:
                    dst.write(data, list(range(1, channel_count + 1)))
    return dataset_fixture_dir


def test_dir(dataset_fixture_dir):
    assert dataset_fixture_dir.is_dir()


def test_load_patch():
    path = Path("fixtures/0_cgls.tif")
    border_length = 444
    patch_size = sen12tp.dataset.Patchsize(border_length, border_length)
    patch, bbox, crs, transform = sen12tp.dataset.load_patch(
        path=path, col=0, row=0, patch_size=patch_size
    )
    assert patch.shape == (1, border_length, border_length)


def test_get_col_row_count():
    path = Path("fixtures/0_cgls.tif")
    test_data = [
        (128, (15, 15)),
        (256, (7, 7)),
        (512, (3, 3)),
        (2000, (1, 1)),
        (1, (2001, 2001)),
    ]
    for stride, expected_value in test_data:
        actual = sen12tp.dataset.get_col_row_count(image_path=path, stride=stride)
        assert actual == expected_value


def test_sample_dict_from_sample(arr_s2):
    # def sample_dict_from_sample(sample: xr.DataArray, bands: List[str], indexes: List[str]) -> Dict[str, np.ndarray]:
    indexes = ["NDVI"]
    bands = ["B1", "B4"]
    sample = sen12tp.dataset.sample_dict_from_sample(
        arr_s2, bands=bands, indexes=indexes
    )
    assert "image" in sample
    assert "label" in sample
    assert sample["label"].shape == (len(indexes), SAMPLE_SIZE, SAMPLE_SIZE)
    assert sample["image"].shape == (len(bands), SAMPLE_SIZE, SAMPLE_SIZE)


def test_sample_dict_from_sample_two_indixes(arr_s2):
    # def sample_dict_from_sample(sample: xr.DataArray, bands: List[str], indexes: List[str]) -> Dict[str, np.ndarray]:
    indexes = ["NDVI", "NDWI_11", "NDWI_12"]
    bands = ["B1", "B4"]
    sample = sen12tp.dataset.sample_dict_from_sample(
        arr_s2, bands=bands, indexes=indexes
    )
    assert "image" in sample
    assert "label" in sample
    assert sample["label"].shape == (len(indexes), SAMPLE_SIZE, SAMPLE_SIZE)
    assert sample["image"].shape == (len(bands), SAMPLE_SIZE, SAMPLE_SIZE)


def test_get_tifs_in_roidir(dataset_fixture_dir: Path):
    roi = 0
    roi_dir = dataset_fixture_dir / str(roi)
    assert roi_dir.is_dir()
    tif_dict = sen12tp.dataset.get_tifs_in_roidir(roi_dir)
    assert isinstance(tif_dict, dict)
    for modality in sen12tp.constants.BandNames:
        assert modality in tif_dict
        mod_path = tif_dict[modality]
        assert f"{roi}_{modality}.tif" == mod_path.name


def test_SEN12TP(dataset_fixture_dir):
    """Test to check dataset loading."""
    ds = sen12tp.dataset.SEN12TP(
        path=dataset_fixture_dir,
        patch_size=sen12tp.dataset.Patchsize(DATASET_IMG_SIZE, DATASET_IMG_SIZE),
        model_inputs=["VV_sigma0", "VH_sigma0"],
        stride=DATASET_IMG_SIZE,
    )
    assert ds is not None, "A object should be returned"
    assert isinstance(ds, object), "A object should be returned"
    assert isinstance(
        ds, sen12tp.dataset.SEN12TP
    ), "The returned value should be a SEN12TC object"
    assert len(ds) == 20, "The length of the dataset should be 20!"
    for sample in ds:
        assert isinstance(sample, dict), "Each sample of the dataset should be a dict!"
        assert "label" in sample, "Each dataset sample should contain a 'label'!"
        assert "image" in sample, "Each dataset sample should contain a 'image'!"

    # get the first value of each sample of the dataset
    img_values = [ds[i]["image"][0, 0, 0] for i in range(len(ds))]
    assert len(img_values) == len(
        set(img_values)
    ), "Dataset should not contain identical images!"


@pytest.mark.parametrize("num_workers", [1, 2])
def test_dataloader(dataset_fixture_dir, num_workers):
    """Test a dataloader using SEN12TP and FilteredSEN12TP."""
    batch_size = 5
    model_inputs = ["VV_sigma0", "VH_sigma0"]
    ds = sen12tp.dataset.SEN12TP(
        dataset_fixture_dir,
        patch_size=sen12tp.dataset.Patchsize(DATASET_IMG_SIZE, DATASET_IMG_SIZE),
        model_inputs=model_inputs,
        stride=DATASET_IMG_SIZE,
    )
    ds_iter = sen12tp.dataset.FilteredSEN12TP(ds)
    dataloader = torch.utils.data.DataLoader(
        ds_iter, num_workers=num_workers, batch_size=batch_size
    )

    img_values = list()
    for batch in dataloader:
        img = batch["image"].detach().numpy()
        assert img.shape == (
            batch_size,
            len(model_inputs),
            6,
            6,
        ), "Shape of sample image not correct!"
        img_values.extend(
            list(img[:, 0, 0, 0])
        )  # get the values of the first pixel of each roi
    assert len(img_values) == 20, "Number of retrieved batch elements not correct"
    assert len(img_values) == len(
        set(img_values)
    ), "No duplicate values should be in the dataset"
    if num_workers == 1:
        assert img_values != sorted(img_values)


def test_FilteredSEN12TP_shuffling(dataset_fixture_dir):
    model_inputs = ["VV_sigma0", "VH_sigma0"]
    ds = sen12tp.dataset.SEN12TP(
        dataset_fixture_dir,
        patch_size=sen12tp.dataset.Patchsize(DATASET_IMG_SIZE, DATASET_IMG_SIZE),
        model_inputs=model_inputs,
        stride=DATASET_IMG_SIZE,
    )
    ds_iter_unshuffled = sen12tp.dataset.FilteredSEN12TP(ds)
    ds_iter_shuffled = sen12tp.dataset.FilteredSEN12TP(ds, shuffle=True)
    img_values_unshuffled = [sample["image"][0, 0, 0] for sample in ds_iter_unshuffled]
    img_values_shuffled = [sample["image"][0, 0, 0] for sample in ds_iter_shuffled]
    assert len(img_values_shuffled) == len(img_values_unshuffled) == 20
    assert len(img_values_shuffled) == len(
        set(img_values_shuffled)
    ), "Duplicate values found!"
    assert len(img_values_unshuffled) == len(
        set(img_values_unshuffled)
    ), "Duplicate values found!"
    assert img_values_unshuffled != img_values_shuffled

    # convert img_values to original roi number
    rois_unshuffled = list(map(lambda i: (i - 1) / 2, img_values_unshuffled))
    # string sorting is required, because the patches are sorted according to the string of the roi
    # and therefore 21 comes before 3
    rois_unshuffled_str = list(map(lambda i: str(i), rois_unshuffled))
    assert rois_unshuffled_str == sorted(rois_unshuffled_str)


@pytest.mark.parametrize(
    "index_name", [t.name for t in sen12tp.dataset.TrainingTargets]
)
def test_SEN12TP_single_index(dataset_fixture_dir, index_name):
    """Test to get the dataset with a single vegetation index as label."""
    used_bands = ["VV_sigma0", "VH_sigma0"]
    used_indexes = [index_name]
    ds = sen12tp.dataset.SEN12TP(
        dataset_fixture_dir,
        patch_size=sen12tp.dataset.Patchsize(DATASET_IMG_SIZE, DATASET_IMG_SIZE),
        model_inputs=used_bands,
        model_targets=used_indexes,
        stride=DATASET_IMG_SIZE,
    )
    for elem_dict in ds:
        assert "image" in elem_dict
        assert "label" in elem_dict
        elem_label = elem_dict["label"]
        assert elem_label.shape == (
            1,
            DATASET_IMG_SIZE,
            DATASET_IMG_SIZE,
        ), elem_label.shape
        break


def test_SEN12TP_all_indexes(dataset_fixture_dir):
    """Test to get the dataset with all vegetation indices as label."""
    used_bands = ["VV_sigma0", "VH_sigma0"]
    used_indexes = [t.name for t in sen12tp.dataset.TrainingTargets]
    ds = sen12tp.dataset.SEN12TP(
        dataset_fixture_dir,
        patch_size=sen12tp.dataset.Patchsize(DATASET_IMG_SIZE, DATASET_IMG_SIZE),
        model_inputs=used_bands,
        model_targets=used_indexes,
        stride=DATASET_IMG_SIZE,
    )

    for elem_dict in ds:
        assert "image" in elem_dict
        assert "label" in elem_dict
        elem_label = elem_dict["label"]
        assert elem_label.shape == (
            len(used_indexes),
            DATASET_IMG_SIZE,
            DATASET_IMG_SIZE,
        ), elem_label.shape

        for i in range(len(used_indexes) - 1):
            v1, v2 = elem_label[i, 0, 0], elem_label[i + 1, 0, 0]
            assert pytest.approx(v1) != pytest.approx(
                v2
            ), f"Label data should not be equal! {v1}=={v2}"
