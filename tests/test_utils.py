import numpy as np
from sen12tp import utils
import xarray as xr
from pytest import fixture


@fixture
def arr_np():
    return np.random.random((4, 6, 6))


@fixture
def arr_xr(arr_np):
    return xr.DataArray(
        arr_np, dims=["band", "w", "h"], coords={"band": ["B1", "B2", "B3", "B4"]}
    )


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
    s2_arr = np.random.random((len(bands), 7, 7))
    s2 = xr.DataArray(s2_arr, dims=["band", "w", "h"], coords={"band": bands})
    return s2


def test_calculate_normalized_difference_xarray(arr_np, arr_xr):
    ndiff = utils.calculate_normalized_difference_xarray(arr_xr, "B1", "B2", "b12")
    ndiff_expected_values = (arr_np[0] - arr_np[1]) / (arr_np[0] + arr_np[1])

    assert isinstance(ndiff, xr.DataArray)
    assert ndiff.shape == (1, 6, 6)
    assert ndiff.loc["b12"].shape == (6, 6)
    assert (ndiff.loc["b12"].values == ndiff_expected_values).all()


def test_calculate_normalized_difference_xarray_same_band(arr_np, arr_xr):
    ndiff = utils.calculate_normalized_difference_xarray(arr_xr, "B1", "B1", "b11")

    assert isinstance(ndiff, xr.DataArray)
    assert ndiff.shape == (1, 6, 6)
    assert ndiff.loc["b11"].shape == (6, 6)
    assert (ndiff.loc["b11"].values == 0).all()


def test_calculate_ndvi(arr_s2):
    ndvi = utils.calculate_ndvi(arr_s2)
    ndvi_expected = (arr_s2.loc["B8"] - arr_s2.loc["B4"]) / (
        arr_s2.loc["B8"] + arr_s2.loc["B4"]
    )
    assert ndvi.shape[1:] == ndvi_expected.shape
    assert (ndvi.values == ndvi_expected.values).all()


def test_calculate_ndwi11(arr_s2):
    ndwi11 = utils.calculate_ndwi11(arr_s2)
    ndwi_11_expected = (arr_s2.loc["B8A"] - arr_s2.loc["B11"]) / (
        arr_s2.loc["B8A"] + arr_s2.loc["B11"]
    )
    assert isinstance(ndwi11, xr.DataArray)
    assert ndwi11.shape[1:] == ndwi_11_expected.shape
    assert (ndwi11.values == ndwi_11_expected.values).all()


def test_calculate_ndwi12(arr_s2):
    ndwi12 = utils.calculate_ndwi12(arr_s2)
    ndwi_12_expected = (arr_s2.loc["B8A"] - arr_s2.loc["B12"]) / (
        arr_s2.loc["B8A"] + arr_s2.loc["B12"]
    )
    assert isinstance(ndwi12, xr.DataArray)
    assert ndwi12.shape[1:] == ndwi_12_expected.shape
    assert (ndwi12.values == ndwi_12_expected.values).all()


def test_calculate_ndre(arr_s2):
    ndre = utils.calculate_ndre(arr_s2)
    ndre_expected = (arr_s2.loc["B8A"] - arr_s2.loc["B5"]) / (
        arr_s2.loc["B8A"] + arr_s2.loc["B5"]
    )
    assert isinstance(ndre, xr.DataArray)
    assert ndre.shape[1:] == ndre_expected.shape
    assert (ndre.values == ndre_expected.values).all()


def test_calculate_gndvi(arr_s2):
    gndvi = utils.calculate_gndvi(arr_s2)
    gndvi_expected = (arr_s2.loc["B8A"] - arr_s2.loc["B3"]) / (
        arr_s2.loc["B8A"] + arr_s2.loc["B3"]
    )
    assert isinstance(gndvi, xr.DataArray)
    assert gndvi.shape[1:] == gndvi_expected.shape
    assert (gndvi.values == gndvi_expected.values).all()
