import warnings

import numpy as np
import xarray as xr

from sen12tp.constants import BandNames

from sen12tp.constants import (
    MIN_VV_VALUE,
    MIN_VH_VALUE,
    MIN_DEM_VALUE,
    MAX_DEM_VALUE,
    cgls_simplified_mapping,
)


def calculate_ndvi(data: xr.DataArray) -> xr.DataArray:
    ndvi = calculate_normalized_difference_xarray(data, "B8", "B4", "ndvi")
    return ndvi


def calculate_ndwi11(data: xr.DataArray) -> xr.DataArray:
    ndwi11 = calculate_normalized_difference_xarray(data, "B8A", "B11", "ndwi_11")
    return ndwi11


def calculate_ndwi12(data: xr.DataArray) -> xr.DataArray:
    ndwi12 = calculate_normalized_difference_xarray(data, "B8A", "B12", "ndwi_12")
    return ndwi12


def calculate_ndre(data: xr.DataArray) -> xr.DataArray:
    ndre = calculate_normalized_difference_xarray(data, "B8A", "B5", "ndre")
    return ndre


def calculate_gndvi(data: xr.DataArray) -> xr.DataArray:
    gndvi = calculate_normalized_difference_xarray(data, "B8A", "B3", "gndvi")
    return gndvi


def calculate_worldcover(data: xr.DataArray) -> xr.DataArray:
    worldcover_bandname = "worldcover"
    worldcover = data.loc[worldcover_bandname]
    # worldcover is in intervall [0, 1] by method ndvi_prediction/utils.py min_max_transform()
    worldcover = (2 * worldcover) - 1  # to interval [-1, 1] to be the same as NDVI
    return worldcover


def get_target_band_data(data: xr.DataArray, bandname: str) -> xr.DataArray:
    band_data = data.loc[bandname]
    # band data is in intervall [0, 1] by method ndvi_prediction/utils.py min_max_transform()
    band_data = (2 * band_data) - 1  # to interval [-1, 1] to be the same as NDVI
    return band_data


def get_b4_target_band(data: xr.DataArray) -> xr.DataArray:
    return get_target_band_data(data, "B4")


def get_b3_target_band(data: xr.DataArray) -> xr.DataArray:
    return get_target_band_data(data, "B3")


def get_b2_target_band(data: xr.DataArray) -> xr.DataArray:
    return get_target_band_data(data, "B2")


def calculate_normalized_difference_xarray(
    arr: xr.DataArray, band1: str, band2: str, index_name: str
) -> xr.DataArray:
    ndiff = calculate_normalized_difference(
        arr1=arr.loc[band1].values, arr2=arr.loc[band2].values
    )
    ndiff = ndiff.reshape((1,) + ndiff.shape)
    ndiff = xr.DataArray(ndiff, dims=arr.dims, coords={"band": [index_name]})
    return ndiff


def calculate_normalized_difference(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Calculates the Normalized difference from two arrays.

    ndiff = (arr1 - arr2) / (arr1 + arr1)
    Replaces NaN values with 0.
    """
    assert (
        arr1.shape == arr2.shape
    ), f"Shapes are not equal: {arr1.shape} != {arr2.shape}!"
    assert isinstance(arr1, np.ndarray)
    assert isinstance(arr2, np.ndarray)
    # calculating the NDVI can raise a warning, when dividing by 0
    # as the nan values will be replaced with 0, this warning can be thrown away
    with warnings.catch_warnings(record=True) as w:
        norm_diff = (arr1 - arr2) / (arr1 + arr2)
        # test, that really only the correct warning ("RuntimeWarning: invalid
        # value encountered in true_divide") was catched
        assert len(w) <= 1, f"There should at most 1 warning, not {len(w)}!"
        if len(w) == 1:
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "invalid value encountered in true_divide" in str(w[-1].message)
    norm_diff = np.nan_to_num(norm_diff, nan=0)
    return norm_diff


def default_clipping_transform(sample: xr.DataArray) -> xr.DataArray:
    """
    Clips and cleans a data sample to default values.

    Sentinel-2 data ist clipped to [0, 10000].
    Sentinel-1 VV data is clipped to MIN/MAX_VV_VALUE, which should be -25, 0.
    Sentinel-1 VH data is clipped to MIN/MAX_VH_VALUE, which should be -32.5, 0.
    DEM data is clipped to MIN/MAX_DEM_VALUE, which should be -450, 9000.
    For DEM and optical data NaN values are replaced by 0, for VV and VH to MIN_VV_VALUE and MIN_VH_value,
    respectively.
    """
    avail_s2_bands = [band for band in BandNames["s2"] if band in sample.band]
    if len(avail_s2_bands) > 0:
        sample.loc[avail_s2_bands] = (
            sample.loc[avail_s2_bands]
            .astype(np.float32)
            .clip(min=0, max=10000)
            .fillna(value=0)
        )
        sample.loc["cloud_probability"] = sample.loc["cloud_probability"].clip(
            min=0, max=100
        )

    if "VV_sigma0" in sample.band:
        vv_bands = ["VV_sigma0", "VV_corrected"]
        avail_vv_bands = [band for band in vv_bands if band in sample.band]

        sample.loc[avail_vv_bands] = (
            sample.loc[avail_vv_bands]
            .astype(np.float32)
            .clip(min=MIN_VV_VALUE, max=0)
            .fillna(value=MIN_VV_VALUE)
        )

    if "VH_sigma0" in sample.band:
        vh_bands = ["VH_sigma0", "VH_corrected"]
        avail_vh_bands = [band for band in vh_bands if band in sample.band]
        sample.loc[avail_vh_bands] = (
            sample.loc[avail_vh_bands]
            .astype(np.float32)
            .clip(min=MIN_VH_VALUE, max=0)
            .fillna(value=MIN_VH_VALUE)
        )

    if "dem" in sample.band:
        sample.loc[BandNames["dem"]] = (
            sample.loc["dem"]
            .astype(np.float32)
            .clip(min=MIN_DEM_VALUE, max=MAX_DEM_VALUE)
            .fillna(value=0)
        )
    return sample


def min_max_transform(sample: xr.DataArray) -> xr.DataArray:
    """Transforms the sample data from the clipped value range to [0, 1]"""
    sample_bands = sample.coords["band"].values
    ### Sentinel-1
    vv_bands = list(filter(lambda b: "VV_" in b, sample_bands))
    vh_bands = list(filter(lambda b: "VH_" in b, sample_bands))
    s1_bands = vv_bands + vh_bands
    if vv_bands:
        sample.loc[vv_bands] = sample.loc[vv_bands] / MIN_VV_VALUE
    if vh_bands:
        sample.loc[vh_bands] = sample.loc[vh_bands] / MIN_VH_VALUE
    if s1_bands:
        assert sample.loc[s1_bands].min() >= 0, sample.loc[s1_bands].min()
        assert sample.loc[s1_bands].max() <= 1, sample.loc[s1_bands].max()

    ### Sentinel-2
    s2_bands = list(filter(lambda b: b in BandNames["s2"], sample_bands))
    if s2_bands:
        sample.loc[s2_bands] = sample.loc[s2_bands] / 10_000
    if "cloud_probability" in s2_bands:
        # in the first step the cloud probability gets normalized from [0, 100] to [0, 0.01]
        # therefore, it is normalized in this step to [0, 1]
        sample.loc[s2_bands] = sample.loc[s2_bands] * 100

    ### CGLS
    cgls_bands = list(filter(lambda b: "cgls" in b, sample_bands))
    if cgls_bands:
        sample.loc[cgls_bands] = simplify_cgls(sample.loc[cgls_bands])
        sample.loc[cgls_bands] = sample.loc[cgls_bands] / max(
            cgls_simplified_mapping.values()
        )

    ### DEM
    dem_bands = list(filter(lambda b: "dem" in b, sample_bands))
    if dem_bands:
        assert MIN_DEM_VALUE <= sample.loc[dem_bands].min(), (
            f"DEM should be above {MIN_DEM_VALUE}, but min is",
            sample.loc[dem_bands].min(),
        )
        assert sample.loc[dem_bands].max() <= MAX_DEM_VALUE, (
            f"DEM should be below {MAX_DEM_VALUE}, but max is",
            sample.loc[dem_bands].max(),
        )
        range_dem = MAX_DEM_VALUE - MIN_DEM_VALUE
        sample.loc[dem_bands] = (sample.loc[dem_bands] - MIN_DEM_VALUE) / range_dem

    ### Worldcover
    worldcover_bands = list(filter(lambda b: "worldcover" in b, sample_bands))
    if worldcover_bands:
        sample.loc[worldcover_bands] = sample.loc[worldcover_bands] / 100

    ### Off Nadir Angle
    angle_bands = list(filter(lambda b: "incAngle" in b, sample_bands))
    if angle_bands:
        sample.loc[angle_bands] = sample.loc[angle_bands] / 90

    ### Modis
    modis_bands = list(filter(lambda b: "LC_" in b, sample_bands))
    if modis_bands:
        sample.loc[modis_bands] = sample.loc[modis_bands] / 1000

    return sample
