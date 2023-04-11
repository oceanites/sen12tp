MIN_VV_VALUE = -25
MIN_VH_VALUE = -32.5
MIN_DEM_VALUE = -450
MAX_DEM_VALUE = 9000


cgls_simplified_mapping = {
    0: 0,  # unknown
    20: 1,  # shrubs
    30: 2,  # herbaceaous vegetation/grassland
    40: 3,  # cropland
    50: 4,  # built up
    60: 5,  # bare/sparse
    70: 6,  # ice/snow
    80: 7,  # water
    90: 8,  # wetlands
    100: 5,  # moss/lichen -> barren
    111: 9,
    112: 9,
    113: 9,
    114: 9,
    115: 9,
    116: 9,  # forest
    121: 9,
    122: 9,
    123: 9,
    124: 9,
    125: 9,
    126: 9,
    200: 7,  # water
}

BandNames = {
    "s2": [
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
    ],
    "s1": ["VV_sigma0", "VH_sigma0", "VV_corrected", "VH_corrected", "incAngle"],
    "dem": ["dem"],
    "worldcover": ["worldcover"],
}
