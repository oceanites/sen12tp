# SEN12TP dataset and dataloader for pytorch

* dataset is organized in directories of ROIs -> can be loaded using SEN12TP
* large ROIs contain some small clouds, need to be filtered during training -> FilteredSEN12TP excludes them on the fly
* for training split up ROI directories in train/val/test directories -> datamodule

## Dataset structure

```
├── 0
│   ├── 0_cgls.tif
│   ├── 0_dem.tif
│   ├── 0_modis.tif
│   ├── 0_s1.tif
│   └── 0_s2.tif
├── 1
│   ├── 1_cgls.tif
│   ├── 1_dem.tif
│   ├── 1_modis.tif
│   ├── 1_s1.tif
│   └── 1_s2.tif
├── 2
.
.
.
```

## Code Examples
### SEN12TP
```python
from functools import partial
from sen12tp.dataset import SEN12TP, Patchsize
import sen12tp.utils

model_input_bands = ["VV_corrected", "VH_corrected"]
# The sensor data has some outliers inside, therefore clipping the data to a defined value range
# is useful to exclude these outliers using the `clip_transform()` method.
clip_transform = sen12tp.utils.default_clipping_transform

# As deep learning models converge better with normalized data, this transform method takes the
# clipped data and normalizes it. This example used min-max-normalization to a value range [0, 1].
normalization_transform = sen12tp.utils.min_max_transform

ds = SEN12TP("dataset_dir/train", 
             model_inputs=model_input_bands, 
             model_targets=["NDVI", "NDRE"],
             patch_size=Patchsize(256, 256),
             transform=normalization_transform,
             clip_transform=clip_transform,
             stride=256,
             )
# access a sample: use integer index
patch = ds[34]  # returns a dictionary
# returns a dictionary with the keys "image" and "label"
patch.keys()
# dict_keys(['image', 'label'])
patch['image'].shape
# [number of input bands x patchsize x patchsize]
patch['label'].shape
# [number of target bands x patchsize x patchsize]
```

### FilteredSEN12TP

```python
import random
from sen12tp.dataset import SEN12TP, FilteredSEN12TP

ds = SEN12TP(...)
# if the data should be shuffled for training, shuffle the patches of the dataset once here
random.shuffle(ds.patches)
filtered_ds = FilteredSEN12TP(ds)

# accessing the values by running the iterator
for sample in filtered_ds:
    print(sample['image'].shape, sample['label'].shape)
```

### PyTorch Lightning DataModule

```python
from sen12tp.datamodule import SEN12TPDataModule
from sen12tp.dataset import Patchsize
from sen12tp.utils import min_max_transform
from functools import partial

min_max_transform_with_bands = partial(min_max_transform, bands=["VV_sigma0", "VH_sigma0"])

dm = SEN12TPDataModule(
    dataset_dir="/data/sen12tc-full-splitted/",
    batch_size=24,
    max_workers=2,
    patch_size=Patchsize(256, 256),
    stride=256,
    model_inputs=["VV_sigma0", "VH_sigma0"],
    model_targets=["NDVI"],
    transform=min_max_transform_with_bands,
)
```


## Dependencies
see `requirements.txt`

## Install