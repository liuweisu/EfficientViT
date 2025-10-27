"""
dataset_splitter.py is a script that splits data from the unet_dataset folder structure into a training, validation and testing dataset and saves it into the efficientvit_dataset.

Usage:
python dataset_splitter.py

* Required datastructure: * 

unet_dataset/
├── images/
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
└── masks/
    ├── img_001.png
    ├── img_002.png
    └── ...

- This data structure can be achieved by running: "python dataset_maker.py" in the repository "leaf_damage/Pytorch-UNet/"

* Output data structure: * 

efficientvit_dataset/
├── train/
│   ├── images/
│   │   ├── img_001.png
│   │   ├── img_002.png
│   │   └── ...
│   └── masks/
│       ├── img_001.png
│       ├── img_002.png
│       └── ...
├── val/
│   ├── images/
│   │   ├── img_021.png
│   │   ├── img_022.png
│   │   └── ...
│   └── masks/
│       ├── img_021.png
│       ├── img_022.png
│       └── ...
└── test/
    ├── images/
    │   ├── img_041.png
    │   ├── img_042.png
    │   └── ...
    └── masks/
        ├── img_041.png
        ├── img_042.png
        └── ...
"""

import os, shutil, random
from tqdm import tqdm

# Source dataset
src_img_dir = "/lustre/BIF/nobackup/su030/leaf_images/unet_dataset/images"
src_mask_dir = "/lustre/BIF/nobackup/su030/leaf_images/unet_dataset/masks"

# Output dataset root
out_root = "/lustre/BIF/nobackup/su030/leaf_images/efficientvit_dataset"
splits = ["train", "val", "test"]
ratios = [0.7, 0.15, 0.15]

# Create folder structure
for split in splits:
    os.makedirs(f"{out_root}/{split}/images", exist_ok=True)
    os.makedirs(f"{out_root}/{split}/masks", exist_ok=True)

# Get all filenames
images = sorted(os.listdir(src_img_dir))
random.seed(42)
random.shuffle(images)

n = len(images)
train_split = int(ratios[0] * n)
val_split = int(ratios[1] * n)

split_dict = {
    "train": images[:train_split],
    "val": images[train_split:train_split + val_split],
    "test": images[train_split + val_split:]
}

for split, files in split_dict.items():
    for f in tqdm(files, desc=f"Copying {split}"):
        shutil.copy(os.path.join(src_img_dir, f), f"{out_root}/{split}/images/{f}")
        shutil.copy(os.path.join(src_mask_dir, f), f"{out_root}/{split}/masks/{f}")
