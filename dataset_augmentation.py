"""
dataset_augmentation.py is a script which applies augmentation (Albumentations) on the training dataset.

Usage:
python dataset_augmentation.py

* Required folder structure before running this script: * 

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

- If you do not have this structure yet, first run: "python dataset_splitter.py"

* Output folder structure: * 

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
├── train_aug/
│   ├── images/
│   │   ├── img_021_aug.png
│   │   ├── img_022_aug.png
│   │   └── ...
│   └── masks/
│       ├── img_021_aug.png
│       ├── img_022_aug.png
│       └── ...

"""

import albumentations as A
import cv2
import os
from tqdm import tqdm

train_img_dir = "/lustre/BIF/nobackup/su030/leaf_images/efficientvit_dataset/train/images"
train_mask_dir = "/lustre/BIF/nobackup/su030/leaf_images/efficientvit_dataset/train/masks"

aug_img_dir = "/lustre/BIF/nobackup/su030/leaf_images/efficientvit_dataset/train_aug/images"
aug_mask_dir = "/lustre/BIF/nobackup/su030/leaf_images/efficientvit_dataset/train_aug/masks"

os.makedirs(aug_img_dir, exist_ok=True)
os.makedirs(aug_mask_dir, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.4),
    A.GaussianBlur(blur_limit=3, p=0.3),
])

for file_name in tqdm(os.listdir(train_img_dir)):
    img = cv2.imread(os.path.join(train_img_dir, file_name))
    mask = cv2.imread(os.path.join(train_mask_dir, file_name), cv2.IMREAD_GRAYSCALE)

    augmented = transform(image=img, mask=mask)
    aug_img = augmented['image']
    aug_mask = augmented['mask']

    base, ext = os.path.splitext(file_name)
    cv2.imwrite(os.path.join(aug_img_dir, base + "_aug.png"), aug_img)
    cv2.imwrite(os.path.join(aug_mask_dir, base + "_aug.png"), aug_mask)
