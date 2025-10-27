"""
efficientvit_runner.py is a script which trains, validates and tests EfficientVit-SAM XL

usage: 
python efficientvit_runner.py

Required datastructure:

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

- This data structure can be achieved by running: "python dataset_augmentation.py"
"""
# import statements
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator

# efficientvit_mask_generator = EfficientViTSamAutomaticMaskGenerator(efficientvit_sam)

# Dataset creation 
class LeafDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        mask = (mask > 127).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = torch.tensor(image).permute(2,0,1).float()
        mask = torch.tensor(mask).float()

        return image, mask
    
class EfficientViTSegmentation(nn.module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.encoder = pretrained_model.image_encoder
        self.decoder = pretrained_model.image_decoder
        
    def forward(self, x):
        image_embedding = self.encoder(x)
        sparse_embeddings = torch.zeros(x.size(0),1,256, device = x.device)
        dense_embeddings = torch.zeros(x.size(0), 256, x.shape[2]//4, x.shape[3]//4, device = x.device)
        masks, _ = self.decoder(
            image_embedding = image_embedding,
            image_pe = None,
            sparse_prompt_embeddings = sparse_embeddings,
            send_prompt_embeddings = dense_embeddings,
            multimask_output = False
        )
        return masks
    
# testing and evaluation
def dice_coeff(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + eps)

def iou_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return inter / (union + eps)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # train, val and test datasets
    train_img_dir = "/lustre/BIF/nobackup/su030/leaf_images/efficientvit_dataset/train/images"
    train_mask_dir = "/lustre/BIF/nobackup/su030/leaf_images/efficientvit_dataset/train/masks"

    aug_img_dir = "/lustre/BIF/nobackup/su030/leaf_images/efficientvit_dataset/train_aug/images"
    aug_mask_dir = "/lustre/BIF/nobackup/su030/leaf_images/efficientvit_dataset/train_aug/masks"

    val_img_dir = "/lustre/BIF/nobackup/su030/leaf_images/efficientvit_dataset/val/images"
    val_mask_dir = "/lustre/BIF/nobackup/su030/leaf_images/efficientvit_dataset/val/masks"

    test_img_dir = "/lustre/BIF/nobackup/su030/leaf_images/efficientvit_dataset/test/images"
    test_mask_dir = "/lustre/BIF/nobackup/su030/leaf_images/efficientvit_dataset/test/masks"

    # Make the data loaders
    train_dataset = torch.utils.data.ConcatDataset([
        LeafDataset(train_img_dir, train_mask_dir),
        LeafDataset(aug_img_dir, aug_mask_dir)])

    val_dataset = LeafDataset(val_img_dir, val_mask_dir)
    test_dataset = LeafDataset(test_img_dir, test_mask_dir)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch=4, shuffle=True)

    # initializing efficientvit-SAM
    efficientvit_sam = create_efficientvit_sam_model(name="efficientvit-sam-xl1", pretrained=True)
    model = EfficientViTSegmentation(efficientvit_sam).cuda()

    # training loop
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for imgs, masks in tqdm(train_loader, desc= f"Epoch {epoch+1}/{num_epochs}"):
            imgs, masks = imgs.cuda(), masks.cuda()

            preds = model(imgs)
            preds = torch.sigmoid(preds)

            loss = F.binary_cross_entropy(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print(f"Train loss: {train_loss/len(train_loader):.4f}")

        # validation 
        model.eval()
        val_loss = 0
        dice_scores = []
        iou_scores = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.cuda(), masks.cuda()

                preds = model(imgs)
                preds = torch.sigmoid(preds)

                val_loss += F.binary_cross_entropy(preds, masks).item()
                dice_scores.append(dice_coeff(preds, masks).item())
                iou_scores.append(iou_score(preds, masks).item())

        print(f"Val loss: {val_loss/len(val_loader):.4f}")

    #testing
    print("\nEvaluating best model on test set")

    model.eval()
    dice_scores = []
    iou_scores = []
    with torch.nograd():
        for imgs, masks in test_loader:
            imgs, masks = imgs.cuda(), masks.cuda()
            preds = model(imgs)
            preds = torch.sigmoid(preds)
            dice_scores.append(dice_coeff(preds, masks).item())
            iou_scores.append(iou_score(preds, masks).item())

    print(f"Mean Dice score on test set: {np.mean(dice_scores):.4f}, Test IoU: {np.mean(iou_scores):.4f}")

if __name__ == "main":
    main()