# dataset.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import h5py # Added
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BraTSDataset(Dataset):
    def __init__(self, image_files, mask_files, augmentations=None, num_input_channels=4, num_classes=3):
        self.file_paths = image_files 
        self.augmentations = augmentations
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        with h5py.File(file_path, 'r') as hf:
            image_np = hf['image'][:]
            mask_np = hf['mask'][:]

        image_np = image_np.astype(np.float32)

        if self.augmentations:
            augmented = self.augmentations(image=image_np, mask=mask_np)
            image = augmented['image'] # From ToTensorV2: (C, H, W)
            mask = augmented['mask']   # From ToTensorV2 (with transpose_mask=True): (C_mask, H, W)
        else:
            image = torch.from_numpy(image_np.transpose(2, 0, 1)).float() # (C, H, W)
            mask = torch.from_numpy(mask_np.transpose(2, 0, 1)).float() 

        return image, mask

class BraTSDataModule(pl.LightningDataModule):
    def __init__(self, all_h5_file_paths,
                 train_val_split_ratio=0.8,
                 batch_size=8, num_workers=4,
                 num_input_channels=4, num_classes=3, seed=42):
        super().__init__()
        self.save_hyperparameters()
        self.all_h5_file_paths = all_h5_file_paths

    def setup(self, stage=None):
        self.hparams.norm_mean = [0.5] * self.hparams.num_input_channels
        self.hparams.norm_std = [0.5] * self.hparams.num_input_channels

        train_augs = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7),
            A.Normalize(mean=self.hparams.norm_mean, std=self.hparams.norm_std),
            ToTensorV2(transpose_mask=True), # <--- MODIFICATION HERE
        ])
        val_augs = A.Compose([
            A.Normalize(mean=self.hparams.norm_mean, std=self.hparams.norm_std),
            ToTensorV2(transpose_mask=True), 
        ])

        num_files = len(self.all_h5_file_paths)
        num_train = int(num_files * self.hparams.train_val_split_ratio)
        
        np.random.seed(self.hparams.seed)
        shuffled_paths = np.random.permutation(self.all_h5_file_paths)
        
        train_files = list(shuffled_paths[:num_train])
        val_files = list(shuffled_paths[num_train:])

        if stage == 'fit' or stage is None:
            self.train_dataset = BraTSDataset(
                image_files=train_files, mask_files=train_files,
                augmentations=train_augs,
                num_input_channels=self.hparams.num_input_channels,
                num_classes=self.hparams.num_classes
            )
            self.val_dataset = BraTSDataset(
                image_files=val_files, mask_files=val_files,
                augmentations=val_augs,
                num_input_channels=self.hparams.num_input_channels,
                num_classes=self.hparams.num_classes
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, shuffle=True,
                          persistent_workers=True if self.hparams.num_workers > 0 else False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=True if self.hparams.num_workers > 0 else False)

    