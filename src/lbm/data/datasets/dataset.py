import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import os

# --- Dataset Class (Same as before) ---
class ImageTranslationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, root_dir: str = "/home/huutien/filter_dataset"):
        drop_columns = ['time_A','time_B', 'index', 'img_path_A', 'img_path_B']
        existing_columns = [col for col in drop_columns if col in dataframe.columns]
        self.df = dataframe.drop(columns=existing_columns)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path_A = os.path.join(self.root_dir, row['img2D_path_A'])
        img_path_B = os.path.join(self.root_dir, row['img2D_path_B'])
        # Load .npy files
        img_A = np.load(img_path_A, allow_pickle=True)
        img_B = np.load(img_path_B, allow_pickle=True)

        # Add channel dimension if missing (H, W) -> (1, H, W)
        img_A = np.expand_dims(img_A, axis=0).astype(np.float32) - 0.8
        img_B = np.expand_dims(img_B, axis=0).astype(np.float32) - 0.8

        return {
            "bl": torch.from_numpy(img_A),
            "m36": torch.from_numpy(img_B),
            # "subject_id": row['subject ID'],
            # "dx_A": row['DX_A'],
            # "dx_B": row['DX_B']
        }

# --- Updated DataModule ---
class MedicalImageTranslationDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_csv: str, 
        val_csv: str, 
        batch_size: int = 16, 
        num_workers: int = 4,
        root_dir: str = "/home/huutien/filter_ds"
    ):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        
        self.train_df = None
        self.val_df = None

    def setup(self, stage: Optional[str] = None):
        # Simply read the pre-defined CSVs
        if stage == "fit" or stage is None:
            self.train_df = pd.read_csv(self.train_csv)
            self.val_df = pd.read_csv(self.val_csv)

            print(f"Loaded Data:")
            print(f"  Training:   {len(self.train_df)} samples")
            print(f"  Validation: {len(self.val_df)} samples")

    def train_dataloader(self):
        return DataLoader(
            ImageTranslationDataset(self.train_df, root_dir=self.root_dir),
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle training data
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            ImageTranslationDataset(self.val_df, root_dir=self.root_dir),
            batch_size=self.batch_size,
            shuffle=False, # Don't shuffle validation
            num_workers=self.num_workers,
            pin_memory=True
        )