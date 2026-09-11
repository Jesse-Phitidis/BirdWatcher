import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torch
from PIL import Image
import pandas as pd
from typing import Callable


class BirdsDataset(Dataset):

    def __init__(self, csv_path: Path, dataset_path: Path, transforms: Callable):
        self.df = pd.read_csv(csv_path)
        self.dataset_path = dataset_path
        self.transforms = transforms

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]

        img = Image.open(self.dataset_path / row["filepaths"].replace("jpg", ".jpg"))
        img = self.transforms(img)

        sample = {
            "img": img,
            "class_id": torch.tensor(row["class id"], dtype=torch.long),
            "label": row["label"]
        }

        return sample
    
    def __len__(self):
        return len(self.df)


class ClassifierDataModule(pl.LightningDataModule):

    def __init__(
            self, 
            train_csv_path: str, 
            test_csv_path: str, 
            dataset_path: str, 
            train_transforms: Callable,
            test_transforms: Callable,
            batch_size: int,
            num_workers: int,
            ):
        super().__init__()

        self.trainer_csv_path = Path(train_csv_path)
        self.test_csv_path = Path(test_csv_path)
        self.dataset_path = Path(dataset_path)
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = BirdsDataset(self.trainer_csv_path, self.dataset_path, self.train_transforms)
            self.val_dataset = BirdsDataset(self.test_csv_path, self.dataset_path, self.test_transforms)
        if stage == "test":
            self.test_dataset = BirdsDataset(self.test_csv_path, self.dataset_path, self.test_transforms)
        if stage == "predict":
            self.predict_dataset = BirdsDataset(self.test_csv_path, self.dataset_path, self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
