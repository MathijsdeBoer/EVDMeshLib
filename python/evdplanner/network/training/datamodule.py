from pathlib import Path

import monai.transforms as mt
from lightning.pytorch import LightningDataModule
from monai.data import CacheDataset
from torch.utils.data import DataLoader, Dataset, random_split


def validate_samples(samples: list[dict]) -> None:
    for sample in samples:
        if not isinstance(sample, dict):
            print(sample)
            raise ValueError("Each sample must be a dictionary.")
        if "map_depth" not in sample:
            print(sample)
            raise ValueError("Each sample must contain an 'map_depth' key.")
        if "map_normal" not in sample:
            print(sample)
            raise ValueError("Each sample must contain an 'map_normal' key.")
        if "label" not in sample:
            print(sample)
            raise ValueError("Each sample must contain a 'label' key.")

        if not isinstance(sample["map_depth"], Path):
            print(sample)
            raise ValueError("The 'map_depth' key must contain a Path object.")
        if not isinstance(sample["map_normal"], Path):
            print(sample)
            raise ValueError("The 'map_normal' key must contain a Path object.")
        if not isinstance(sample["label"], Path):
            print(sample)
            raise ValueError("The 'label' key must contain a Path object.")

        if not sample["map_depth"].exists():
            print(sample)
            raise ValueError("The 'map_depth' file does not exist.")
        if not sample["map_normal"].exists():
            print(sample)
            raise ValueError("The 'map_normal' file does not exist.")
        if not sample["label"].exists():
            print(sample)
            raise ValueError("The 'label' file does not exist.")


class EVDPlannerDataModule(LightningDataModule):
    def __init__(
        self,
        train_samples: list[dict],
        test_samples: list[dict],
        load_transforms: list[mt.Transform] = None,
        augment_transforms: list[mt.Transform] = None,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        validate_samples(train_samples)
        validate_samples(test_samples)

        super().__init__()
        self.train_samples = train_samples
        self.test_samples = test_samples

        self.load_transforms = load_transforms or []
        self.augment_transforms = augment_transforms or []

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage: str | None = None):
        self.train_data, self.val_data = random_split(
            CacheDataset(
                self.train_samples,
                transform=mt.Compose(self.load_transforms + self.augment_transforms),
            ),
            [0.8, 0.2],
        )
        self.test_data = CacheDataset(
            self.test_samples,
            transform=mt.Compose(self.load_transforms),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )
