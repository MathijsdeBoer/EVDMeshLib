from pathlib import Path

import monai.transforms as mt
from lightning.pytorch import LightningDataModule
from monai.data import CacheDataset
from torch.utils.data import DataLoader, random_split


def validate_samples(
    samples: list[dict],
    maps: list[str],
    keypoints_key: str,
) -> None:
    for sample in samples:
        if not isinstance(sample, dict):
            print(sample)
            raise ValueError("Each sample must be a dictionary.")
        for m in maps:
            if m not in sample:
                print(sample)
                raise ValueError(f"Each sample must contain a '{m}' key.")
        if keypoints_key not in sample:
            print(sample)
            raise ValueError(f"Each sample must contain a '{keypoints_key}' key.")

        for m in maps:
            if not isinstance(sample[m], Path):
                print(sample)
                raise ValueError(f"The '{m}' key must contain a Path object.")
        if not isinstance(sample[keypoints_key], Path):
            print(sample)
            raise ValueError("The 'label' key must contain a Path object.")

        for m in maps:
            if not sample[m].exists():
                print(sample)
                raise ValueError(f"The '{m}' file does not exist.")
        if not sample[keypoints_key].exists():
            print(sample)
            raise ValueError("The 'label' file does not exist.")


class EVDPlannerDataModule(LightningDataModule):
    def __init__(
        self,
        train_samples: list[dict],
        maps: list[str],
        keypoints_key: str,
        test_samples: list[dict] = None,
        load_transforms: list[mt.Transform] = None,
        augment_transforms: list[mt.Transform] = None,
        val_split: float = 0.2,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        validate_samples(train_samples, maps, keypoints_key)
        if test_samples:
            validate_samples(test_samples, maps, keypoints_key)

        super().__init__()
        self.train_samples = train_samples
        self.test_samples = test_samples

        self.load_transforms = load_transforms or []
        self.augment_transforms = augment_transforms or []

        self.train_split = 1.0 - val_split
        self.val_split = val_split

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage: str | None = None) -> None:
        self.train_data, self.val_data = random_split(
            CacheDataset(
                self.train_samples,
                transform=mt.Compose(self.load_transforms + self.augment_transforms),
            ),
            [self.train_split, self.val_split],
        )
        if self.test_samples:
            self.test_data = CacheDataset(
                self.test_samples,
                transform=mt.Compose(self.load_transforms),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
        )
