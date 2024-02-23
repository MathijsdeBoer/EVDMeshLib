"""
Data module for the EVDPlanner network.
"""
import json
from pathlib import Path

import monai.transforms as mt
from lightning.pytorch import LightningDataModule
from loguru import logger
from monai.data import CacheDataset
from torch.utils.data import ConcatDataset, DataLoader


def validate_samples(
    samples: list[dict],
    maps: list[str],
    keypoints_key: str,
) -> None:
    """
    Validate the samples for the EVDPlannerDataModule.

    Parameters
    ----------
    samples : list[dict]
        List of samples.
    maps : list[str]
        List of map keys.
    keypoints_key : str
        Key for the keypoints.

    Returns
    -------
    None
    """
    for sample in samples:
        if not isinstance(sample, dict):
            msg = "Each sample must be a dictionary."
            logger.debug(json.dumps(sample, indent=4))
            logger.error(msg)
            raise ValueError(msg)
        for m in maps:
            if m not in sample:
                msg = f"Each sample must contain a '{m}' key."
                logger.debug(json.dumps(sample, indent=4))
                logger.error(msg)
                raise ValueError(msg)
        if keypoints_key not in sample:
            msg = f"Each sample must contain a '{keypoints_key}' key."
            logger.debug(json.dumps(sample, indent=4))
            logger.error(msg)
            raise ValueError(msg)

        for m in maps:
            if not isinstance(sample[m], Path):
                msg = f"The '{m}' key must contain a Path object."
                logger.debug(json.dumps(sample, indent=4))
                logger.error(msg)
                raise ValueError(msg)
        if not isinstance(sample[keypoints_key], Path):
            msg = "The 'label' key must contain a Path object."
            logger.debug(json.dumps(sample, indent=4))
            logger.error(msg)
            raise ValueError(msg)

        for m in maps:
            if not sample[m].exists():
                msg = f"The '{m}' file does not exist."
                logger.debug(json.dumps(sample, indent=4))
                logger.error(msg)
                raise ValueError(msg)
        if not sample[keypoints_key].exists():
            msg = "The 'label' file does not exist."
            logger.debug(json.dumps(sample, indent=4))
            logger.error(msg)
            raise ValueError(msg)


class EVDPlannerDataModule(LightningDataModule):
    """
    Data module for the EVDPlanner network.
    """

    def __init__(
        self,
        train_samples: list[dict],
        val_samples: list[dict],
        maps: list[str],
        keypoints_key: str,
        test_samples: list[dict] = None,
        augmented_samples: list[dict] = None,
        load_transforms: list[mt.Transform] = None,
        augment_transforms: list[mt.Transform] = None,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        """
        Initialize the EVDPlannerDataModule.

        Parameters
        ----------
        train_samples : list[dict]
            List of training samples.
        val_samples : list[dict]
            List of validation samples.
        maps : list[str]
            List of map keys.
        keypoints_key : str
            Key for the keypoints.
        test_samples : list[dict], optional
            List of test samples, by default None
        load_transforms : list[mt.Transform], optional
            List of transforms to load the data, by default None
        augment_transforms : list[mt.Transform], optional
            List of transforms to augment the data, by default None
        batch_size : int, optional
            Batch size, by default 1
        num_workers : int, optional
            Number of workers, by default 0
        """
        validate_samples(train_samples, maps, keypoints_key)
        if test_samples:
            validate_samples(test_samples, maps, keypoints_key)

        super().__init__()
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.augmented_samples = augmented_samples

        self.load_transforms = load_transforms or []
        self.augment_transforms = augment_transforms or []

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_data = None
        self.augmented_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage: str | None = None) -> None:
        """
        Setup the data module.

        Parameters
        ----------
        stage : str, optional
            Stage, by default None

        Returns
        -------
        None
        """
        self.train_data = CacheDataset(
            self.train_samples,
            transform=mt.Compose(self.load_transforms + self.augment_transforms),
        )

        self.val_data = CacheDataset(
            self.val_samples,
            transform=mt.Compose(self.load_transforms),
        )

        if self.augmented_samples:
            self.augmented_data = CacheDataset(
                self.augmented_samples,
                transform=mt.Compose(self.load_transforms + self.augment_transforms),
            )

        if self.test_samples:
            self.test_data = CacheDataset(
                self.test_samples,
                transform=mt.Compose(self.load_transforms),
            )

    def train_dataloader(self) -> DataLoader:
        """
        Train dataloader.

        Returns
        -------
        DataLoader
            Train dataloader.
        """
        if self.augmented_data:
            return DataLoader(
                ConcatDataset([self.train_data, self.augmented_data]),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                persistent_workers=self.num_workers > 0,
            )
        else:
            return DataLoader(
                self.train_data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                persistent_workers=self.num_workers > 0,
            )

    def val_dataloader(self) -> DataLoader:
        """
        Validation dataloader.

        Returns
        -------
        DataLoader
            Validation dataloader.
        """
        return DataLoader(
            self.val_data,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Test dataloader.

        Returns
        -------
        DataLoader
            Test dataloader.
        """
        return DataLoader(
            self.test_data,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
        )
