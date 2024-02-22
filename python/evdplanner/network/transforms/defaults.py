"""
Default transforms for loading and preprocessing data.
"""

import monai.transforms as mt
import torch

from .json_keypoint_loader import JsonKeypointLoaderd


def default_load_transforms(
    maps: list[str],
    keypoints: list[str] = None,
    json_key: str = "keypoints",
    image_key: str = "image",
    label_key: str = "label",
    allow_missing_keys: bool = False,
) -> list[mt.Transform]:
    """
    Default transforms for loading data.

    Parameters
    ----------
    maps : list[str]
        The list of keys for the input maps.
    keypoints : list[str], optional
        The list of keys for the keypoints.
    json_key : str, optional
        The key for the JSON file containing the keypoints.
    image_key : str, optional
        The key for the image.
    label_key : str, optional
        The key for the label.

    Returns
    -------
    list[mt.Transform]
        The list of transforms.
    """
    return [
        mt.LoadImaged(keys=maps),
        mt.EnsureTyped(keys=maps, dtype=torch.float32),
        JsonKeypointLoaderd(json_key=json_key, output_key=label_key, keypoint_names=keypoints, allow_missing_keys=allow_missing_keys),
        mt.EnsureChannelFirstd(keys=maps),
        mt.ConcatItemsd(keys=maps, name=image_key),
        mt.DeleteItemsd(keys=[*maps, json_key]),
    ]


def default_preparation_transforms(
    image_key: str = "image",
    label_key: str = "label",
    allow_missing_keys: bool = False,
    channel_dim: int | None = None
) -> list[mt.Transform]:
    """
    Default transforms for preprocessing data.

    Parameters
    ----------
    image_key : str, optional
        The key for the image.
    label_key : str, optional
        The key for the label.

    Returns
    -------
    list[mt.Transform]
        The list of transforms.
    """
    return [
        mt.EnsureChannelFirstd(keys=[image_key], channel_dim=channel_dim),
        mt.ScaleIntensityd(keys=[image_key], minv=-1.0, maxv=1.0, channel_wise=False),
        mt.ToTensord(keys=[image_key, label_key], allow_missing_keys=allow_missing_keys),
    ]


def default_raw_transforms(
    maps: list[str],
    keypoints: list[str] = None,
    json_key: str = "keypoints",
    image_key: str = "image",
    label_key: str = "label",
    allow_missing_keys: bool = False,
) -> list[mt.Transform]:
    """
    Default transforms for loading and preprocessing data.

    Parameters
    ----------
    maps : list[str]
        The list of keys for the input maps.
    keypoints : list[str], optional
        The list of keys for the keypoints.
    json_key : str, optional
        The key for the JSON file containing the keypoints.
    image_key : str, optional
        The key for the image.
    label_key : str, optional
        The key for the label.

    Returns
    -------
    list[mt.Transform]
        The list of transforms.
    """
    return default_load_transforms(
        maps, keypoints, json_key, image_key, label_key, allow_missing_keys
    ) + default_preparation_transforms(image_key, label_key, allow_missing_keys)
