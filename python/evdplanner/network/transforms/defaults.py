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
    return [
        mt.LoadImaged(keys=maps),
        mt.EnsureTyped(keys=maps, dtype=torch.float32),
        JsonKeypointLoaderd(json_key=json_key, output_key=label_key, keypoint_names=keypoints),
        mt.EnsureChannelFirstd(keys=maps),
        mt.ScaleIntensityd(keys=maps, minv=-1.0, maxv=1.0, channel_wise=False),
        mt.ConcatItemsd(keys=maps, name=image_key),
        mt.DeleteItemsd(keys=[*maps, json_key]),
        mt.ToTensord(keys=["image"]),
    ]
