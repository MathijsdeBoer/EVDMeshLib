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
    allow_missing_keys : bool, optional
        Whether to allow missing keys or not.

    Returns
    -------
    list[mt.Transform]
        The list of transforms.
    """
    return [
        mt.LoadImaged(keys=maps),
        mt.EnsureTyped(keys=maps, dtype=torch.float32),
        JsonKeypointLoaderd(
            json_key=json_key,
            output_key=label_key,
            keypoint_names=keypoints,
            allow_missing_keys=allow_missing_keys,
        ),
        mt.EnsureChannelFirstd(keys=maps),
        mt.ScaleIntensityd(keys=maps, minv=-1.0, maxv=1.0, channel_wise=False),
        mt.ConcatItemsd(keys=maps, name=image_key, dim=0),
        mt.DeleteItemsd(keys=[*maps, json_key]),
        mt.ToTensord(keys=[image_key, label_key], allow_missing_keys=allow_missing_keys),
    ]


def default_augment_transforms(
    image_key: str = "image",
    label_key: str = "label",
) -> list[mt.Transform]:
    """
    Default augmentations for training.

    Note that these are only applied to the images, further augmentations
    may be applied to the source meshes ahead of training.

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
        mt.OneOf(
            [
                mt.RandGaussianSmoothd(
                    keys=[image_key], prob=0.5, sigma_x=(1.0, 2.0), sigma_y=(1.0, 2.0)
                ),
                mt.RandGaussianSharpend(keys=[image_key], prob=0.5),
            ]
        ),
        mt.OneOf(
            [
                mt.RandBiasFieldd(keys=[image_key], prob=0.5, coeff_range=(0.0, 0.1)),
                mt.RandGaussianNoised(keys=[image_key], prob=0.5, std=0.1),
                mt.RandAdjustContrastd(keys=[image_key], prob=0.5, gamma=(0.5, 1.5)),
            ]
        ),
    ]
