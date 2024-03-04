"""
Default transforms for loading and preprocessing data.
"""

import monai.transforms as mt
import torch

from .json_keypoint_loader import JsonKeypointLoaderd
from .landmarks_loader import LandmarksLoaderd
from .landmarks_to_keypoints import LandmarksToKeypointsd
from .mesh_loader import MeshLoaderd
from .mesh_rand_deform import MeshRandDeformed
from .mesh_rand_rotate import MeshRandRotated
from .mesh_rendererer import MeshRenderd


def default_load_transforms(
    maps: list[str],
    keypoints: list[str] = None,
    json_key: str = "keypoints",
    image_key: str = "image",
    label_key: str = "label",
    include_file_reading: bool = True,
    allow_missing_keys: bool = False,
    input_channel_dim: int | None = None,
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
    include_file_reading : bool, optional
        Whether to include file reading or not.
    allow_missing_keys : bool, optional
        Whether to allow missing keys or not.
    input_channel_dim : int, optional
        The input channel dimension.

    Returns
    -------
    list[mt.Transform]
        The list of transforms.
    """
    transforms = [
        mt.EnsureTyped(keys=maps, dtype=torch.float32),
        mt.EnsureChannelFirstd(keys=maps, channel_dim=input_channel_dim),
        mt.ScaleIntensityd(keys=maps, minv=-1.0, maxv=1.0, channel_wise=False),
        mt.ConcatItemsd(keys=maps, name=image_key, dim=0),
        mt.DeleteItemsd(keys=[*maps, json_key]),
        mt.ToTensord(keys=[image_key, label_key], allow_missing_keys=allow_missing_keys),
    ]

    if include_file_reading:
        transforms = [
            mt.LoadImaged(keys=maps),
            JsonKeypointLoaderd(
                json_key=json_key,
                output_key=label_key,
                keypoint_names=keypoints,
                allow_missing_keys=allow_missing_keys,
            ),
        ] + transforms

    return transforms


def default_mesh_load_transforms(
    x_resolution: int,
    y_resolution: int,
    mesh_key: str = "mesh",
    landmarks_key: str = "landmarks",
    image_key: str = "image",
    label_key: str = "label",
    include_augmentations: bool = True,
    allow_missing_keys: bool = False,
) -> list[mt.Transform]:
    transforms = [
        MeshLoaderd(keys=[mesh_key], allow_missing_keys=allow_missing_keys),
        LandmarksLoaderd(keys=[landmarks_key], allow_missing_keys=allow_missing_keys),
    ]

    if include_augmentations:
        transforms += [
            MeshRandRotated(keys=[mesh_key], mesh_key=mesh_key, landmarks_key=landmarks_key),
            MeshRandDeformed(mesh_key=mesh_key, landmarks_key=landmarks_key),
        ]

    transforms += [
        MeshRenderd(
            keys=[mesh_key],
            render_key=image_key,
            x_resolution=x_resolution,
            y_resolution=y_resolution,
        ),
        mt.SplitDimd(keys=image_key, output_postfixes=["_depth", "_norm_x", "_norm_y", "_norm_z"]),
        mt.ConcatItemsd(keys=[f"{image_key}_norm_x", f"{image_key}_norm_y", f"{image_key}_norm_z"], name="normal", dim=0),
        mt.ScaleIntensityd(keys=[f"{image_key}_depth", "normal"], minv=-1.0, maxv=1.0, channel_wise=False),
        mt.ConcatItemsd(keys=[f"{image_key}_depth", "normal"], name=image_key, dim=0),
        mt.DeleteItemsd(keys=[f"{image_key}_depth", f"{image_key}_norm_x", f"{image_key}_norm_y", f"{image_key}_norm_z", "normal"]),

        LandmarksToKeypointsd(
            landmarks_keys=landmarks_key,
            mesh_key=mesh_key,
            keypoints_key=label_key,
        ),

        mt.ToTensord(keys=[image_key, label_key], allow_missing_keys=allow_missing_keys),
    ]

    return transforms


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
        mt.RandBiasFieldd(keys=[image_key], prob=0.5, coeff_range=(0.0, 0.1)),
        mt.OneOf(
            [
                mt.RandGaussianNoised(keys=[image_key], prob=0.5, std=0.1),
                mt.OneOf([
                    mt.RandAdjustContrastd(keys=[image_key], prob=0.5, gamma=(0.5, 1.5), invert_image=False),
                    mt.RandAdjustContrastd(keys=[image_key], prob=0.5, gamma=(0.5, 1.5), invert_image=True),
                ])
            ]
        ),
    ]
