"""
Utility functions for the network.
"""
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from imageio.v3 import imread
from loguru import logger
from monai.metrics import MAEMetric, MSEMetric
from torch import Tensor, nn, optim

from evdplanner.markups import MarkupManager
from evdplanner.network.training.losses import (
    MeanAbsoluteAngularError,
    MeanSquaredAngularError,
)
from evdplanner.network.training.lr_schedulers import PolyLRScheduler


def get_loss_fn(
    loss: nn.Module | Callable[[Tensor, Tensor], Tensor] | str
) -> nn.Module | Callable[[Tensor, Tensor], Tensor]:
    """
    Get the loss function.

    Parameters
    ----------
    loss : nn.Module | Callable[[Tensor, Tensor], Tensor] | str
        The loss function.

    Returns
    -------
    nn.Module | Callable[[Tensor, Tensor], Tensor]
        The loss function.
    """
    if isinstance(loss, str):
        match loss.lower():
            case "mae":
                return nn.L1Loss()
            case "mse":
                return nn.MSELoss()
            case "maae":
                return MeanAbsoluteAngularError()
            case "msae":
                return MeanSquaredAngularError()
            case _:
                msg = f"Unknown loss function: {loss}"
                raise ValueError(msg)
    else:
        return loss


def get_metric_fn(
    metric: nn.Module | Callable[[Tensor, Tensor], Tensor] | str
) -> nn.Module | Callable[[Tensor, Tensor], Tensor]:
    """
    Get the metric function.

    Parameters
    ----------
    metric : nn.Module | Callable[[Tensor, Tensor], Tensor] | str
        The metric function.

    Returns
    -------
    nn.Module | Callable[[Tensor, Tensor], Tensor]
        The metric function.
    """
    if isinstance(metric, str):
        match metric.lower():
            case "mae":
                return MAEMetric()
            case "mse":
                return MSEMetric()
            case "maae":
                return MeanAbsoluteAngularError()
            case "msae":
                return MeanSquaredAngularError()
            case _:
                msg = f"Unknown metric function: {metric}"
                raise ValueError(msg)
    else:
        return metric


def get_optimizer(
    optimizer: optim.Optimizer | str,
    params: Iterable[Tensor] | Iterable[dict[str, Any]],
    **kwargs: dict[str, Any],
) -> torch.optim.Optimizer:
    """
    Get the optimizer.

    Parameters
    ----------
    optimizer : optim.Optimizer | str
        The optimizer.
    params : Iterable[Tensor] | Iterable[dict[str, Any]]
        The parameters to optimize.
    kwargs : dict[str, Any]
        Additional keyword arguments for the optimizer.

    Returns
    -------
    torch.optim.Optimizer
        The optimizer.
    """
    if isinstance(optimizer, str):
        match optimizer.lower():
            case "adam":
                return optim.Adam(params, **kwargs)
            case "sgd":
                return optim.SGD(params, **kwargs)
            case _:
                msg = f"Unknown optimizer: {optimizer}"
                raise ValueError(msg)
    else:
        return optimizer


def get_lr_scheduler(
    lr_scheduler: optim.lr_scheduler.LRScheduler | str | None,
    optimizer: optim.Optimizer,
    epochs: int,
    **kwargs: Any,
) -> optim.lr_scheduler.LRScheduler | None:
    """
    Get the learning rate scheduler.

    Parameters
    ----------
    lr_scheduler : optim.lr_scheduler.LRScheduler | str | None
        The learning rate scheduler.
    optimizer : optim.Optimizer
        The optimizer.
    epochs : int
        The number of epochs.
    kwargs : dict[str, Any]
        Additional keyword arguments for the learning rate scheduler.

    Returns
    -------
    optim.lr_scheduler.LRScheduler | None
        The learning rate scheduler.
    """
    if lr_scheduler is None:
        return None

    if isinstance(lr_scheduler, str):
        match lr_scheduler.lower():
            case "step" | "steplr":
                return optim.lr_scheduler.StepLR(optimizer, **kwargs)
            case "multistep" | "multisteplr":
                return optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
            case "exponential" | "exponentiallr":
                return optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
            case "cosine" | "cosineannealing" | "cosineannealinglr":
                return optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=int(epochs * kwargs.get("t_max_ratio", 1.0)),
                    **{k: v for k, v in kwargs.items() if k != "t_max_ratio"},
                )
            case "reduce" | "reduceonplateau":
                return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
            case "poly" | "polylr":
                return PolyLRScheduler(optimizer, **kwargs)
            case _:
                msg = f"Unknown lr_scheduler {lr_scheduler}"
                raise ValueError(msg)
    else:
        return lr_scheduler


def _get_maps(
    root: Path,
    anatomy: str,
    image_files: tuple[str] = ("map_{anatomy}_depth.png", "map_{anatomy}_normal.png"),
    label_file: str = "projected_{anatomy}.kp.json",
    output_image_keys: tuple[str] = ("map_{anatomy}_depth", "map_{anatomy}_normal"),
    output_label_key: str = "keypoints",
    resolution: int = 1024,
) -> tuple[list[dict[str, Path]], list[str], list[str]]:
    data = []

    if not len(image_files) == len(output_image_keys):
        msg = "The length of 'image_files' must match the length of 'output_image_keys'."
        raise ValueError(msg)

    maps = None
    keypoints = None

    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue

        images = [
            subdir / "maps" / f"{resolution}" / file.format(anatomy=anatomy)
            for file in image_files
        ]
        label = subdir / "maps" / f"{resolution}" / label_file.format(anatomy=anatomy)

        if not all([file.exists() for file in images]) or not label.exists():
            continue

        for image in images:
            im = imread(image)
            if im.shape[1] != resolution:
                logger.warning(f"Image {image} has a different x resolution ({im.shape[1]}, {im.shape[0]}) than {resolution}.")

        if not maps:
            maps = [file.stem for file in images]

        if not keypoints:
            with label.open("r") as f:
                keypoints = json.load(f)
            keypoints = [x["label"] for x in keypoints]

        sample_dict = {
            key.format(anatomy=anatomy): file
            for key, file in zip(output_image_keys, images, strict=True)
        }
        sample_dict[output_label_key] = label
        data.append(sample_dict)

    return data, maps, keypoints


def _get_mesh(
    root: Path,
    anatomy: str,
    mesh_file: str = "mesh_{anatomy}.stl",
    landmarks_file: str = "landmarks_{anatomy}.mrk.json",
    output_mesh_key: str = "mesh",
    output_landmarks_key: str = "landmarks",
) -> tuple[list[dict[str, Path]], list[str], list[str]]:
    data = []

    landmark_names = []
    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue

        mesh = subdir / mesh_file.format(anatomy=anatomy)
        landmarks = subdir / landmarks_file.format(anatomy=anatomy)

        if len(landmark_names) == 0:
            manager = MarkupManager.load(landmarks)

            for markup in manager.markups:
                for control_point in markup.control_points:
                    landmark_names.append(control_point.label)

        if not mesh.exists() or not landmarks.exists():
            continue

        sample_dict = {
            output_mesh_key: mesh,
            output_landmarks_key: landmarks,
        }
        data.append(sample_dict)

    return data, [f"map_{anatomy}_depth", f"map_{anatomy}_normal"], landmark_names


def get_data(
    root: Path,
    anatomy: str,
    mesh_file: str = "mesh_{anatomy}.stl",
    landmarks_file: str = "landmarks_{anatomy}.mrk.json",
    output_mesh_key: str = "mesh",
    output_landmarks_key: str = "landmarks",
    use_maps: bool = False,
    image_files: tuple[str] = ("map_{anatomy}_depth.png", "map_{anatomy}_normal.png"),
    label_file: str = "projected_{anatomy}.kp.json",
    output_image_keys: tuple[str] = ("map_{anatomy}_depth", "map_{anatomy}_normal"),
    output_keypoints_key: str = "keypoints",
    resolution: int = 1024,
) -> tuple[list[dict[str, Path]], list[str], list[str]]:
    """
    Get the data.

    Parameters
    ----------
    root : Path
        The root directory.
    anatomy : str
        The anatomy.
    mesh_file : str, optional
        The mesh file, by default "mesh_{anatomy}.stl"
    landmarks_file : str, optional
        The landmarks file, by default "landmarks_{anatomy}.mrk.json"
    output_mesh_key : str, optional
        The output mesh key, by default "mesh"
    output_landmarks_key : str, optional
        The output landmarks key, by default "landmarks"
    use_maps : bool, optional
        Whether to use pre-rendered maps or not, by default False
    image_files : tuple[str], optional
        The image files, by default ("map_{anatomy}_depth.png", "map_{anatomy}_normal.png")
    label_file : str, optional
        The label file, by default "projected_{anatomy}.kp.json"
    output_image_keys : tuple[str], optional
        The output image keys, by default ("map_{anatomy}_depth", "map_{anatomy}_normal")
    output_keypoints_key : str, optional
        The output label key, by default "keypoints"
    resolution : int, optional
        The resolution, by default 1024

    Returns
    -------
    tuple[list[dict[str, Path]], list[str], list[str]]
        The data, the maps, and the keypoints.
    """
    if use_maps:
        return _get_maps(
            root,
            anatomy,
            image_files=image_files,
            label_file=label_file,
            output_image_keys=output_image_keys,
            output_label_key=output_keypoints_key,
            resolution=resolution,
        )
    else:
        return _get_mesh(
            root,
            anatomy,
            mesh_file=mesh_file,
            landmarks_file=landmarks_file,
            output_mesh_key=output_mesh_key,
            output_landmarks_key=output_landmarks_key,
        )
