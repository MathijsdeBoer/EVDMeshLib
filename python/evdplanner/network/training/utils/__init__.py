from typing import Any, Callable, Iterable

import torch
from evdplanner.network.training.losses import (
    MeanAbsoluteAngularError,
    MeanSquaredAngularError,
)
from monai.metrics import MAEMetric, MSEMetric
from torch import Tensor, nn, optim


def get_loss_fn(
    loss: nn.Module | Callable[[Tensor, Tensor], Tensor] | str
) -> Callable[[Tensor, Tensor], Tensor]:
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
) -> Callable[[Tensor, Tensor], Tensor]:
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
    **kwargs,
) -> optim.lr_scheduler.LRScheduler | None:
    if lr_scheduler is None:
        return None

    if isinstance(lr_scheduler, str):
        match lr_scheduler.lower():
            case "step":
                return optim.lr_scheduler.StepLR(optimizer, **kwargs)
            case "multistep":
                return optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
            case "exponential":
                return optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
            case "cosine":
                return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
            case "reduceonplateau":
                return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
            case _:
                msg = f"Unknown lr_scheduler {lr_scheduler}"
                raise ValueError(msg)
    else:
        return lr_scheduler
