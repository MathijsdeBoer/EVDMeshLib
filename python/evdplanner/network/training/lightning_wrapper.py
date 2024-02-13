from typing import Callable

import lightning.pytorch as pl
import optuna
import torch
from evdplanner.network.training.optimizable_model import OptimizableModel
from evdplanner.network.training.utils import (
    get_loss_fn,
    get_lr_scheduler,
    get_optimizer,
)
from torch import Tensor, nn


class LightningWrapper(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.model = None
        self.loss = None
        self.optimizer = None
        self.scheduler = None
        self.metrics = []

        self.config = None
        self.loggable_hparams = []

    @staticmethod
    def build_wrapper(
        model: nn.Module,
        loss: nn.Module | Callable[[Tensor, Tensor], Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        metrics: list[nn.Module | Callable[[Tensor, Tensor], Tensor]] | None = None,
        config: dict[str, any] | None = None,
    ) -> "LightningWrapper":
        wrapper = LightningWrapper()
        wrapper.model = model
        wrapper.loss = loss
        wrapper.optimizer = optimizer
        wrapper.scheduler = scheduler
        wrapper.metrics = metrics or []
        wrapper.config = config or wrapper.hparams
        return wrapper

    @classmethod
    def from_optuna_parameters(
        cls,
        model: type[OptimizableModel],
        trial: optuna.Trial,
        metrics: list[nn.Module | Callable[[Tensor, Tensor], Tensor]] | None = None,
    ) -> "LightningWrapper":
        params = model.get_optuna_parameters(trial)
        loggable_params = model.loggable_parameters()
        model = model.from_optuna_parameters(params)
        loss = get_loss_fn(params["loss"])
        optimizer = get_optimizer(
            params["optimizer"], model.parameters(), **params["optimizer_kwargs"]
        )

        if params.get("scheduler", None) is None:
            scheduler = None
        else:
            scheduler = get_lr_scheduler(
                params["scheduler"], optimizer, **params["scheduler_kwargs"]
            )

        wrapper = cls.build_wrapper(
            model,
            loss,
            optimizer,
            scheduler,
            metrics,
            config=params,
        )

        wrapper.loggable_hparams = loggable_params
        return wrapper

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(
        self,
    ) -> dict[str, torch.optim.Optimizer | torch.optim.lr_scheduler.LRScheduler | str]:
        if self.lr_scheduler is None:
            return {"optimizer": self.optimizer}
        else:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": self.lr_scheduler,
                "monitor": self.lr_scheduler_metric,
            }

    def on_train_start(self) -> None:
        params = {
            key: value
            for key, value in self.config.items()
            if not key.startswith("hp/") and key in self.loggable_parameters
        }

        params["optimizer"] = self.optimizer.__class__.__name__
        params["optimizer_args"] = self.optimizer.defaults

        if self.scheduler is not None:
            params["scheduler"] = self.scheduler.__class__.__name__
            params["scheduler_args"] = self.scheduler.defaults

        self.logger.log_hyperparams(
            params, {f"hp/{x.__class__.__name__}": float("inf") for x in self.metrics}
        )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        output = {"loss": loss}

        for metric in self.metrics:
            name = f"{metric.__class__.__name__}"
            value = metric(y_hat, y).mean()
            output[f"metric/train/{name}"] = value

        self.log_dict(
            output,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.shape[0],
        )

        return output

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, Tensor]:
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        output = {
            "val_loss": loss,
        }

        for metric in self.metrics:
            name = f"{metric.__class__.__name__}"
            value = metric(y_hat, y).mean()
            output[f"metric/val/{name}"] = value

        self.log_dict(output, prog_bar=True, batch_size=x.shape[0], on_step=False, on_epoch=True)

        output["prediction"] = y_hat
        self.log("hp_metric", loss)
        return output

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, Tensor]:
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        output = {
            "test_loss": loss,
        }

        for metric in self.metrics:
            name = f"{metric.__class__.__name__}"
            value = metric(y_hat, y).mean()
            output[f"metric/test/{name}"] = value
            self.log(f"hp/{name}", value)

        self.log_dict(output, prog_bar=True, batch_size=x.shape[0], on_step=False, on_epoch=True)
        return output
