from typing import Callable

import lightning.pytorch as pl
import optuna
import torch
from torch import Tensor, nn

from evdplanner.network.training.optimizable_model import OptimizableModel
from evdplanner.network.training.utils import get_loss_fn, get_optimizer, get_lr_scheduler


class LightningWrapper(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.model = None
        self.loss = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.metrics = []

        self.config = None
        self.loggable_hparams = []

    def set_input_shape(self, shape: tuple[int, ...]) -> None:
        self.example_input_array = torch.rand(shape)

    @staticmethod
    def build_wrapper(
        model: nn.Module,
        loss: nn.Module | Callable[[Tensor, Tensor], Tensor],
        optimizer: torch.optim.Optimizer,
        config: dict[str, any] | None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        metrics: list[nn.Module | Callable[[Tensor, Tensor], Tensor]] | None = None,
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
        **kwargs,
    ) -> "LightningWrapper":
        config = model.get_optuna_parameters(trial)
        loggable_params = model.loggable_parameters()
        model = model.from_optuna_parameters(config, **kwargs)
        loss = get_loss_fn(config["loss_fn"])
        optimizer = get_optimizer(
            config["optimizer"], model.parameters(), **config["optimizer_args"]
        )

        if config.get("scheduler", None) is None:
            scheduler = None
        else:
            scheduler = get_lr_scheduler(
                config["scheduler"], optimizer, **config["scheduler_args"]
            )

        wrapper = cls.build_wrapper(
            model,
            loss,
            optimizer,
            config,
            scheduler,
            metrics,
        )

        wrapper.loggable_hparams = loggable_params
        return wrapper

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(
        self,
    ) -> dict[str, torch.optim.Optimizer | torch.optim.lr_scheduler.LRScheduler | str]:
        if self.scheduler is None:
            return {"optimizer": self.optimizer}
        else:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": self.scheduler,
            }

    def on_train_start(self) -> None:
        params = {}

        params["model"] = {}
        for key, value in self.config.items():
            if key in self.loggable_hparams:
                params["model"][key] = value
        params["model"]["name"] = self.model.__class__.__name__
        params["model"]["input_shape"] = self.example_input_array.shape[1:]
        params["model"]["output_shape"] = self.model.out_shape
        params["model"]["maps"] = self.model.maps
        params["model"]["keypoints"] = self.model.keypoints

        params["loss"] = self.loss.__class__.__name__

        params["optimizer"] = self.optimizer.defaults
        params["optimizer"]["name"] = self.optimizer.__class__.__name__

        if self.scheduler is not None:
            params["scheduler"] = {}
            params["scheduler"]["name"] = self.scheduler.__class__.__name__

            for key, value in self.scheduler.__dict__.items():
                if key in self.loggable_hparams:
                    params["scheduler"][key] = value

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
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.shape[0],
        )

        return output

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, Tensor]:
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        output = {
            "val_loss": loss,
        }

        for metric in self.metrics:
            name = f"{metric.__class__.__name__}"
            value = metric(y_hat, y).mean()
            output[f"metric/val/{name}"] = value
            self.log(f"hp/{name}", value)

        self.log_dict(output, prog_bar=True, batch_size=x.shape[0], on_step=False, on_epoch=True)

        output["prediction"] = y_hat
        self.log("hp_metric", loss)
        return output

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, Tensor]:
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = self.loss(y_hat, y)

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

    @property
    def log_name(self) -> str:
        return self.model.log_name
