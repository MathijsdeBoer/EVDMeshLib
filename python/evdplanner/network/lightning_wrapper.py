"""
The LightningWrapper class, which is a wrapper around a PyTorch model.
"""
from collections.abc import Callable
from typing import Any

import lightning.pytorch as pl
import optuna
import torch
from loguru import logger
from torch import Tensor, nn

from evdplanner.network.training.optimizable_model import OptimizableModel
from evdplanner.network.training.utils import (
    get_loss_fn,
    get_lr_scheduler,
    get_optimizer,
)


class LightningWrapper(pl.LightningModule):
    """
    A wrapper around a PyTorch model that is compatible with PyTorch Lightning.

    Attributes
    ----------
    model : nn.Module | None
        The model to wrap.
    loss : nn.Module | Callable | None
        The loss function to use.
    optimizer : torch.optim.Optimizer | None
        The optimizer to use.
    scheduler : torch.optim.lr_scheduler.LRScheduler | None
        The learning rate scheduler to use.
    metrics : list[nn.Module | Callable[[Tensor, Tensor], Tensor]]
        The metrics to use.
    config : dict[str, Any] | None
        The configuration of the model.
    loggable_hparams : list[str]
        The hyperparameters to log.
    """

    def __init__(self, model: nn.Module | OptimizableModel = None) -> None:
        """
        Initializes the LightningWrapper.
        """
        super().__init__()

        self.model: nn.Module | OptimizableModel | None = model
        self.loss: nn.Module | Callable | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.metrics = []

        self.config: dict[str, Any] | None = None
        self.loggable_hparams: list[str] = []

    def set_input_shape(self, shape: tuple[int, ...]) -> None:
        """
        Sets the input shape of the model.

        Parameters
        ----------
        shape : tuple[int, ...]
            The input shape of the model.

        Returns
        -------
        None
        """
        self.example_input_array = torch.rand(shape)

    @staticmethod
    def build_wrapper(
        model: nn.Module,
        loss: nn.Module | Callable[[Tensor, Tensor], Tensor],
        optimizer: torch.optim.Optimizer,
        config: dict[str, any] | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        metrics: list[nn.Module | Callable[[Tensor, Tensor], Tensor]] | None = None,
    ) -> "LightningWrapper":
        """
        Builds a LightningWrapper from the given parameters.

        Parameters
        ----------
        model : nn.Module
            The model to wrap.
        loss : nn.Module | Callable[[Tensor, Tensor], Tensor]
            The loss function to use.
        optimizer : torch.optim.Optimizer
            The optimizer to use.
        config : dict[str, any], optional
            The configuration of the model.
        scheduler : torch.optim.lr_scheduler.LRScheduler, optional
            The learning rate scheduler to use.
        metrics : list[nn.Module | Callable[[Tensor, Tensor], Tensor]], optional
            The metrics to use.

        Returns
        -------
        LightningWrapper
            The built LightningWrapper.
        """
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
        cls: type["LightningWrapper"],
        model: type[OptimizableModel],
        trial: optuna.Trial,
        metrics: list[nn.Module | Callable[[Tensor, Tensor], Tensor]] | None = None,
        **kwargs: dict[str, any],
    ) -> "LightningWrapper":
        """
        Builds a LightningWrapper from the given Optuna trial and model.

        Parameters
        ----------
        model : type[OptimizableModel]
            The model to build the wrapper for.
        trial : optuna.Trial
            The trial to get the parameters from.
        metrics : list[nn.Module | Callable[[Tensor, Tensor], Tensor]], optional
            The metrics to use.
        kwargs : dict[str, any]
            Additional keyword arguments to pass to the model.

        Returns
        -------
        LightningWrapper
            The built LightningWrapper.
        """
        config = model.get_optuna_parameters(trial, **kwargs)
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
                config["scheduler"], optimizer, epochs=kwargs["epochs"], **config["scheduler_args"]
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
        """
        Forwards the input through the model.

        Parameters
        ----------
        x : torch.Tensor
            The input to forward through the model.

        Returns
        -------
        torch.Tensor
            The output of the model.
        """
        return self.model(x)

    def configure_optimizers(
        self,
    ) -> dict[str, torch.optim.Optimizer | torch.optim.lr_scheduler.LRScheduler | str]:
        """
        Configures the optimizers and learning rate schedulers to use.

        Returns
        -------
        dict[str, torch.optim.Optimizer | torch.optim.lr_scheduler.LRScheduler | str]
            The optimizers and learning rate schedulers to use.
        """
        if self.scheduler is None:
            return {"optimizer": self.optimizer}
        else:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": self.scheduler,
                "monitor": "hp_metric",
            }

    def on_train_start(self) -> None:
        """
        Called when the training starts.

        Returns
        -------
        None
        """
        params = {}

        params["model"] = {}
        logger.debug("Logging hyperparameters.")
        logger.debug(f"Using {self.loggable_hparams} as loggable hyperparameters.")
        for key, value in self.config.items():
            if key in self.loggable_hparams:
                logger.debug(f"Logging hyperparameter {key} with value {value}.")
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

        self.logger.log_hyperparams(
            params, {f"hp/{x.__class__.__name__}": float("inf") for x in self.metrics}
        )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        """
        Performs a training step.

        Parameters
        ----------
        batch : dict[str, Tensor]
            The batch to train on.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        dict[str, Tensor]
            The output of the training step.
        """
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
        """
        Performs a validation step.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The batch to validate on.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        dict[str, Tensor]
            The output of the validation step.
        """
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
        """
        Performs a test step.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The batch to test on.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        dict[str, Tensor]
            The output of the test step.
        """
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

    def predict_step(self, batch: dict[str, Tensor], batch_dix: int) -> Tensor:
        return self(batch["image"])

    @property
    def log_name(self) -> str:
        """
        Returns the name of the model.

        Returns
        -------
        str
            The name of the model.
        """
        return self.model.log_name
