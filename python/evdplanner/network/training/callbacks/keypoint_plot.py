"""
Callback to log the loss and a sample image with the predicted keypoints.
"""
from pathlib import Path

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


class KeypointPlotCallback(pl.Callback):
    """
    Callback that logs the loss and a sample image with the predicted keypoints.
    """

    def __init__(
        self,
        filename: str = "keypoint_plot.png",
        log_image: bool = True,
        log_loss: bool = True,
        log_lr: bool = False,
    ) -> None:
        """
        Initialize the callback.

        Parameters
        ----------
        filename : str
            The filename to save the plot to.
        log_image : bool
            Whether to log the image or not.
        log_loss : bool
            Whether to log the loss or not.
        log_lr : bool
            Whether to log the learning rate or not.
            May only be set to True if log_loss is also True.
        """
        super().__init__()

        if log_lr and not log_loss:
            msg = "Cannot log learning rate without logging loss."
            raise ValueError(msg)

        self.log_image = log_image
        self.log_loss = log_loss
        self.log_lr = log_lr
        self.output_filename = filename

        self._train_batches = []
        self._val_batches = []
        self._losses = {
            "value": [],
            "step": [],
            "epoch": [],
            "stage": [],
            "lr": [],
        }

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """
        Log the loss and the learning rate at the end of each training batch.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer object.
        pl_module : pl.LightningModule
            The lightning module.
        outputs : dict[str, torch.Tensor]
            The outputs of the model.
        batch : dict[str, torch.Tensor]
            The current batch.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        None
        """
        train_step = {
            "step": trainer.global_step,
            "epoch": trainer.current_epoch,
        }

        if self.log_loss:
            train_step["train_loss"] = outputs["loss"].cpu().numpy().item()

        self._train_batches.append(train_step)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, torch.Tensor] | None,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Log the loss and the image at the end of each validation batch.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer object.
        pl_module : pl.LightningModule
            The lightning module.
        outputs : dict[str, torch.Tensor], optional
            The outputs of the model.
        batch : dict[str, torch.Tensor]
            The current batch.
        batch_idx : int
            The index of the current batch.
        dataloader_idx : int, optional
            The index of the dataloader.

        Returns
        -------
        None
        """
        if trainer.sanity_checking:
            return

        val_step = {
            "step": trainer.global_step,
            "epoch": trainer.current_epoch,
        }

        if self.log_image:
            val_step["image"] = batch["image"]
            val_step["label"] = batch["label"]
            val_step["prediction"] = outputs["prediction"]

        if self.log_loss:
            val_loss = outputs.get("val_loss", torch.tensor(float("nan")))
            val_step["val_loss"] = val_loss.cpu().numpy().item()

        self._val_batches.append(val_step)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Log the loss and the image at the end of each validation epoch.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer object.
        pl_module : pl.LightningModule
            The lightning module.

        Returns
        -------
        None
        """
        if trainer.sanity_checking:
            return

        logger = pl_module.logger.experiment

        # Aggregate all validation losses
        if self.log_loss:
            for batch in self._train_batches:
                self._losses["value"].append(batch["train_loss"])
                self._losses["stage"].append("train")
                self._losses["step"].append(batch["step"])
                self._losses["epoch"].append(batch["epoch"])
                self._losses["lr"].append(trainer.optimizers[0].param_groups[0]["lr"])

            for batch in self._val_batches:
                self._losses["value"].append(batch["val_loss"])
                self._losses["stage"].append("val")
                self._losses["step"].append(batch["step"])
                self._losses["epoch"].append(batch["epoch"])
                self._losses["lr"].append(trainer.optimizers[0].param_groups[0]["lr"])

        sns.set_theme(context="paper", style="dark")
        sns.despine(left=True, bottom=True)

        if self.log_image and self.log_loss:
            fig, ax = plt.subplots(2, 1, figsize=(5, 5), dpi=300)

            self._log_image_fn(ax[0])
            self._log_loss_fn(ax[1])
        elif self.log_image:
            fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
            self._log_image_fn(ax)
        elif self.log_loss:
            fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
            self._log_loss_fn(ax)
        else:
            msg = "Nothing to log"
            raise RuntimeError(msg)

        logger.add_figure(
            tag=f"validation/{pl_module.log_name}",
            figure=fig,
            global_step=pl_module.global_step,
            close=False,
        )

        plot_dir = Path(logger.log_dir) / "plots"
        if not plot_dir.exists():
            plot_dir.mkdir(parents=True)

        plt.savefig(plot_dir / f"step_{pl_module.global_step:>09}.png")
        plt.close("all")

        self._train_batches = []
        self._val_batches = []

    def _log_loss_fn(self, ax: plt.Axes) -> None:
        sns.lineplot(
            data=self._losses,
            x="epoch",
            y="value",
            hue="stage",
            ax=ax,
            err_style="band",
            errorbar=("ci", 95),
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_xlim(0, None)
        ax.set_yscale("log")

        if self.log_lr:
            ax2 = ax.twinx()
            sns.lineplot(
                data=self._losses,
                x="epoch",
                y="lr",
                ax=ax2,
                color="black",
                alpha=0.5,
                label="Learning Rate",
            )
            ax2.set_ylabel("Learning Rate")
            ax2.set_yscale("log")

    def _log_image_fn(self, ax: plt.Axes) -> None:
        x = torch.cat([batch["image"] for batch in self._val_batches], dim=0)
        y = torch.cat([batch["label"] for batch in self._val_batches], dim=0)
        y_hat = torch.cat([batch["prediction"] for batch in self._val_batches], dim=0)

        image = np.flipud(np.transpose(x[0, 0].cpu().numpy()))
        image_x = image.shape[1]
        image_y = image.shape[0]

        # Because our image origin is in the upper left corner, but matplotlib's origin is in
        # the lower left corner,
        # we have to do some magic to get the coordinates right.
        # invert y coordinates for plotting
        y[:, :, 1] = 1.0 - y[:, :, 1]
        y_hat[:, :, 1] = 1.0 - y_hat[:, :, 1]

        ax.imshow(
            image,
            cmap="gray",
            origin="upper",
        )

        for keypoint in range(y.shape[1]):
            sns.lineplot(
                x=[
                    y[0, keypoint, 0].cpu().numpy() * image_x,
                    y_hat[0, keypoint, 0].cpu().numpy() * image_x,
                ],
                y=[
                    y[0, keypoint, 1].cpu().numpy() * image_y,
                    y_hat[0, keypoint, 1].cpu().numpy() * image_y,
                ],
                ax=ax,
            )

        sns.scatterplot(
            x=y[0, :, 0].cpu().numpy() * image_x,
            y=y[0, :, 1].cpu().numpy() * image_y,
            ax=ax,
            markers="x",
        )
        sns.scatterplot(
            x=y_hat[0, :, 0].cpu().numpy() * image_x,
            y=y_hat[0, :, 1].cpu().numpy() * image_y,
            ax=ax,
            markers="o",
        )

        ax.set_xlim(0, image_x)
        ax.set_ylim(0, image_y)
        ax.axis("off")
