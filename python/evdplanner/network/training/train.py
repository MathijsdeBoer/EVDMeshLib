"""
The training function for the EVDPlanner model.
"""
from pathlib import Path
from typing import Mapping

import arrow
from lightning import pytorch as pl

from evdplanner.network.training import EVDPlannerDataModule, LightningWrapper


def train_model(
    model: LightningWrapper,
    datamodule: EVDPlannerDataModule,
    log_dir: Path,
    epochs: int,
    anatomy: str,
    mode: str = "train",
    additional_callbacks: list[pl.callbacks.Callback] = None,
    session_name: str = None,
) -> tuple[LightningWrapper, Mapping[str, float], Path]:
    """
    Train the model using the provided data module and log the results to the specified directory.

    Parameters
    ----------
    model : LightningWrapper
        The model to train.
    datamodule : EVDPlannerDataModule
        The data module to use for training.
    log_dir : Path
        The directory to log the results to.
    epochs : int
        The number of epochs to train for.
    anatomy : str
        The anatomy the model is being trained on.
    mode : str, optional
        The mode the model is being trained in. Default is "train".
    additional_callbacks : list[pl.callbacks.Callback], optional
        Additional callbacks to use during training.
    session_name : str, optional
        The name of the session to log. Default is None.

    Returns
    -------
    tuple[LightningWrapper, Mapping[str, float], Path]
        The trained model, the results of the training, and the path to the log directory.
    """
    from lightning.pytorch import loggers

    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor="val_loss"),
    ] + (additional_callbacks or [])

    if session_name:
        name = f"{mode}/{anatomy}/{model.log_name}/{session_name}"
    else:
        name = f"{mode}/{anatomy}/{model.log_name}/{arrow.now().format('YYYY-MM-DD')}"

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        max_epochs=epochs,
        logger=loggers.TensorBoardLogger(
            save_dir=log_dir,
            name=name,
            log_graph=True,
        ),
        log_every_n_steps=1,
        precision="16-mixed",
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=datamodule)

    if datamodule.test_data:
        result = trainer.test(model, datamodule=datamodule)
    else:
        result = trainer.validate(model, datamodule=datamodule)

    return model, result[0], log_dir / name
