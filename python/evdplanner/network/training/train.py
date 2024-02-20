from pathlib import Path
from typing import Mapping

import arrow
from lightning import pytorch as pl

from evdplanner.network.training import LightningWrapper, EVDPlannerDataModule


def train_model(
        model: LightningWrapper,
        datamodule: EVDPlannerDataModule,
        log_dir: Path,
        epochs: int,
        anatomy: str,
        mode="train",
        additional_callbacks: list[pl.callbacks.Callback] = None,
        session_name: str = None,
) -> tuple[LightningWrapper, Mapping[str, float], Path]:
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
