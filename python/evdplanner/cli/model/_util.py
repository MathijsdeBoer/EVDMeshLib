import json
from pathlib import Path
from typing import Mapping

import arrow
import lightning.pytorch as pl
from evdplanner.network.training.datamodule import EVDPlannerDataModule
from evdplanner.network.training.lightning_wrapper import LightningWrapper


def get_data(
    root: Path,
    anatomy: str,
    image_files: tuple[str] = ("map_{anatomy}_depth.png", "map_{anatomy}_normal.png"),
    label_file: str = "projected_{anatomy}.kp.json",
    output_image_keys: tuple[str] = ("map_{anatomy}_depth", "map_{anatomy}_normal"),
    output_label_key: str = "keypoints",
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

        images = [subdir / file.format(anatomy=anatomy) for file in image_files]
        label = subdir / label_file.format(anatomy=anatomy)

        if not all([file.exists() for file in images]) or not label.exists():
            continue

        if not maps:
            maps = [file.stem for file in images]

        if not keypoints:
            with label.open("r") as f:
                keypoints = json.load(f)
            keypoints = [x["label"] for x in keypoints]

        sample_dict = {
            key.format(anatomy=anatomy): file for key, file in zip(output_image_keys, images)
        }
        sample_dict[output_label_key] = label
        data.append(sample_dict)

    return data, maps, keypoints


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
