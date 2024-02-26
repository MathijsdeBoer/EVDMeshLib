"""
Train a model to predict the keypoint locations for a given anatomy.
"""

from pathlib import Path

import click


@click.command()
@click.option(
    "-a",
    "--anatomy",
    type=click.Choice(["skin", "ventricles"]),
    required=True,
    help="Anatomy to train on.",
)
@click.option(
    "--train",
    "train_root",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
    required=True,
    help="Root directory containing training data.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
    ),
    required=True,
    help="Path to JSON configuration file.",
)
@click.option(
    "-l",
    "--log-dir",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
    required=True,
    help="Directory to store logs.",
)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(
        exists=False, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
    ),
    required=True,
    help="Path to store trained model.",
)
@click.option(
    "-e",
    "--epochs",
    type=int,
    default=100,
    help="Number of epochs to train.",
)
@click.option(
    "--augmented",
    "augment_root",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
    required=False,
    help="Root directory containing ahead-of-time augmented data for training.",
)
@click.option(
    "--val",
    "val_root",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
    required=False,
    help="Root directory containing validation data.",
)
@click.option(
    "--test",
    "test_root",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
    required=False,
    help="Root directory containing test data.",
)
@click.option(
    "--seed",
    type=int,
    required=False,
    help="Random seed for reproducibility.",
)
@click.option(
    "--resolution",
    type=int,
    default=1024,
    help="Resolution of input images.",
)
@click.option(
    "--num-workers",
    type=int,
    default=0,
    help="Number of workers for data loading.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity. (Use multiple times for more verbosity.)",
)
@click.option(
    "--plot-progress",
    is_flag=True,
    help="Plot progress to file and TensorBoard during training.",
)
@click.option(
    "--train-augment",
    "use_augmentations",
    is_flag=True,
    help="Use augmentations during training.",
)
@click.option(
    "--final-bias",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
    ),
    required=False,
    help="Path to a file containing the initial weights for the final layer.",
)
def train(
    anatomy: str,
    train_root: Path,
    config: Path,
    log_dir: Path,
    model_path: Path,
    epochs: int = 100,
    augment_root: Path | None = None,
    val_root: Path | None = None,
    test_root: Path | None = None,
    seed: int | None = None,
    resolution: int = 1024,
    num_workers: int = 0,
    verbose: int = 0,
    plot_progress: bool = False,
    use_augmentations: bool = False,
    final_bias: Path | None = None,
) -> None:
    """
    Train a model to predict the keypoint locations for a given anatomy.

    Parameters
    ----------
    anatomy : str
        Anatomy to train on.
    train_root : Path
        Root directory containing training data.
    config : Path
        Path to JSON configuration file.
    log_dir : Path
        Directory to store logs.
    model_path : Path
        Path to store trained model.
    epochs : int
        Number of epochs to train.
    augment_root : Path, optional
        Root directory containing augmented training data.
    val_root : Path, optional
        Root directory containing validation data.
    test_root : Path, optional
        Root directory containing test data.
    seed : int, optional
        Random seed for reproducibility.
    resolution : int, optional
        Resolution of input images.
    num_workers : int, optional
        Number of workers for data loading.
    verbose : int, optional
        Increase verbosity. (Use multiple times for more verbosity.)
    plot_progress : bool, optional
        Plot progress to file and TensorBoard during training.
    use_augmentations : bool, optional
        Use augmentations during training.
    final_bias : Path, optional
        Path to a file containing the initial weights for the final layer.

    Returns
    -------
    None
    """
    import json
    from math import pi
    from random import sample

    import lightning.pytorch as pl
    import torch
    from loguru import logger
    from monai.metrics import MAEMetric, MSEMetric

    from evdplanner.cli import set_verbosity
    from evdplanner.network.architecture import PointRegressor
    from evdplanner.network.lightning_wrapper import LightningWrapper
    from evdplanner.network.training import train_model
    from evdplanner.network.training.callbacks import KeypointPlotCallback
    from evdplanner.network.training.datamodule import EVDPlannerDataModule
    from evdplanner.network.training.losses import (
        MeanAbsoluteAngularError,
        MeanSquaredAngularError,
    )
    from evdplanner.network.training.utils import (
        get_data,
        get_loss_fn,
        get_lr_scheduler,
        get_optimizer,
    )
    from evdplanner.network.transforms.defaults import (
        default_augment_transforms,
        default_load_transforms,
    )

    set_verbosity(verbose)
    logger.info(f"Training model for {anatomy}.")

    if seed:
        pl.seed_everything(seed)

    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True)

    logger.info(f"Collecting data from {train_root}.")
    train_samples, maps, keypoints = get_data(
        train_root,
        anatomy,
        output_label_key="keypoints",
    )

    if augment_root:
        logger.info(f"Collecting data from {augment_root}.")
        augment_samples, _, _ = get_data(augment_root, anatomy)
        train_samples += augment_samples

    if val_root:
        logger.info(f"Collecting data from {val_root}.")
        val_samples, _, _ = get_data(val_root, anatomy)
    else:
        logger.warning("No validation data provided. Using 20% of training data for validation.")
        val_samples = sample(train_samples, int(0.2 * len(train_samples)))
        train_samples = [s for s in train_samples if s not in val_samples]

    if test_root:
        logger.info(f"Collecting data from {test_root}.")
        test_samples, _, _ = get_data(test_root, anatomy)
    else:
        logger.info("No test data provided. Using validation data for testing.")
        test_samples = None

    if final_bias and final_bias.exists():
        logger.info(f"Loading initial weights for final layer from {final_bias}.")
        with final_bias.open("r") as file:
            final_bias = json.load(file)

        bias = []
        for key in keypoints:
            for b in final_bias:
                if b["label"] == key:
                    bias.append(b["position"])
                    break
        final_bias = bias
    elif final_bias:
        logger.warning(f"File {final_bias} does not exist. Ignoring.")
        final_bias = None

    with config.open("r") as file:
        logger.info(f"Loading configuration from {config}.")
        config = json.load(file)

        logger.debug("Configuration:")
        logger.debug(json.dumps(config, indent=4))

        dm = EVDPlannerDataModule(
            train_samples=train_samples,
            val_samples=val_samples,
            maps=maps,
            keypoints_key="keypoints",
            test_samples=test_samples,
            load_transforms=default_load_transforms(maps, keypoints),
            augment_transforms=default_augment_transforms() if use_augmentations else None,
            batch_size=config["batch_size"],
            num_workers=num_workers,
        )

        if anatomy == "skin":
            logger.debug("Setting ranges for skin.")
            x_resolution = resolution
            y_resolution = resolution // 2
            x_range = (0.0, 2.0 * pi)
            y_range = (0.0, pi)
        elif anatomy == "ventricles":
            logger.debug("Setting ranges for ventricles.")
            x_resolution = resolution
            y_resolution = resolution
            x_range = (0.0, 1.0)
            y_range = (0.0, 1.0)
        else:
            msg = f"Anatomy '{anatomy}' not recognized."
            logger.error(msg)
            raise ValueError(msg)

        metrics = [
            MAEMetric(),
            MSEMetric(),
            MeanSquaredAngularError(x_range=x_range, y_range=y_range),
            MeanAbsoluteAngularError(x_range=x_range, y_range=y_range),
        ]
        logger.debug(f"Using metrics: {metrics}")

        core_model = PointRegressor.from_optuna_parameters(
            config,
            maps=maps,
            keypoints=keypoints,
            in_shape=(4, x_resolution, y_resolution),
            out_shape=(len(keypoints), 2),
            final_bias=final_bias,
        )

        optimizer = get_optimizer(
            config["optimizer"], core_model.parameters(), **config["optimizer_args"]
        )
        if "scheduler" in config:
            scheduler = get_lr_scheduler(
                config.get("scheduler", None),
                optimizer,
                epochs=epochs,
                **config["scheduler_args"],
            )
        else:
            scheduler = None

        logger.debug("Creating model.")
        model = LightningWrapper.build_wrapper(
            model=core_model,
            loss=get_loss_fn(config["loss_fn"]),
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            config=config,
        )
        logger.debug(f"Setting input shape to (1, 4, {x_resolution}, {y_resolution}).")
        model.set_input_shape((1, 4, x_resolution, y_resolution))

        model.loggable_hparams = core_model.loggable_parameters()

    logger.info("Training model.")

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    if plot_progress:
        logger.debug("Adding plot progress callback.")
        callbacks.append(
            KeypointPlotCallback(
                filename="keypoint_plot.png",
                log_image=True,
                log_loss=True,
                log_lr=True,
            ),
        )

    model, test_loss, _ = train_model(
        model,
        dm,
        log_dir,
        epochs,
        anatomy,
        mode="train",
        additional_callbacks=callbacks,
    )

    logger.info(f"Saving model to {model_path}.")
    torch.save(model.model, model_path)

    results_path = model_path.parent / f"{model_path.stem}_test_results.json"
    logger.info(f"Saving test results to {results_path}.")
    with results_path.open("w") as file:
        json.dump(test_loss, file, indent=4)
