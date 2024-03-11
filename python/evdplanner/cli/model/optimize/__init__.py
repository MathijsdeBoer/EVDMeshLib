"""
Optimize a point regressor model for a given anatomy using optuna.
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
    "-l",
    "--log-dir",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
    required=True,
    help="Directory to store logs.",
)
@click.option(
    "-n",
    "--n-trials",
    type=int,
    required=True,
    help="Number of trials to run.",
)
@click.option(
    "-e",
    "--epochs",
    type=int,
    default=100,
    help="Number of epochs to train.",
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
    "--initial",
    "initial_config",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
    ),
    required=False,
    help="Path to initial configuration file.",
)
@click.option(
    "--seed",
    type=int,
    required=False,
    help="Seed for random number generator.",
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
    "--use-maps",
    is_flag=True,
    help="Whether to use pre-rendered maps or not.",
)
@click.option(
    "--augmentations",
    "use_augmentations",
    is_flag=True,
    help="Whether to use augmentations or not.",
)
@click.option(
    "--final-bias",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
    ),
    required=False,
    help="Path to a file containing the initial weights for the final layer.",
)
def optimize(
    anatomy: str,
    train_root: Path,
    log_dir: Path,
    n_trials: int,
    epochs: int = 100,
    val_root: Path | None = None,
    test_root: Path | None = None,
    initial_config: Path | None = None,
    seed: int | None = None,
    num_workers: int = 0,
    verbose: int = 0,
    use_maps: bool = False,
    use_augmentations: bool = False,
    final_bias: Path | None = None,
) -> None:
    """
    Optimize a point regressor model for a given anatomy using optuna.

    Parameters
    ----------
    anatomy : str
        Anatomy to train on.
    train_root : Path
        Root directory containing training data.
    val_root : Path
        Root directory containing validation data.
    log_dir : Path
        Directory to store logs.
    n_trials : int
        Number of trials to run.
    epochs : int
        Number of epochs to train.
    test_root : Path, optional
        Root directory containing test data.
    initial_config : Path, optional
        Path to initial configuration file.
    seed : int, optional
        Seed for random number generator.
    num_workers : int, optional
        Number of workers for data loading.
    verbose : int, optional
        Increase verbosity. (Use multiple times for more verbosity.)
    use_maps : bool, optional
        Whether to use pre-rendered maps or not.
    use_augmentations : bool, optional
        Whether to use augmentations or not.
    final_bias : Path, optional
        Path to a file containing the initial weights for the final layer.

    Returns
    -------
    None
    """
    import json
    from functools import partial
    from math import pi
    from random import sample

    import arrow
    import lightning.pytorch as pl
    import optuna
    import torch
    from loguru import logger
    from monai.metrics import MAEMetric, MSEMetric
    from optuna.integration import PyTorchLightningPruningCallback

    from evdplanner.cli import set_verbosity
    from evdplanner.network.architecture import PointRegressor
    from evdplanner.network.lightning_wrapper import LightningWrapper
    from evdplanner.network.training import train_model
    from evdplanner.network.training.datamodule import EVDPlannerDataModule
    from evdplanner.network.training.losses import (
        MeanAbsoluteAngularError,
        MeanSquaredAngularError,
    )
    from evdplanner.network.training.utils import get_data
    from evdplanner.network.transforms.defaults import (
        default_augment_transforms,
        default_load_transforms,
        default_mesh_load_transforms,
    )

    set_verbosity(verbose)

    logger.info(f"Training on {anatomy} data.")

    if seed:
        pl.seed_everything(seed)

    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    def _objective(trial: optuna.Trial, final_bias: Path | None) -> float:
        """
        Objective function for optuna optimization.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object.

        Returns
        -------
        float
            The trial loss.
        """
        if seed:
            pl.seed_everything(seed)

        resolution = trial.suggest_categorical(
            "resolution", [256, 512, 1024, 2048, 4096, 8192, 16384]
        )

        logger.info(f"Collecting data from {train_root}.")
        train_samples, maps, label_names = get_data(
            train_root,
            anatomy,
            use_maps=use_maps,
            resolution=resolution,
        )

        if val_root:
            logger.info(f"Collecting data from {val_root}.")
            val_samples, _, _ = get_data(
                val_root,
                anatomy,
                use_maps=use_maps,
                resolution=resolution,
            )
        else:
            logger.warning(
                "No validation data provided. Using 20% of training data for validation."
            )
            val_samples = sample(train_samples, int(0.2 * len(train_samples)))

        if test_root:
            logger.info(f"Collecting data from {test_root}.")
            test_samples, _, _ = get_data(
                test_root,
                anatomy,
                use_maps=use_maps,
                resolution=resolution,
            )
        else:
            logger.warning("No test data provided. Using validation data for testing.")
            test_samples = None

        if final_bias and final_bias.exists():
            logger.info(f"Loading initial weights for final layer from {final_bias}.")
            with final_bias.open("r") as file:
                final_bias = json.load(file)

            bias = []
            for key in label_names:
                for b in final_bias:
                    if b["label"] == key:
                        bias.append(b["position"])
                        break
            final_bias = bias
        elif final_bias:
            logger.warning(f"File {final_bias} does not exist. Ignoring.")
            final_bias = None

        batch_size = trial.suggest_int("batch_size", 1, 32)
        logger.debug(f"batch_size: {batch_size}")

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

        dm = EVDPlannerDataModule(
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            load_transforms=default_load_transforms(maps, label_names)
            if use_maps
            else default_mesh_load_transforms(
                x_resolution=x_resolution,
                y_resolution=y_resolution,
                include_augmentations=use_augmentations,
            ),
            augment_transforms=default_augment_transforms() if use_augmentations else None,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        metrics = [
            MAEMetric(),
            MSEMetric(),
            MeanSquaredAngularError(x_range=x_range, y_range=y_range),
            MeanAbsoluteAngularError(x_range=x_range, y_range=y_range),
        ]

        logger.debug(f"Using metrics: {metrics}")
        logger.info(f"Using hp/{metrics[0].__class__.__name__} for optuna optimization.")

        logger.debug("Creating model.")
        model = LightningWrapper.from_optuna_parameters(
            model=PointRegressor,
            trial=trial,
            metrics=metrics,
            maps=maps,
            keypoints=label_names,
            epochs=epochs,
            in_shape=(4, x_resolution, y_resolution),
            out_shape=(len(label_names), 2),
            final_bias=final_bias,
        )
        logger.debug(f"Setting input shape to (1, 4, {x_resolution}, {y_resolution}).")
        model.set_input_shape((1, 4, x_resolution, y_resolution))

        logger.info("Training model.")
        try:
            model, test_loss, output_log_dir = train_model(
                model,
                dm,
                log_dir,
                epochs,
                anatomy,
                mode="optimize",
                additional_callbacks=[
                    PyTorchLightningPruningCallback(
                        trial, monitor=f"hp/{metrics[0].__class__.__name__}"
                    ),
                ],
                session_name=arrow.now().format("YYYY-MM-DD"),
            )
        except Exception:
            logger.exception("Training failed.")
            raise optuna.TrialPruned()

        model.config["batch_size"] = batch_size
        trial.set_user_attr("model_config", model.config)
        trial.set_user_attr("log_dir", output_log_dir)
        return test_loss[f"hp/{metrics[0].__class__.__name__}"]

    logger.info(f"Running {n_trials} trials.")
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=50,
        interval_steps=1,
        n_min_trials=5,
    )
    study = optuna.create_study(
        study_name=f"optuna/{anatomy}/point_regressor",
        direction="minimize",
        pruner=pruner,
    )

    if initial_config:
        logger.info(f"Enqueuing trial from {initial_config}.")
        with initial_config.open("r") as file:
            starting_parameters = json.load(file)
            study.enqueue_trial(
                starting_parameters, user_attrs={"source": f"{initial_config.name}"}
            )

    logger.info("Starting optimization.")
    study.optimize(
        partial(_objective, final_bias=final_bias),
        n_trials=n_trials,
        catch=[torch.cuda.OutOfMemoryError, RuntimeError],
        gc_after_trial=True,
    )

    logger.info(f"Number of finished trials: {len(study.trials)}")

    success_percentage = 100 * (len(study.trials) - len(study.failed_trials)) / len(study.trials)

    logger.info(
        f"Of which, {len(study.trials) - len(study.failed_trials)} "
        f"({success_percentage} %) succeeded."
    )
    logger.info("Best trial:")
    best_trial = study.best_trial
    actual_log_dir = best_trial.user_attrs["log_dir"]

    logger.info(f"Value: {best_trial.value}")
    logger.info("Params: ")
    for key, value in best_trial.params.items():
        logger.info(f"\t{key:>16}: {value}")

    logger.info(f"Writing best trial config to {actual_log_dir}.")
    with (actual_log_dir / "best_trial.optuna.json").open("w") as trial_file:
        json.dump(best_trial.params, trial_file, indent=4)

    with (actual_log_dir / "best_trial.evdplanner.json").open("w") as trial_file:
        json.dump(best_trial.user_attrs["model_config"], trial_file, indent=4)
