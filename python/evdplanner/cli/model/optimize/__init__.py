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
def optimize(
    anatomy: str,
    train_root: Path,
    log_dir: Path,
    n_trials: int,
    epochs: int = 100,
    test_root: Path | None = None,
    initial_config: Path | None = None,
    seed: int | None = None,
    resolution: int = 1024,
    num_workers: int = 0,
    verbose: int = 0,
):
    import json
    import logging
    from math import pi

    import arrow
    import lightning.pytorch as pl
    import optuna
    import torch
    from evdplanner.cli import set_verbosity
    from evdplanner.network.architecture import PointRegressor
    from evdplanner.network.training import train_model
    from evdplanner.network.training.datamodule import EVDPlannerDataModule
    from evdplanner.network.training.lightning_wrapper import LightningWrapper
    from evdplanner.network.training.losses import (
        MeanAbsoluteAngularError,
        MeanSquaredAngularError,
    )
    from evdplanner.network.training.utils import get_data
    from evdplanner.network.transforms.defaults import default_load_transforms
    from monai.metrics import MAEMetric, MSEMetric
    from optuna.integration import PyTorchLightningPruningCallback

    set_verbosity(verbose)
    logger = logging.getLogger(__name__)

    logger.info(f"Training on {anatomy} data.")

    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    logger.info(f"Collecting data from {train_root}.")
    train_samples, maps, keypoints = get_data(
        train_root,
        anatomy,
        output_label_key="keypoints",
    )
    if test_root:
        logger.info(f"Collecting data from {test_root}.")
        test_samples, _, _ = get_data(test_root, anatomy)
    else:
        logger.warning("No test data provided. Using validation data for testing.")
        test_samples = None

    def _objective(trial: optuna.Trial) -> float:
        if seed:
            pl.seed_everything(seed)

        batch_size = trial.suggest_int("batch_size", 1, 32)
        logger.debug(f"batch_size: {batch_size}")

        dm = EVDPlannerDataModule(
            train_samples=train_samples,
            maps=maps,
            keypoints_key="keypoints",
            test_samples=test_samples,
            load_transforms=default_load_transforms(maps, keypoints),
            batch_size=batch_size,
            num_workers=num_workers,
        )

        if anatomy == "skin":
            logger.debug("Setting ranges for skin.")
            x_range = (0.0, 2.0 * pi)
            y_range = (0.0, pi)
        elif anatomy == "ventricles":
            logger.debug("Setting ranges for ventricles.")
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
        logger.info(f"Using hp/{metrics[0].__class__.__name__} for optuna optimization.")

        logger.debug("Creating model.")
        model = LightningWrapper.from_optuna_parameters(
            model=PointRegressor,
            trial=trial,
            metrics=metrics,
            maps=maps,
            keypoints=keypoints,
            in_shape=(4, resolution // 2, resolution),
            out_shape=(len(keypoints), 2),
        )
        logger.debug(f"Setting input shape to (1, 4, {resolution // 2}, {resolution}).")
        model.set_input_shape((1, 4, resolution // 2, resolution))

        logger.info("Training model.")
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
        _objective,
        n_trials=n_trials,
        catch=[torch.cuda.OutOfMemoryError, RuntimeError],
        gc_after_trial=True,
    )

    logger.info(f"Number of finished trials: {len(study.trials)}")
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
