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
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path),
    required=True,
    help="Root directory containing training data.",
)
@click.option(
    "-l",
    "--log-dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path),
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
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path),
    required=False,
    help="Root directory containing test data.",
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
def optimize(
    anatomy: str,
    train_root: Path,
    log_dir: Path,
    n_trials: int,
    epochs: int = 100,
    test_root: Path | None = None,
    seed: int | None = None,
    resolution: int = 1024,
    num_workers: int = 0,
):
    import json

    import lightning.pytorch as pl
    import optuna
    import torch
    from monai.metrics import MSEMetric, MAEMetric

    from evdplanner.cli.model._util import get_data, train_model
    from evdplanner.network.architecture import PointRegressor
    from evdplanner.network.training.datamodule import EVDPlannerDataModule
    from evdplanner.network.training.lightning_wrapper import LightningWrapper
    from evdplanner.network.training.losses import MeanSquaredAngularError, MeanAbsoluteAngularError
    from evdplanner.network.training.utils import get_loss_fn, get_optimizer
    from evdplanner.network.transforms.defaults import default_load_transforms

    if seed:
        pl.seed_everything(seed)

    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    train_samples, maps, keypoints = get_data(
        train_root,
        anatomy,
        output_label_key="keypoints",
    )
    if test_root:
        test_samples, _, _ = get_data(test_root, anatomy)
    else:
        test_samples = None

    def _objective(trial: optuna.Trial) -> float:
        dm = EVDPlannerDataModule(
            train_samples=train_samples,
            maps=maps,
            keypoints_key="keypoints",
            test_samples=test_samples,
            load_transforms=default_load_transforms(maps, keypoints),
            batch_size=trial.suggest_int("batch_size", 1, 32),
            num_workers=num_workers,
        )

        config = PointRegressor.get_optuna_parameters(trial)
        core_model = PointRegressor.from_optuna_parameters(
            config,
            maps=maps,
            keypoints=keypoints,
            in_shape=(4, resolution // 2, resolution),
            out_shape=(len(keypoints), 2),
        )

        model = LightningWrapper.build_wrapper(
            model=core_model,
            loss=get_loss_fn(config["loss_fn"]),
            optimizer=get_optimizer(
                config["optimizer"], core_model.parameters(), **config["optimizer_args"]
            ),
            scheduler=None,
            metrics=[
                MSEMetric(),
                MAEMetric(),
                MeanSquaredAngularError(),
                MeanAbsoluteAngularError(),
            ],
        )
        model.set_input_shape((1, 4, resolution // 2, resolution))

        model, test_loss = train_model(model, dm, log_dir, epochs)
        return test_loss

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=50,
        interval_steps=1,
        n_min_trials=5,
    )
    study = optuna.create_study(
        study_name="optuna/point_regressor",
        direction="minimize",
        pruner=pruner,
    )
    study.optimize(
        _objective,
        n_trials=n_trials,
        catch=[torch.cuda.OutOfMemoryError],
        gc_after_trial=True,
    )

    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    best_trial = study.best_trial

    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key:>16}: {value}")

    with (log_dir / "best_trial.json").open("w") as file:
        json.dump(best_trial.params, file, indent=4)
