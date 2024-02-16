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
    "-c",
    "--config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path),
    required=True,
    help="Path to JSON configuration file.",
)
@click.option(
    "-l",
    "--log-dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path),
    required=True,
    help="Directory to store logs.",
)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path),
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
def train(
    anatomy: str,
    train_root: Path,
    config: Path,
    log_dir: Path,
    model_path: Path,
    epochs: int = 100,
    test_root: Path | None = None,
    seed: int | None = None,
    resolution: int = 1024,
    num_workers: int = 0,
) -> None:
    import json

    import lightning.pytorch as pl
    import torch
    from evdplanner.cli.model._util import get_data, train_model
    from evdplanner.network.architecture import PointRegressor
    from evdplanner.network.training.datamodule import EVDPlannerDataModule
    from evdplanner.network.training.lightning_wrapper import LightningWrapper
    from evdplanner.network.training.utils import get_loss_fn, get_optimizer
    from evdplanner.network.transforms.defaults import default_load_transforms

    if seed:
        pl.seed_everything(seed)

    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True)

    train_samples, maps, keypoints = get_data(
        train_root,
        anatomy,
        output_label_key="keypoints",
    )
    if test_root:
        test_samples, _, _ = get_data(test_root, anatomy)
    else:
        test_samples = None

    with config.open("r") as file:
        config = json.load(file)
        dm = EVDPlannerDataModule(
            train_samples=train_samples,
            maps=maps,
            keypoints_key="keypoints",
            test_samples=test_samples,
            load_transforms=default_load_transforms(maps, keypoints),
            batch_size=config["batch_size"],
            num_workers=num_workers,
        )

        core_model = PointRegressor.from_optuna_parameters(
            config,
            maps=maps,
            keypoints=keypoints,
            in_shape=(4, resolution // 2, resolution),
            out_shape=(len(keypoints), 2),
        )
        model = LightningWrapper.build_wrapper(
            model=core_model,
            loss=get_loss_fn(config["loss"]),
            optimizer=get_optimizer(
                config["optimizer"], core_model.parameters(), **config["optimizer_kwargs"]
            ),
            scheduler=None,
            metrics=None,
        )
        model.set_input_shape((1, 4, resolution // 2, resolution))

    model, test_loss = train_model(model, dm, log_dir, epochs)

    torch.save(model, model_path)
