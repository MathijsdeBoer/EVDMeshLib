from pathlib import Path

import click
from evdplanner.network.training.utils import get_loss_fn, get_optimizer


@click.command()
def train(
    anatomy: str,
    train_root: Path,
    config: Path,
    log_dir: Path,
    test_root: Path | None = None,
) -> None:
    import json

    import lightning.pytorch as pl
    from evdplanner.cli.model._util import get_data
    from evdplanner.network.architecture import PointRegressor
    from evdplanner.network.training.datamodule import EVDPlannerDataModule
    from evdplanner.network.training.lightning_wrapper import LightningWrapper

    train_samples = get_data(train_root, anatomy)
    test_samples = get_data(test_root, anatomy) if test_root else None

    dm = EVDPlannerDataModule(
        train_samples=train_samples,
        test_samples=test_samples,
        batch_size=1,
        num_workers=0,
    )

    with config.open("r") as file:
        config = json.load(file)
        core_model = PointRegressor.from_optuna_parameters(
            config,
            in_shape=dm.in_shape,
            out_shape=dm.out_shape,
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
