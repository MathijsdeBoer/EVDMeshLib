import logging

import click
from evdplanner.cli.model import model
from evdplanner.cli.plot import plot
from evdplanner.cli.preprocessing import preprocess


@click.group()
def cli():
    """EVD Planner CLI."""
    pass


cli.add_command(model)
cli.add_command(plot)
cli.add_command(preprocess)


def set_verbosity(level: int) -> None:
    match level:
        case 0:
            logging.basicConfig(level=logging.WARNING)
        case 1:
            logging.basicConfig(level=logging.INFO)
        case 2:
            logging.basicConfig(level=logging.DEBUG)
        case _:
            logging.basicConfig(level=logging.DEBUG)
