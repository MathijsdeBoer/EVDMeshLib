import sys

import click
from evdplanner.cli.model import model
from evdplanner.cli.plot import plot
from evdplanner.cli.preprocessing import preprocess
from loguru import logger


@click.group()
def cli():
    """EVD Planner CLI."""
    pass


cli.add_command(model)
cli.add_command(plot)
cli.add_command(preprocess)


def set_verbosity(level: int) -> None:
    logger.remove()
    match level:
        case 0:
            logger.add(sys.stderr, level="ERROR")
        case 1:
            logger.add(sys.stderr, level="INFO")
        case 2:
            logger.add(sys.stderr, level="DEBUG")
        case _:
            logger.add(sys.stderr, level="DEBUG")
