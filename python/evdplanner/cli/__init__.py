"""
CLI functions for the EVD Planner module.
"""

import sys

import click
from loguru import logger

from evdplanner.cli.augment import augment
from evdplanner.cli.generate import generate
from evdplanner.cli.model import model
from evdplanner.cli.plan import plan
from evdplanner.cli.plot import plot
from evdplanner.cli.preprocessing import preprocess
from evdplanner.cli.validate import validate


@click.group()
def cli() -> None:
    """
    EVD Planner CLI.

    This is the main entry point for the command line interface. It groups the commands
    that can be executed.

    Returns
    -------
    None
    """
    pass


# Add the commands to the CLI group
cli.add_command(augment)
cli.add_command(generate)
cli.add_command(model)
cli.add_command(plan)
cli.add_command(plot)
cli.add_command(preprocess)
cli.add_command(validate)


def set_verbosity(level: int) -> None:
    """
    Set the verbosity level for the logger.

    Parameters
    ----------
    level : int
        The verbosity level. 0 for ERROR, 1 for INFO, 2 for DEBUG. Any other
        value will also set DEBUG level.

    Returns
    -------
    None
    """
    logger.remove()
    match level:
        case 0:
            # Add a handler for error level messages
            logger.add(sys.stderr, level="ERROR")
        case 1:
            # Add a handler for info level messages
            logger.add(sys.stderr, level="INFO")
        case 2:
            # Add a handler for debug level messages
            logger.add(sys.stderr, level="DEBUG")
        case _:
            # Add a handler for debug level messages for any other value
            logger.add(sys.stderr, level="DEBUG")
