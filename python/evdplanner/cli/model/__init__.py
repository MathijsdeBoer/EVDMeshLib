"""
Model commands.
"""

import click

from evdplanner.cli.model.optimize import optimize
from evdplanner.cli.model.train import train


@click.group()
def model() -> None:
    """
    Model commands.

    Returns
    -------
    None
    """
    pass


model.add_command(train)
model.add_command(optimize)
