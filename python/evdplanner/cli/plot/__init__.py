"""
Plotting commands for the CLI.
"""
import click

from .errors import errors
from .keypoints import keypoints


@click.group()
def plot() -> None:
    """
    Plotting commands for the CLI.

    Returns
    -------
    None
    """
    pass


plot.add_command(errors)
plot.add_command(keypoints)
