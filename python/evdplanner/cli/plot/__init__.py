"""
Plotting commands for the CLI.
"""
import click

from .errors import errors
from .evd import evd
from .keypoints import keypoints
from .times import times


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
plot.add_command(evd)
plot.add_command(keypoints)
plot.add_command(times)
