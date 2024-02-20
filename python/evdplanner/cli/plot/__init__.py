"""
Plotting commands for the CLI.
"""
import click

from evdplanner.cli.plot.keypoints import keypoints


@click.group()
def plot() -> None:
    """
    Plotting commands for the CLI.

    Returns
    -------
    None
    """
    pass


plot.add_command(keypoints)
