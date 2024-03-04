"""
Preprocessing commands for EVD Planner.
"""
import click

from evdplanner.cli.preprocessing.convert_volume import convert_volume
from evdplanner.cli.preprocessing.project_mesh import project_mesh
from evdplanner.cli.preprocessing.snap_landmarks import snap_landmarks


@click.group()
def preprocess() -> None:
    """
    Preprocess data for EVD Planner.

    Returns
    -------
    None
    """
    pass


preprocess.add_command(convert_volume)
preprocess.add_command(project_mesh)
preprocess.add_command(snap_landmarks)
