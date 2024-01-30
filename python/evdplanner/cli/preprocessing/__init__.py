import click
from evdplanner.cli.preprocessing.convert_volume import convert_volume
from evdplanner.cli.preprocessing.project_mesh import project_mesh


@click.group()
def preprocess():
    """Preprocess data for EVD Planner."""
    pass


preprocess.add_command(convert_volume)
preprocess.add_command(project_mesh)
