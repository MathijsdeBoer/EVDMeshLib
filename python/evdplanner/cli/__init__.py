import click
from evdplanner.cli.preprocessing import preprocess


@click.group()
def cli():
    """EVD Planner CLI."""
    pass


cli.add_command(preprocess)
