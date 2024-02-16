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
