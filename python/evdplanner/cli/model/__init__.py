import click
from evdplanner.cli.model.optimize import optimize
from evdplanner.cli.model.train import train


@click.group()
def model():
    pass


model.add_command(train)
model.add_command(optimize)
