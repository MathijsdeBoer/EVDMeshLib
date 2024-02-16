import click

from evdplanner.cli.model.train import train
from evdplanner.cli.model.optimize import optimize

@click.group()
def model():
    pass


model.add_command(train)
model.add_command(optimize)
