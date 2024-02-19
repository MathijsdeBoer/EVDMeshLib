import click
from evdplanner.cli.plot.keypoints import keypoints


@click.group()
def plot():
    pass


plot.add_command(keypoints)
