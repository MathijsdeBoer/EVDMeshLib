import click

from .kocher import kocher
from .landmarks import landmarks


@click.group()
def generate():
    pass


generate.add_command(kocher)
generate.add_command(landmarks)
