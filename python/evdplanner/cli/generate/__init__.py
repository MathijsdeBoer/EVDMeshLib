import click

from .kocher import kocher


@click.group()
def generate():
    pass


generate.add_command(kocher)
