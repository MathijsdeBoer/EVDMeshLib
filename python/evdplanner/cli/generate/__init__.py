"""
CLI commands for generating EVD plans.
"""
import click

from .kocher import kocher
from .landmarks import landmarks


@click.group()
def generate() -> None:
    """
    Generate EVD plans.
    """
    pass


generate.add_command(kocher)
generate.add_command(landmarks)
