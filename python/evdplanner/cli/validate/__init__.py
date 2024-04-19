import click

from .data import data
from .model import model


@click.group()
def validate() -> None:
    """
    Validate the input data.
    """
    pass


validate.add_command(data)
validate.add_command(model)
