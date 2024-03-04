import click

from .data import data


@click.group()
def validate() -> None:
    """
    Validate the input data.
    """
    pass


validate.add_command(data)
