import click

from evdplanner.cli.augment.mesh import mesh


@click.group()
def augment() -> None:
    """
    Augment data.
    """
    pass


augment.add_command(mesh)
