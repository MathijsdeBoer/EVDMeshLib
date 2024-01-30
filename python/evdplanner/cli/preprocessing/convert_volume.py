from pathlib import Path

import click
import numpy as np


@click.command(name="convert-volume")
@click.argument(
    "volume",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
)
@click.argument(
    "output",
    type=click.Path(
        exists=False, dir_okay=False, writable=True, resolve_path=True, path_type=Path
    ),
)
def convert_volume(
    volume: Path,
    output: Path,
):
    """Convert volume to stl."""
    import SimpleITK as sitk
    from evdplanner.geometry.conversion import volume_to_mesh

    volume = sitk.ReadImage(volume)
    spacing = volume.GetSpacing()
    volume = sitk.GetArrayFromImage(volume)

    mesh = volume_to_mesh(np.swapaxes(volume, 0, -1), spacing)
    mesh.laplacian_smooth(iterations=16, smoothing_factor=0.1)
    mesh.save(str(output))
