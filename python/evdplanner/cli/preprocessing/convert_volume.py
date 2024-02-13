from pathlib import Path

import click


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
    import numpy as np
    import SimpleITK as sitk
    from evdplanner.geometry.conversion import volume_to_mesh

    print(f"Converting {volume} to {output}...")
    print("Reading volume...")
    volume = sitk.ReadImage(volume)
    origin = volume.GetOrigin()
    spacing = volume.GetSpacing()
    volume = sitk.DICOMOrient(volume, "LPS")
    volume = sitk.GetArrayFromImage(volume)

    print("Converting volume to mesh...")
    mesh = volume_to_mesh(np.swapaxes(volume, 0, -1), origin, spacing, num_samples=1_000_000)

    print("Smoothing mesh...")
    mesh.laplacian_smooth(iterations=10, smoothing_factor=0.25)

    print("Saving mesh...")
    mesh.save(str(output))
