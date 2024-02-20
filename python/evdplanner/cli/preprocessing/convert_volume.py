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
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (can be specified multiple times)",
)
def convert_volume(
    volume: Path,
    output: Path,
    verbose: int = 0,
):
    """Convert volume to stl."""
    import logging

    import numpy as np
    import SimpleITK as sitk

    from evdplanner.cli import set_verbosity
    from evdplanner.geometry import volume_to_mesh

    set_verbosity(verbose)
    logger = logging.getLogger(__name__)

    logger.info(f"Converting {volume} to {output}...")
    logger.debug(f"Reading volume...")
    volume = sitk.ReadImage(volume)
    origin = volume.GetOrigin()
    spacing = volume.GetSpacing()
    volume = sitk.DICOMOrient(volume, "LPS")
    volume = sitk.GetArrayFromImage(volume)

    logger.debug(f"Volume shape: {volume.shape}")
    logger.debug(f"Volume origin: {origin}")
    logger.debug(f"Volume spacing: {spacing}")

    logger.info("Converting volume to mesh...")
    mesh = volume_to_mesh(np.swapaxes(volume, 0, -1), origin, spacing, num_samples=1_000_000)

    logger.info("Smoothing mesh...")
    mesh.laplacian_smooth(iterations=10, smoothing_factor=0.25)

    logger.info(f"Saving mesh to {output}...")
    mesh.save(str(output))
