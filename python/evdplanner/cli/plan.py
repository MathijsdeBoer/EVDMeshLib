from pathlib import Path

import click


@click.command()
@click.argument(
    "skin", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path)
)
@click.argument(
    "ventricles", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path)
)
@click.argument(
    "output", type=click.Path(exists=False, dir_okay=True, resolve_path=True, path_type=Path)
)
@click.option(
    "--skin-model",
    "skin_model",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    required=True,
    help="Path to the skin model.",
)
@click.option(
    "--ventricles-model",
    "ventricles_model",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    required=True,
    help="Path to the ventricles model.",
)
@click.option(
    "--gpu-model",
    is_flag=True,
    help="Use GPU for model inference.",
)
@click.option(
    "--gpu-render",
    is_flag=True,
    help="Use GPU for rendering.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity. (Use multiple times for more verbosity.)",
)
@click.option(
    "--write-intermediate",
    is_flag=True,
    help="Write intermediate results to file.",
)
def plan(
    skin: Path,
    ventricles: Path,
    output: Path,
    skin_model: Path,
    ventricles_model: Path,
    gpu_model: bool = False,
    gpu_render: bool = False,
    verbose: int = 0,
    write_intermediate: bool = False,
) -> None:
    """
    Plan EVD trajectory.

    Parameters
    ----------
    skin : Path
        Path to the skin mesh.
    ventricles : Path
        Path to the ventricles.
    output : Path
        Path to the output directory.
    verbose : int
        Verbosity level. 0 for ERROR, 1 for INFO, 2 for DEBUG. Any other value will also
        set DEBUG level.
    write_intermediate : bool
        Write intermediate results to file.

    Returns
    -------
    None
    """
    from loguru import logger
    from torch import device
    from torch import load as load_torch_model

    from evdplanner.cli import set_verbosity
    from evdplanner.geometry import Mesh
    from evdplanner.linalg import Vec3
    from evdplanner.network.architecture import PointRegressor
    from evdplanner.rendering import Camera, CameraType, IntersectionSort

    if gpu_render:
        from evdplanner.rendering.gpu import GPURenderer as Renderer
    else:
        from evdplanner.rendering.cpu import CPURenderer as Renderer

    # Set the verbosity level
    set_verbosity(verbose)

    # Log the input parameters
    logger.info(f"Skin mesh: {skin}")
    logger.info(f"Ventricles: {ventricles}")
    logger.info(f"Output directory: {output}")
    logger.info(f"Skin model: {skin_model}")
    logger.info(f"Ventricles model: {ventricles_model}")
    logger.info(f"Use GPU for model inference: {gpu_model}")
    logger.info(f"Use GPU for rendering: {gpu_render}")
    logger.info(f"Verbosity level: {verbose}")
    logger.info(f"Write intermediate results: {write_intermediate}")

    logger.info("Loading skin model...")
    device = device("cuda" if gpu_model else "cpu")
    skin_model: PointRegressor = load_torch_model(skin_model, map_location=device)

    # Plan the EVD trajectory
    logger.info("Loading skin mesh...")
    skin = Mesh.load(str(skin), num_samples=10_000_000)

    camera = Camera(
        origin=skin.origin,
        forward=Vec3(0.0, -1.0, 0.0),
        up=Vec3(0.0, 0.0, 1.0),
        x_resolution=skin_model.in_shape[0],
        y_resolution=skin_model.in_shape[1],
        camera_type=CameraType.Equirectangular,
    )
    renderer = Renderer(camera, skin)
    skin_render = renderer.render(IntersectionSort.Farthest)

    raise NotImplementedError("The 'plan' command is not completed yet.")
