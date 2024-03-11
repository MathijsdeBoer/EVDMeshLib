from pathlib import Path

import click


@click.command()
@click.option(
    "-m",
    "--mesh",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    required=True,
    help="Path to the mesh file.",
)
@click.option(
    "-l",
    "--landmarks",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    required=True,
    help="Path to the landmarks file.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (can be specified multiple times)",
)
def snap_landmarks(
    mesh: Path,
    landmarks: Path,
    verbose: int = 0,
) -> None:
    """
    Snap landmarks to a mesh.

    Parameters
    ----------
    mesh : Path
        Path to the mesh file.
    landmarks : Path
        Path to the landmarks file.
    verbose : int
        Verbosity level.

    Returns
    -------
    None
    """
    import numpy as np
    from loguru import logger

    from evdplanner.cli import set_verbosity
    from evdplanner.geometry import Mesh
    from evdplanner.linalg import Vec3
    from evdplanner.markups import MarkupManager
    from evdplanner.rendering import Camera, CameraType, IntersectionSort

    set_verbosity(verbose)

    logger.info(f"Snapping landmarks to mesh for {mesh.parent.name}...")
    logger.info("Reading mesh...")
    mesh = Mesh.load(str(mesh))

    logger.info("Reading landmarks...")
    original_path = landmarks

    logger.info("Creating backup...")
    backup_path = landmarks.name.split(".")[0] + "_backup." + landmarks.name.split(".")[1]
    landmarks = MarkupManager.load(landmarks)
    landmarks.save(original_path.parent / backup_path)

    search_resolution = 512
    logger.debug(f"Search resolution: {search_resolution}")

    logger.info("Snapping landmarks...")
    new_manager = MarkupManager()
    for markup in landmarks.markups:
        new_control_points = []
        logger.debug(f"Snapping {markup.markup_type}...")
        for control_point in markup.control_points:
            position = Vec3(*control_point.position)
            logger.debug(f"Snapping {position}...")

            camera = Camera(
                origin=position,
                forward=mesh.origin - position,
                up=Vec3(0.0, 0.0, 1.0),
                camera_type=CameraType.Equirectangular,
                x_resolution=1,
                y_resolution=1,
            )

            hits = np.zeros((search_resolution, search_resolution), dtype=float)
            for i in range(search_resolution):
                for j in range(search_resolution):
                    u = i / (search_resolution - 1)
                    v = j / (search_resolution - 1)
                    ray = camera.cast_ray(u, v)
                    intersection = mesh.intersect(ray, IntersectionSort.Nearest)
                    if intersection:
                        hits[i, j] = intersection.distance
                    else:
                        hits[i, j] = np.inf

            i, j = np.unravel_index(np.argmin(hits, axis=None), hits.shape)
            u = i / (search_resolution - 1)
            v = j / (search_resolution - 1)
            ray = camera.cast_ray(u, v)
            new_control_points.append(
                mesh.intersect(ray, IntersectionSort.Nearest).position.as_float_list()
            )

        new_manager._add_item(
            markup.markup_type,
            [control_point.description for control_point in markup.control_points],
            [control_point.label for control_point in markup.control_points],
            new_control_points,
            display=markup.display_settings,
            visible_points=[control_point.visible for control_point in markup.control_points][0],
        )

    new_manager.save(original_path)
    logger.info(f"Landmarks snapped to {mesh}.")
