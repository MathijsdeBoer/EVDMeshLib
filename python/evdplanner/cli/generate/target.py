from pathlib import Path

import click
import numpy as np
from loguru import logger

from evdplanner.geometry import Mesh
from evdplanner.linalg import Mat4, Vec3
from evdplanner.rendering import Camera, CameraType, IntersectionSort, Ray


def _objective_fn(
    distance: float,
    thickness: float,
    distance_weight: float = 0.66,
    thickness_threshold: float = 10.0,
) -> float:
    return distance * distance_weight - (
        (1 - distance_weight) * max(0.0, thickness - thickness_threshold)
    )


def _ray_sample(
    mesh: Mesh,
    ray: Ray,
) -> tuple[Vec3, float, float, bool]:
    intersection = mesh.intersect(ray, IntersectionSort.Nearest)

    if not intersection:
        return Vec3.zero(), float("inf"), 0.0, False

    distance = intersection.distance
    new_ray = Ray(intersection.position + ray.direction * 1e-8, ray.direction)
    new_intersection = mesh.intersect(new_ray, IntersectionSort.Nearest)
    if not new_intersection:
        return intersection.position, distance, 0.0, False

    thickness = (intersection.position - new_intersection.position).length
    midpoint = (intersection.position + new_intersection.position) / 2

    return midpoint, distance, thickness, True


def _measure(
    mesh: Mesh,
    kp: Vec3,
    n_steps: int = 128,
    n_iter: int = 4,
    check_radially: bool = True,
    radius: float = 2.0,
    radial_samples: int = 8,
) -> Vec3:
    camera = Camera(
        origin=kp,
        forward=(mesh.origin - kp).unit_vector,
        up=Vec3(0.0, 0.0, 1.0),
        x_resolution=1,
        y_resolution=1,
        camera_type=CameraType.Equirectangular,
    )

    min_loss = float("inf")
    best_point = None

    current_x = 0.5
    current_y = 0.5
    spread = 0.25

    for i in range(n_iter):
        logger.debug(f"Iteration {i + 1}/{n_iter}.")
        logger.debug(f"Current point: ({current_x}, {current_y}).")
        logger.debug(f"Spread: {spread}.")
        for x in np.linspace(current_x - spread, current_x + spread, n_steps):
            for y in np.linspace(current_y - spread, current_y + spread, n_steps):
                ray = camera.cast_ray(x, y)
                midpoint, distance, thickness, valid = _ray_sample(mesh, ray)

                if valid:
                    objective = _objective_fn(distance, thickness)
                    if check_radially:
                        radial_distance = 0.0
                        radial_thickness = 0.0

                        for radial in range(radial_samples):
                            angle = radial / radial_samples * 2 * np.pi
                            matrix = Mat4.rotation(ray.direction, angle)

                            perpendicular = ray.direction.cross(Vec3(0.0, 0.0, 1.0)).unit_vector
                            perpendicular = perpendicular @ matrix
                            radial_origin = ray.origin + perpendicular * radius

                            radial_ray = Ray(radial_origin, ray.direction)
                            _, rd, rt, _ = _ray_sample(mesh, radial_ray)
                            radial_distance += rd
                            radial_thickness += rt

                        objective += _objective_fn(radial_distance, radial_thickness)
                    if objective < min_loss:
                        min_loss = objective
                        best_point = midpoint
                        current_x = x
                        current_y = y

        logger.debug(f"Best loss: {min_loss}.")
        logger.debug(f"Best point: {best_point}.")

        spread /= 8

    return best_point


@click.command()
@click.argument(
    "mesh",
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.argument(
    "kocher",
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.argument(
    "output",
    type=click.Path(
        dir_okay=False, file_okay=True, writable=True, resolve_path=True, path_type=Path
    ),
)
@click.option(
    "-v",
    "--verbosity",
    count=True,
    help="Increase output verbosity.",
)
def target(
    mesh: Path,
    kocher: Path,
    output: Path,
    verbosity: int = 0,
) -> None:
    """
    CLI command for measuring target points.
    """
    from evdplanner.cli import set_verbosity
    from evdplanner.geometry import Mesh
    from evdplanner.generation import find_closest_intersection
    from evdplanner.linalg import Vec3
    from evdplanner.markups import DisplaySettings, MarkupManager

    set_verbosity(verbosity)

    logger.info(f"Loading mesh from {mesh}.")
    mesh = Mesh.load(str(mesh))
    logger.info(f"Loading Kocher's point from {kocher}.")
    kocher = MarkupManager.load(kocher)

    markups = MarkupManager()

    left_kp = Vec3(*kocher.find_fiducial("Left Kocher").position)
    right_kp = Vec3(*kocher.find_fiducial("Right Kocher").position)
    logger.debug(f"Left Kocher's point: {left_kp}.")
    logger.debug(f"Right Kocher's point: {right_kp}.")

    logger.info("Measuring target points.")
    left_tp = _measure(mesh, left_kp, radius=4.0)
    right_tp = _measure(mesh, right_kp, radius=4.0)

    logger.info("Adding target points to markups.")
    markups.add_fiducial(
        label=["Left Target", "Right Target"],
        description=["Left Target", "Right Target"],
        position=[left_tp.as_float_list(), right_tp.as_float_list()],
        display=DisplaySettings(
            color=(0.0, 1.0, 0.0),
            selected_color=(0.0, 1.0, 0.0),
            active_color=(0.0, 1.0, 0.0),
        ),
    )

    logger.info("Saving target points.")
    markups.save(output)

    left_distance = (left_tp - left_kp).length
    right_distance = (right_tp - right_kp).length

    logger.info(f"Left target point: {left_tp}.")
    logger.info(f"Right target point: {right_tp}.")

    logger.info(f"Left target distance: {left_distance:.3g} mm.")
    logger.info(f"Right target distance: {right_distance:3g} mm.")

    left_closest = find_closest_intersection(mesh, left_tp)
    right_closest = find_closest_intersection(mesh, right_tp)

    logger.info(f"Left target closest wall distance: {left_closest}.")
    logger.info(f"Right target closest wall distance: {right_closest}.")
