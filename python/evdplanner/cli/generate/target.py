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
    distance_weight: float = 0.5,
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
    n_iter: int = 3,
    check_radially: bool = True,
    radius: float = 2.0,
    radial_samples: int = 8,
    radial_rings: int = 2,
    objective_distance_weight: float = 0.5,
    thickness_threshold: float = 10.0,
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
                    objective = _objective_fn(distance, thickness, objective_distance_weight, thickness_threshold)
                    if check_radially:
                        radial_distance = 0.0
                        radial_thickness = 0.0

                        for ring in range(radial_rings):
                            for radial in range(radial_samples):
                                ring_radius = radius * (ring + 1) / radial_rings
                                angle = radial / radial_samples * 2 * np.pi
                                matrix = Mat4.rotation(ray.direction, angle)

                                perpendicular = ray.direction.cross(Vec3(0.0, 0.0, 1.0)).unit_vector
                                perpendicular = perpendicular @ matrix
                                radial_origin = ray.origin + perpendicular * ring_radius

                                radial_ray = Ray(radial_origin, ray.direction)
                                _, rd, rt, _ = _ray_sample(mesh, radial_ray)
                                radial_distance += rd
                                radial_thickness += rt

                        objective += _objective_fn(radial_distance, radial_thickness, 1.0 - objective_distance_weight, thickness_threshold)
                    if objective < min_loss:
                        min_loss = objective
                        best_point = midpoint
                        current_x = x
                        current_y = y

        logger.debug(f"Best loss: {min_loss}.")
        logger.debug(f"Best point: {best_point}.")

        spread /= 4

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
    "-s",
    "--steps",
    type=int,
    default=128,
    show_default=True,
    help="Number of steps per iteration.",
)
@click.option(
    "-i",
    "--iterations",
    type=int,
    default=3,
    show_default=True,
    help="Number of iterations.",
)
@click.option(
    "--radial",
    "check_radially",
    is_flag=True,
    help="Check radial distance and thickness.",
)
@click.option(
    "--radius",
    type=float,
    default=4.0,
    show_default=True,
    help="Radius for radial distance and thickness.",
)
@click.option(
    "--radial-samples",
    type=int,
    default=8,
    show_default=True,
    help="Number of radial samples.",
)
@click.option(
    "--radial-rings",
    type=int,
    default=2,
    show_default=True,
    help="Number of radial rings.",
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
    steps: int = 128,
    iterations: int = 3,
    check_radially: bool = True,
    radius: float = 4.0,
    radial_samples: int = 8,
    radial_rings: int = 2,
    verbosity: int = 0,
) -> None:
    """
    CLI command for measuring target points.
    """
    from time import time

    from evdplanner.cli import set_verbosity
    from evdplanner.geometry import Mesh
    from evdplanner.generation import find_closest_intersection
    from evdplanner.linalg import Vec3
    from evdplanner.markups import DisplaySettings, MarkupManager

    start = time()
    set_verbosity(verbosity)
    logger.debug(f"mesh: {mesh}")
    logger.debug(f"kocher: {kocher}")
    logger.debug(f"output: {output}")
    logger.debug(f"steps: {steps}")
    logger.debug(f"iterations: {iterations}")
    logger.debug(f"check_radially: {check_radially}")
    if check_radially:
        logger.debug(f"radius: {radius}")
        logger.debug(f"radial_samples: {radial_samples}")
        logger.debug(f"radial_rings: {radial_rings}")

    logger.info(f"Loading mesh from {mesh}.")
    mesh = Mesh.load(str(mesh))
    logger.info(f"Loading Kocher's point from {kocher}.")
    kocher = MarkupManager.load(kocher)

    markups = MarkupManager()

    left_kp = Vec3(*kocher.find_fiducial("Left Kocher").position)
    right_kp = Vec3(*kocher.find_fiducial("Right Kocher").position)
    logger.debug(f"Left Kocher's point: {left_kp}.")
    logger.debug(f"Right Kocher's point: {right_kp}.")

    # Empirically determined weights and thresholds.
    if check_radially:
        objective_distance_weight = 0.5
        thickness_threshold = 10.0
    else:
        objective_distance_weight = 0.66
        thickness_threshold = 10.0

    logger.debug(f"Objective distance weight: {objective_distance_weight}.")
    logger.debug(f"Thickness threshold: {thickness_threshold} mm.")

    logger.info("Measuring target points.")
    left_tp = _measure(
        mesh,
        left_kp,
        n_steps=steps,
        n_iter=iterations,
        check_radially=check_radially,
        radius=radius,
        radial_samples=radial_samples,
        radial_rings=radial_rings,
        objective_distance_weight=objective_distance_weight,
        thickness_threshold=thickness_threshold,
    )
    right_tp = _measure(
        mesh,
        right_kp,
        n_steps=steps,
        n_iter=iterations,
        check_radially=check_radially,
        radius=radius,
        radial_samples=radial_samples,
        radial_rings=radial_rings
    )

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

    if verbosity > 0:
        left_distance = (left_tp - left_kp).length
        right_distance = (right_tp - right_kp).length

        logger.info(f"Left target point: {left_tp}.")
        logger.info(f"Right target point: {right_tp}.")

        logger.info(f"Left target distance: {left_distance:.3g} mm.")
        logger.info(f"Right target distance: {right_distance:3g} mm.")

        if verbosity > 1:
            left_closest = find_closest_intersection(mesh, left_tp)
            right_closest = find_closest_intersection(mesh, right_tp)

            logger.info(f"Left target closest wall distance: {(left_closest - left_tp).length} mm.")
            logger.info(f"Right target closest wall distance: {(right_closest - right_tp).length} mm.")

    logger.info(f"Elapsed time: {time() - start:.3g} s.")
