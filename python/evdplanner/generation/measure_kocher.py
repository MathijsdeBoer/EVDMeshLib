from math import pi

from loguru import logger

from evdplanner.geometry import Mesh
from evdplanner.linalg import Mat4, Vec3
from evdplanner.rendering import IntersectionSort, Ray


def measure_kocher(
    mesh: Mesh,
    nasion: Vec3,
    left_ear: Vec3,
    right_ear: Vec3,
    n_angles: int = 1000,
    vertical_search_distance: float = 110.0,
    horizontal_search_distance: float = 30.0,
    max_angular_search_distance: float = 0.5 * pi,
    debug: bool = False,
) -> tuple[Vec3, Vec3] | tuple[tuple[Vec3, Vec3], list[Vec3], tuple[list[Vec3], list[Vec3]]]:
    n_vertical_samples = n_angles
    n_horizontal_samples = int(n_angles * (horizontal_search_distance / vertical_search_distance))
    logger.debug(
        f"Using {n_vertical_samples} vertical samples and {n_horizontal_samples} horizontal samples."
    )

    interaural_axis = (right_ear - left_ear).unit_vector
    forward_axis = (nasion - mesh.origin).unit_vector
    logger.debug(f"Forward axis: {forward_axis}.")
    logger.debug(f"Interaural axis: {interaural_axis}.")

    logger.info("Measuring vertical points.")
    vertical_points = _measure_surface_distance(
        mesh=mesh,
        forward_axis=forward_axis,
        rotation_axis=interaural_axis,
        initial_position=nasion,
        search_distance=vertical_search_distance,
        n_samples=n_vertical_samples,
        max_angular_search_distance=max_angular_search_distance,
    )
    logger.debug(f"Found {len(vertical_points)} vertical points.")

    up = (vertical_points[-1] - mesh.origin).unit_vector
    orthogonal = interaural_axis.cross(up).unit_vector
    logger.debug(f"Up: {up}.")
    logger.debug(f"Orthogonal: {orthogonal}.")

    logger.info("Measuring horizontal points.")
    left_horizontal_points = _measure_surface_distance(
        mesh=mesh,
        forward_axis=up,
        rotation_axis=orthogonal,
        initial_position=vertical_points[-1],
        search_distance=horizontal_search_distance,
        n_samples=n_horizontal_samples,
        max_angular_search_distance=max_angular_search_distance,
    )
    logger.debug(f"Found {len(left_horizontal_points)} left horizontal points.")

    right_horizontal_points = _measure_surface_distance(
        mesh=mesh,
        forward_axis=up,
        rotation_axis=-orthogonal,
        initial_position=vertical_points[-1],
        search_distance=horizontal_search_distance,
        n_samples=n_horizontal_samples,
        max_angular_search_distance=max_angular_search_distance,
    )
    logger.debug(f"Found {len(right_horizontal_points)} right horizontal points.")

    if debug:
        return (
            (left_horizontal_points[-1], right_horizontal_points[-1]),
            vertical_points,
            (left_horizontal_points, right_horizontal_points),
        )
    else:
        return left_horizontal_points[-1], right_horizontal_points[-1]


def _measure_surface_distance(
    mesh: Mesh,
    forward_axis: Vec3,
    rotation_axis: Vec3,
    initial_position: Vec3,
    search_distance: float,
    n_samples: int = 1000,
    max_angular_search_distance: float = 0.5 * pi,
) -> list[Vec3]:
    vertical_points = [initial_position]
    rotation_step = max_angular_search_distance / n_samples
    rotation_matrix = Mat4.rotation(rotation_axis, rotation_step)
    direction = forward_axis

    total_distance = 0.0
    total_rotation = 0.0
    for _ in range(n_samples):
        direction = direction @ rotation_matrix
        ray = Ray(mesh.origin, direction)

        intersection = mesh.intersect(ray, IntersectionSort.Nearest)
        if intersection:
            pos = intersection.position
            vertical_points.append(pos)
            total_distance += (vertical_points[-1] - vertical_points[-2]).length
        else:
            msg = f"No intersection found for vertical rotation {total_rotation}."
            logger.warning(msg)

        total_rotation += rotation_step

        if total_distance > search_distance:
            logger.debug(f"Reached search distance of {search_distance}.")
            break
        elif total_rotation > max_angular_search_distance:
            logger.debug(
                f"Reached maximum angular search distance of {max_angular_search_distance}."
            )
            break

    return vertical_points
