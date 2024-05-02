from pathlib import Path

import click
import numpy as np

from evdplanner.rs import Camera, CameraType


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
    default=256,
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
    default=1.5,
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
    "--generate-line",
    is_flag=True,
    help="Generate line between Kocher's and target points.",
)
@click.option(
    "--line-thickness",
    type=float,
    default=3.0,
    show_default=True,
    help="Thickness of the line.",
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
    steps: int = 256,
    iterations: int = 3,
    check_radially: bool = True,
    radius: float = 1.5,
    radial_samples: int = 8,
    radial_rings: int = 2,
    generate_line: bool = False,
    line_thickness: float = 3.0,
    verbosity: int = 0,
) -> None:
    """
    CLI command for measuring target points.
    """
    from time import time

    import SimpleITK as sitk
    from loguru import logger

    from evdplanner.cli import set_verbosity
    from evdplanner.generation import find_closest_intersection
    from evdplanner.geometry import Mesh
    from evdplanner.linalg import Vec3
    from evdplanner.markups import DisplaySettings, MarkupManager
    from evdplanner.rendering import find_target, generate_objective_image

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
    mesh = Mesh.load(str(mesh), num_samples=100_000_000)
    logger.info(f"Loading Kocher's point from {kocher}.")
    kocher = MarkupManager.load(kocher)

    markups = MarkupManager()

    left_kp = Vec3(*kocher.find_fiducial("Left Kocher").position)
    right_kp = Vec3(*kocher.find_fiducial("Right Kocher").position)
    logger.debug(f"Left Kocher's point: {left_kp}.")
    logger.debug(f"Right Kocher's point: {right_kp}.")

    # Empirically determined weights and thresholds.
    thickness_threshold = 10.0
    depth_threshold = 80.0
    if check_radially:
        objective_distance_weight = 0.75
    else:
        objective_distance_weight = 0.66

    logger.debug(f"Objective distance weight: {objective_distance_weight}.")
    logger.debug(f"Thickness threshold: {thickness_threshold} mm.")
    logger.debug(f"Depth threshold: {depth_threshold} mm.")

    logger.info("Measuring target points.")
    left_tp, left_loss = find_target(
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
        depth_threshold=depth_threshold,
    )
    right_tp, right_loss = find_target(
        mesh,
        right_kp,
        n_steps=steps,
        n_iter=iterations,
        check_radially=check_radially,
        radius=radius,
        radial_samples=radial_samples,
        radial_rings=radial_rings,
        objective_distance_weight=objective_distance_weight,
        thickness_threshold=thickness_threshold,
        depth_threshold=depth_threshold,
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

    if generate_line:
        logger.info("Generating line.")
        evd_display = DisplaySettings(
            color=(0.35, 0.55, 0.85),
            selected_color=(0.35, 0.55, 0.85),
            active_color=(0.35, 0.55, 0.85),
        )

        evd_display.glyph_size = line_thickness
        evd_display.use_glyph_scale = False
        evd_display.text_scale = 2.0
        evd_display.line_thickness = 1.0

        markups = MarkupManager()
        markups.add_line(
            label=("Left Kocher", "Left Target"),
            description=("Left Kocher", "Left Target"),
            position=(left_kp.as_float_list(), left_tp.as_float_list()),
            display=evd_display,
            visible_points=True,
        )

        markups.add_line(
            label=("Right Kocher", "Right Target"),
            description=("Right Kocher", "Right Target"),
            position=(right_kp.as_float_list(), right_tp.as_float_list()),
            display=evd_display,
            visible_points=True,
        )

        markups.save(output.parent / f"EVD.mrk.json")

    if verbosity > 0:
        left_distance = (left_tp - left_kp).length
        right_distance = (right_tp - right_kp).length

        logger.info(f"Left target point: {left_tp}. loss: {left_loss:.3g}.")
        logger.info(f"Right target point: {right_tp}. loss: {right_loss:.3g}.")

        logger.info(f"Left target distance: {left_distance:.3g} mm.")
        logger.info(f"Right target distance: {right_distance:.3g} mm.")

        if verbosity > 2:
            logger.info("Measuring closest wall distances.")
            left_closest = find_closest_intersection(mesh, left_tp)
            right_closest = find_closest_intersection(mesh, right_tp)

            logger.info(
                f"Left target closest wall distance: {(left_closest - left_tp).length} mm."
            )
            logger.info(
                f"Right target closest wall distance: {(right_closest - right_tp).length} mm."
            )

            left_forward = (mesh.origin - left_kp).unit_vector
            right_forward = (mesh.origin - right_kp).unit_vector
            for iteration, fov in enumerate([45.0 / 2**i for i in range(iterations)]):
                logger.info(f"Rendering with FOV: {fov:.3g}.")
                left_camera = Camera(
                    origin=left_kp,
                    forward=left_forward,
                    up=Vec3(0.0, 0.0, 1.0),
                    x_resolution=steps,
                    y_resolution=steps,
                    fov=fov,
                    camera_type=CameraType.Perspective,
                )
                right_camera = Camera(
                    origin=right_kp,
                    forward=right_forward,
                    up=Vec3(0.0, 0.0, 1.0),
                    x_resolution=steps,
                    y_resolution=steps,
                    fov=fov,
                    camera_type=CameraType.Perspective,
                )

                logger.debug(f"{left_forward=}")
                logger.debug(f"{right_forward=}")

                iter_radius = radius * (iteration / (iterations - 1))
                logger.debug(f"{iter_radius=}")

                logger.debug("Generating objective images.")
                left_objective_image = generate_objective_image(
                    mesh=mesh,
                    camera=left_camera,
                    check_radially=check_radially and iteration > 0,
                    radius=iter_radius,
                    radial_samples=radial_samples,
                    radial_rings=radial_rings,
                    objective_distance_weight=objective_distance_weight,
                    thickness_threshold=thickness_threshold,
                    depth_threshold=depth_threshold,
                    epsilon=1e-8,
                )
                right_objective_image = generate_objective_image(
                    mesh=mesh,
                    camera=right_camera,
                    check_radially=check_radially and iteration > 0,
                    radius=iter_radius,
                    radial_samples=radial_samples,
                    radial_rings=radial_rings,
                    objective_distance_weight=objective_distance_weight,
                    thickness_threshold=thickness_threshold,
                    depth_threshold=depth_threshold,
                    epsilon=1e-8,
                )

                logger.debug("Finding best objective points.")
                logger.debug(f"{np.min(left_objective_image[..., 0])=}")
                logger.debug(f"{np.max(left_objective_image[..., 0])=}")
                logger.debug(f"{np.min(right_objective_image[..., 0])=}")
                logger.debug(f"{np.max(right_objective_image[..., 0])=}")
                left_best = np.unravel_index(
                    np.argmin(left_objective_image[..., 0]), left_objective_image[..., 0].shape
                )
                right_best = np.unravel_index(
                    np.argmin(right_objective_image[..., 0]), right_objective_image[..., 0].shape
                )
                logger.debug(f"Left best: {left_best}.")
                logger.debug(f"Right best: {right_best}.")

                logger.debug("Setting camera forward vectors.")
                left_forward = left_camera.cast_ray(*left_best[::-1]).direction
                right_forward = right_camera.cast_ray(*right_best[::-1]).direction

                logger.debug("Post-processing objective images.")
                left_penalty = left_objective_image[..., -2] + (
                    left_objective_image[..., -1] * 0.5
                )
                right_penalty = right_objective_image[..., -2] + (
                    right_objective_image[..., -1] * 0.5
                )

                left_penalty -= depth_threshold
                right_penalty -= depth_threshold

                left_penalty[left_penalty < 1] = 1
                right_penalty[right_penalty < 1] = 1

                left_penalty **= 2
                right_penalty **= 2

                left_objective_image = np.concatenate(
                    [left_objective_image, left_penalty[..., None]], axis=-1
                )
                right_objective_image = np.concatenate(
                    [right_objective_image, right_penalty[..., None]], axis=-1
                )

                left_objective_image = np.expand_dims(left_objective_image, axis=-2)
                right_objective_image = np.expand_dims(right_objective_image, axis=-2)

                left_objective_image = sitk.GetImageFromArray(left_objective_image)
                right_objective_image = sitk.GetImageFromArray(right_objective_image)

                logger.debug("Writing objective images.")
                sitk.WriteImage(
                    left_objective_image, output.parent / f"ObjectiveImageLeft_{fov:.3g}.nii.gz"
                )
                sitk.WriteImage(
                    right_objective_image, output.parent / f"ObjectiveImageRight_{fov:.3g}.nii.gz"
                )

    logger.info(f"Elapsed time: {time() - start:.3g} s.")
