"""
CLI command for measuring Kocher points.
"""

from pathlib import Path

import click


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
    "landmarks",
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
    "--n-angles", type=int, default=1000, help="Number of angles to use for the measurement."
)
@click.option("--debug", is_flag=True, help="Enable debug mode.")
@click.option(
    "-v",
    "--verbosity",
    count=True,
    help="Increase output verbosity.",
)
def kocher(
    mesh: Path,
    landmarks: Path,
    output: Path,
    n_angles: int = 1000,
    debug: bool = False,
    verbosity: int = 0,
) -> None:
    """
    Measure Kocher points on a mesh using landmarks.
    """
    from loguru import logger

    from evdplanner.cli import set_verbosity
    from evdplanner.generation import measure_kocher
    from evdplanner.geometry import Mesh
    from evdplanner.linalg import Vec3
    from evdplanner.markups import DisplaySettings, MarkupManager

    set_verbosity(verbosity)

    logger.info(f"Loading mesh from {mesh}.")
    mesh = Mesh.load(str(mesh))
    logger.info(f"Loading landmarks from {landmarks}.")
    landmarks = MarkupManager.load(landmarks)

    nasion = Vec3(*landmarks.find_fiducial("Nasion").position)
    left_ear = Vec3(*landmarks.find_fiducial("Pre-Auricle Left").position)
    right_ear = Vec3(*landmarks.find_fiducial("Pre-Auricle Right").position)

    logger.info("Measuring Kocher points.")
    result = measure_kocher(
        mesh=mesh,
        nasion=nasion,
        left_ear=left_ear,
        right_ear=right_ear,
        n_angles=n_angles,
        debug=debug,
    )

    logger.info("Preparing markups.")
    markups = MarkupManager()

    fiducial_style = DisplaySettings(
        color=(0.0, 1.0, 0.0),
        selected_color=(0.0, 1.0, 0.0),
        active_color=(0.0, 1.0, 0.0),
    )

    if debug:
        logger.info("Adding debug markups.")
        (
            (left_kocher, right_kocher),
            vertical_points,
            (left_horizontal_points, right_horizontal_points),
        ) = result

        logger.info("Adding Kocher points.")
        logger.debug(f"Left Kocher: {left_kocher}.")
        logger.debug(f"Right Kocher: {right_kocher}.")
        markups.add_fiducial(
            label=["Left Kocher", "Right Kocher"],
            description=["Left Kocher", "Right Kocher"],
            position=[left_kocher.as_float_list(), right_kocher.as_float_list()],
            display=fiducial_style,
        )

        curve_style = DisplaySettings(
            color=(1.0, 0.0, 0.0),
            selected_color=(1.0, 0.0, 0.0),
            active_color=(1.0, 0.0, 0.0),
        )
        curve_style.point_labels_visibility = False

        logger.info("Adding vertical points.")
        logger.debug(f"Number of vertical points: {len(vertical_points)}.")
        markups.add_curve(
            label=[f"vp{i}" for i in range(len(vertical_points))],
            description=[f"Vertical Point {i}" for i in range(len(vertical_points))],
            position=[point.as_float_list() for point in vertical_points],
            display=curve_style,
            visible_points=False,
        )

        logger.info("Adding horizontal points.")
        logger.debug(f"Number of left horizontal points: {len(left_horizontal_points)}.")
        markups.add_curve(
            label=[f"lh{i}" for i in range(len(left_horizontal_points))],
            description=[f"Left Horizontal Point {i}" for i in range(len(left_horizontal_points))],
            position=[point.as_float_list() for point in left_horizontal_points],
            display=curve_style,
            visible_points=False,
        )

        logger.debug(f"Number of right horizontal points: {len(right_horizontal_points)}.")
        markups.add_curve(
            label=[f"rh{i}" for i in range(len(right_horizontal_points))],
            description=[
                f"Right Horizontal Point {i}" for i in range(len(right_horizontal_points))
            ],
            position=[point.as_float_list() for point in right_horizontal_points],
            display=curve_style,
            visible_points=False,
        )
    else:
        logger.info("Adding Kocher points.")
        left_kocher, right_kocher = result
        logger.debug(f"Left Kocher: {left_kocher}.")
        logger.debug(f"Right Kocher: {right_kocher}.")
        markups.add_fiducial(
            label=["Left Kocher", "Right Kocher"],
            description=["Left Kocher", "Right Kocher"],
            position=[left_kocher.as_float_list(), right_kocher.as_float_list()],
            display=fiducial_style,
        )

    logger.info(f"Saving markups to {output}.")
    markups.save(output)
