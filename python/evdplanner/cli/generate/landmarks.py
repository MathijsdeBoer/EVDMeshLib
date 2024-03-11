from pathlib import Path

import click


@click.command()
@click.option(
    "-a",
    "--anatomy",
    type=click.Choice(["skin", "ventricles"], case_sensitive=False),
    required=True,
    help="Anatomy to generate landmarks for.",
)
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
    "projection",
    type=click.Path(
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
    "--kocher",
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    required=False,
    help="Path to Kocher's points.",
)
@click.option(
    "--side",
    type=click.Choice(["left", "right"], case_sensitive=False),
    required=False,
    help="Side to generate landmarks for. Leave empty to generate for both sides.",
)
@click.option(
    "-v", "--verbose", count=True, help="Increase output verbosity. Repeat for more verbosity."
)
def landmarks(
    anatomy: str,
    mesh: Path,
    projection: Path,
    output: Path,
    kocher: Path | None = None,
    side: str | None = None,
    verbose: int = 0,
) -> None:
    import json

    from loguru import logger

    from evdplanner.cli import set_verbosity
    from evdplanner.generation import generate_landmarks
    from evdplanner.geometry import Mesh
    from evdplanner.linalg import Vec3
    from evdplanner.markups import DisplaySettings, MarkupManager
    from evdplanner.rendering import Camera, CameraType, IntersectionSort

    set_verbosity(verbose)

    logger.info("Loading mesh...")
    mesh = Mesh.load(str(mesh), num_samples=10_000_000)

    logger.info("Loading projection...")
    with projection.open("r") as f:
        projection = json.load(f)
    points = {x["label"]: x["position"] for x in projection}
    logger.debug(f"Projection: {points}")

    manager = MarkupManager()
    fiducials_to_add = []

    match anatomy:
        case "skin":
            logger.info("Generating landmarks for skin.")
            logger.info("Creating camera...")
            camera = Camera(
                origin=mesh.origin,
                forward=Vec3(0.0, -1.0, 0.0),
                up=Vec3(0.0, 0.0, 1.0),
                x_resolution=1,
                y_resolution=1,
                camera_type=CameraType.Equirectangular,
            )

            logger.info("Generating landmarks...")
            lm = generate_landmarks(
                projection=[(point[0], point[1]) for point in points.values()],
                mesh=mesh,
                camera=camera,
                intersection_sort=IntersectionSort.Farthest,
            )

            for point, label in zip(lm, points.keys(), strict=True):
                logger.debug(f"Adding {label} at {point}.")
                fiducials_to_add.append((label, point))
        case "ventricles":
            logger.info("Generating landmarks for ventricles.")
            if not kocher:
                msg = "Kocher's points are required for ventricles."
                logger.error(msg)
                raise ValueError(msg)

            logger.info("Loading Kocher's points...")
            kocher = MarkupManager.load(kocher)

            if not side:
                sides = ["left", "right"]
            else:
                sides = [side]
            logger.debug(f"Sides: {sides}.")

            for current_side in sides:
                if current_side not in {"left", "right"}:
                    msg = f"Invalid side: {current_side}."
                    logger.error(msg)
                    raise ValueError(msg)
                logger.info(f"Generating landmarks for {current_side} ventricle.")

                kp = Vec3(*kocher.find_fiducial(f"{current_side.capitalize()} Kocher").position)
                logger.debug(f"Kocher's point: {kp}.")

                logger.info("Creating camera...")
                camera = Camera(
                    origin=kp,
                    forward=mesh.origin - kp,
                    up=Vec3(0.0, 0.0, 1.0),
                    x_resolution=1,
                    y_resolution=1,
                    camera_type=CameraType.Equirectangular,
                )

                logger.debug(f"Finding {current_side} points.")
                current_points = {
                    k: v for k, v in points.items() if k.startswith(current_side.capitalize())
                }
                logger.debug(f"{current_side.capitalize()} points: {current_points}.")

                logger.info("Generating landmarks...")
                lm = generate_landmarks(
                    projection=[(point[0], point[1]) for point in current_points.values()],
                    mesh=mesh,
                    camera=camera,
                    intersection_sort=IntersectionSort.Farthest,
                )

                for point, label in zip(lm, current_points.keys(), strict=True):
                    logger.debug(f"Adding {label} at {point}.")
                    fiducials_to_add.append((label, point))
        case _:
            msg = f"Invalid anatomy: {anatomy}."
            logger.error(msg)
            raise ValueError(msg)

    logger.info("Adding fiducials to markup manager.")
    manager.add_fiducial(
        label=[x[0] for x in fiducials_to_add],
        description=[x[0] for x in fiducials_to_add],
        position=[x[1].as_float_list() for x in fiducials_to_add],
        display=DisplaySettings(
            color=(0.0, 1.0, 0.0),
            selected_color=(0.0, 1.0, 0.0),
            active_color=(0.0, 1.0, 0.0),
        ),
        visible_points=True,
    )

    logger.info("Saving markup manager.")
    manager.save(output)
