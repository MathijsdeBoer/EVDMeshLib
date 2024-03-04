from pathlib import Path

import click


@click.command()
@click.argument(
    "input_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path),
)
@click.option(
    "--anatomy",
    type=click.Choice(["skin", "ventricles"]),
    default="skin",
    help="Anatomy of the data.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (can be specified multiple times)",
)
def data(
    input_path: Path,
    anatomy: str,
    verbose: int = 0,
) -> None:
    """
    Validate data.

    Parameters
    ----------
    input_path : Path
        Path to the input directory.
    verbose : int
        Verbosity level.

    Returns
    -------
    None
    """
    from loguru import logger

    from evdplanner.cli import set_verbosity
    from evdplanner.geometry import Mesh
    from evdplanner.linalg import Vec3
    from evdplanner.markups import MarkupManager
    from evdplanner.rendering import Camera, CameraType, IntersectionSort

    set_verbosity(verbose)

    logger.info(f"Validating data in {input_path}...")
    for subdir in [x for x in input_path.iterdir() if x.is_dir()]:
        logger.info(f"Validating {subdir}...")

        landmarks = subdir / f"landmarks_{anatomy}.mrk.json"
        if not landmarks.exists():
            logger.error(f"{subdir}: Landmarks file not found: {landmarks}")
            continue

        landmarks = MarkupManager.load(landmarks)

        if anatomy == "skin":
            landmark_names = [
                "Nasion",
                "Medial Canthus Left",
                "Medial Canthus Right",
                "Lateral Canthus Left",
                "Lateral Canthus Right",
                "Pre-Auricle Left",
                "Pre-Auricle Right",
            ]
        elif anatomy == "ventricles":
            landmark_names = [
                "Foramen of Monro Left",
                "Foramen of Monro Right",
            ]
        else:
            msg = f"Invalid anatomy: {anatomy}"
            logger.error(msg)
            raise ValueError(msg)

        missing_landmark = False
        positions = []
        for name in landmark_names:
            if not landmarks.find_fiducial(name):
                logger.error(f"{subdir}: Landmark not found: {name}")
                missing_landmark = True
            else:
                logger.debug(f"{subdir}: Landmark found: {name}")
                positions.append(landmarks.find_fiducial(name).position)

        if missing_landmark:
            logger.error(f"{subdir}: Missing landmarks.")
            continue

        invalid_position = False
        for idx, (name, position) in enumerate(zip(landmark_names, positions)):
            logger.debug(f"{subdir}: Landmark {name} positions: {position}")
            if len(position) != 3:
                logger.error(f"{subdir}: Landmark {name} position has invalid length: {positions}")
                invalid_position = True
                continue
            else:
                logger.debug(f"{subdir}: Landmark {name} position has valid length: {positions}")
                positions[idx] = Vec3(*position)

        if invalid_position:
            logger.error(f"{subdir}: Invalid landmark positions.")
            continue

        mesh = subdir / f"mesh_{anatomy}.stl"
        if not mesh.exists():
            logger.error(f"{subdir}: Mesh file not found: {mesh}")
            continue

        if anatomy == "skin":
            mesh = Mesh.load(str(mesh))
            logger.debug(f"{subdir}: Mesh origin: {mesh.origin}.")
            camera = Camera(
                origin=mesh.origin,
                forward=Vec3(0.0, -1.0, 0.0),
                up=Vec3(0.0, 0.0, 1.0),
                x_resolution=1,
                y_resolution=1,
                camera_type=CameraType.Equirectangular,
            )

            projections = {}
            for name, position in zip(landmark_names, positions):
                logger.debug(f"{subdir}: Projecting landmark {name} position {position}...")
                projection = camera.project_back(position)
                projections[name] = projection

                surface_hit = mesh.intersect(camera.cast_ray(*projection), IntersectionSort.Farthest).position
                if surface_hit is None:
                    logger.error(f"{subdir}: Landmark {name} does not intersect the mesh.")
                else:
                    logger.debug(f"{subdir}: Landmark {name} intersects the mesh at {surface_hit}.")
                    if (surface_hit - position).length > 10.0:
                        logger.error(f"{subdir}: Landmark {name} is more than 1cm from the mesh.")
                    elif (surface_hit - position).length > 1.0:
                        logger.warning(f"{subdir}: Landmark {name} is more than 1mm from the mesh.")

            logger.debug(f"{subdir}: Landmark projections: {projections}")

            ref_projection = projections["Nasion"]

            medial_canthus_left = projections["Medial Canthus Left"]
            medial_canthus_right = projections["Medial Canthus Right"]
            lateral_canthus_left = projections["Lateral Canthus Left"]
            lateral_canthus_right = projections["Lateral Canthus Right"]

            if medial_canthus_left[0] > ref_projection[0]:
                logger.error(f"{subdir}: Medial Canthus Left is to the right of Nasion.")
            if medial_canthus_right[0] < ref_projection[0]:
                logger.error(f"{subdir}: Medial Canthus Right is to the left of Nasion.")
            if lateral_canthus_left[0] > ref_projection[0]:
                logger.error(f"{subdir}: Lateral Canthus Left is to the right of Nasion.")
            if lateral_canthus_right[0] < ref_projection[0]:
                logger.error(f"{subdir}: Lateral Canthus Right is to the left of Nasion.")

            if medial_canthus_left[1] < ref_projection[1]:
                logger.warning(f"{subdir}: Medial Canthus Left is above Nasion.")
            if medial_canthus_right[1] < ref_projection[1]:
                logger.warning(f"{subdir}: Medial Canthus Right is above Nasion.")

            if medial_canthus_left[0] < lateral_canthus_left[0]:
                logger.error(f"{subdir}: Medial Canthus Left is to the left of Lateral Canthus Left.")
            if medial_canthus_right[0] > lateral_canthus_right[0]:
                logger.error(f"{subdir}: Medial Canthus Right is to the right of Lateral Canthus Right.")

            left_pre_auricle = projections["Pre-Auricle Left"]
            right_pre_auricle = projections["Pre-Auricle Right"]

            if left_pre_auricle[0] > ref_projection[0]:
                logger.error(f"{subdir}: Pre-Auricle Left is to the right of Nasion.")
            if right_pre_auricle[0] < ref_projection[0]:
                logger.error(f"{subdir}: Pre-Auricle Right is to the left of Nasion.")
            if left_pre_auricle[0] > right_pre_auricle[0]:
                logger.error(f"{subdir}: Pre-Auricle Left is to the right of Pre-Auricle Right.")

            left_canthus_distance = (lateral_canthus_left[0] - medial_canthus_left[0]) ** 2 + (lateral_canthus_left[1] - medial_canthus_left[1]) ** 2
            left_canthus_distance = left_canthus_distance ** 0.5

            right_canthus_distance = (lateral_canthus_right[0] - medial_canthus_right[0]) ** 2 + (lateral_canthus_right[1] - medial_canthus_right[1]) ** 2
            right_canthus_distance = right_canthus_distance ** 0.5

            if abs(1 - left_canthus_distance / right_canthus_distance) > 0.1:
                logger.warning(f"{subdir}: Canthus distances are not within 10% of each other.")

        elif anatomy == "ventricles":
            logger.warning(f"{subdir}: Ventricles validation not implemented.")
        else:
            msg = f"Invalid anatomy: {anatomy}"
            logger.error(msg)
            raise ValueError(msg)

    logger.info("Validation complete.")
