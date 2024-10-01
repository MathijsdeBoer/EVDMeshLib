from pathlib import Path

import click

from evdplanner.rendering.utils import spherical_project


def plan(
        skin: Path,
        skin_model: Path,
        output: Path,
        ventricles: Path | None = None,
        gpu_model: bool = False,
        write_intermediate: bool = False,
        return_subtimes: bool = False,
) -> dict | None:
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
    from time import time

    from loguru import logger
    from monai.transforms import Compose
    from torch import device, load, no_grad, transpose

    from evdplanner.generation import generate_landmarks, measure_kocher
    from evdplanner.geometry import Mesh
    from evdplanner.linalg import Vec3
    from evdplanner.markups import DisplaySettings, MarkupManager
    from evdplanner.network.architecture import PointRegressor
    from evdplanner.network.transforms import default_load_transforms
    from evdplanner.rendering import (
        Camera,
        CameraType,
        IntersectionSort,
        find_target,
    )
    from evdplanner.rendering.utils import normalize_image

    start = time()
    time_dict = {
        "read": {},
        "processing": {},
        "write": {},
    }

    if not output.exists():
        output.mkdir(parents=True)

    logger.info("Loading skin model...")
    model_start_time = time()
    device = device("cuda" if gpu_model else "cpu")
    skin_model: PointRegressor = load(str(skin_model), map_location=device)
    skin_model.eval()
    time_dict["read"]["Model"] = time() - model_start_time

    logger.debug("Skin model:")
    logger.debug(f"Input shape: {skin_model.in_shape}")
    logger.debug(f"Output shape: {skin_model.out_shape}")
    logger.debug(f"Keypoints: {skin_model.keypoints}")

    # Plan the EVD trajectory
    logger.info("Loading skin mesh...")
    skin_start_time = time()
    skin = Mesh.load(str(skin), num_samples=10_000_000)
    time_dict["read"]["Skin Mesh"] = time() - skin_start_time

    logger.info("Projecting skin...")
    skin_processing_start_time = time()
    skin_depth, skin_normal = spherical_project(skin, skin_model.in_shape)
    skin_depth = normalize_image(skin_depth)
    time_dict["processing"]["Spherical Projection"] = time() - skin_processing_start_time

    if write_intermediate:
        from imageio.v3 import imwrite

        logger.info("Writing intermediate results...")
        intermediate_image_start_time = time()
        skin_depth_path = output / "map_skin_depth.png"
        skin_normal_path = output / "map_skin_normal.png"
        imwrite(skin_depth_path, (skin_depth * 65535).astype("uint16"))
        imwrite(skin_normal_path, (((skin_normal + 1) / 2) * 255).astype("uint8"))
        time_dict["write"]["Intermediate Images"] = time() - intermediate_image_start_time

    transform = Compose(
        default_load_transforms(
            maps=["depth", "normal"],
            image_key="image",
            include_file_reading=False,
            allow_missing_keys=True,
            input_channel_dim=-1,
        )
    )

    skin_image_transform_start_time = time()
    skin_input = transform({"depth": skin_depth[..., None], "normal": skin_normal})["image"][
        None, ...
    ]
    skin_input = transpose(skin_input, -1, -2)
    logger.debug(f"Skin input {skin_input.shape}")
    time_dict["processing"]["Spherical Projection Transform"] = time() - skin_image_transform_start_time

    logger.info("Predicting skin landmarks...")
    landmark_prediction_start_time = time()
    with no_grad():
        skin_input = skin_input.to(device)
        skin_landmarks = skin_model(skin_input).squeeze().cpu().numpy()
    time_dict["processing"]["Landmarks Prediction"] = time() - landmark_prediction_start_time

    for keypoint, pred in zip(skin_model.keypoints, skin_landmarks, strict=True):
        logger.debug(f"{keypoint}: {pred}")

    if write_intermediate:
        logger.info("Writing intermediate results...")
        import json

        intermediate_keypoints_start_time = time()
        skin_landmarks_path = output / "projected_skin.kp.json"
        with skin_landmarks_path.open("w") as f:
            json.dump(skin_landmarks.tolist(), f)
        time_dict["write"]["Intermediate Landmarks"] = time() - intermediate_keypoints_start_time

    logger.info("Generating landmarks for skin.")
    logger.info("Creating camera...")
    projecting_landmarks_start_time = time()
    camera = Camera(
        origin=skin.origin,
        forward=Vec3(0.0, -1.0, 0.0),
        up=Vec3(0.0, 0.0, 1.0),
        x_resolution=1,
        y_resolution=1,
        camera_type=CameraType.Equirectangular,
    )

    logger.info("Generating landmarks...")
    lm = generate_landmarks(
        projection=[(point[0], point[1]) for point in skin_landmarks],
        mesh=skin,
        camera=camera,
        intersection_sort=IntersectionSort.Farthest,
    )
    time_dict["processing"]["Projecting Landmarks"] = time() - projecting_landmarks_start_time

    for point, label in zip(lm, skin_model.keypoints, strict=True):
        logger.debug(f"Found {label} at {point}.")

    if write_intermediate:
        logger.info("Writing intermediate results...")
        intermediate_landmarks_start_time = time()
        manager = MarkupManager()
        manager.add_fiducial(
            label=skin_model.keypoints,
            description=skin_model.keypoints,
            position=[(point.x, point.y, point.z) for point in lm],
            display=DisplaySettings(
                color=(1, 0, 0),
                selected_color=(1, 0, 0),
                active_color=(1, 0, 0),
            ),
            visible_points=True,
        )
        manager.save(output / "landmarks_skin_predicted.mrk.json")
        time_dict["write"]["Intermediate Landmarks"] = time() - intermediate_landmarks_start_time

    nasion = lm[skin_model.keypoints.index("Nasion")]
    preauricle_left = lm[skin_model.keypoints.index("Pre-Auricle Left")]
    preauricle_right = lm[skin_model.keypoints.index("Pre-Auricle Right")]

    logger.info("Measuring Kocher's points.")
    kp_start_time = time()
    left_kp, right_kp = measure_kocher(
        mesh=skin,
        nasion=nasion,
        left_ear=preauricle_left,
        right_ear=preauricle_right,
        n_angles=1000,
        debug=False,
    )
    time_dict["processing"]["Kocher's Point"] = time() - kp_start_time

    logger.debug(f"Left Kocher's point: {left_kp}.")
    logger.debug(f"Right Kocher's point: {right_kp}.")

    if write_intermediate:
        logger.info("Writing intermediate results...")
        intermediate_kp_start_time = time()
        manager = MarkupManager()
        manager.add_fiducial(
            label=["Left Kocher", "Right Kocher"],
            description=["Left Kocher", "Right Kocher"],
            position=[left_kp.as_float_list(), right_kp.as_float_list()],
            display=DisplaySettings(
                color=(0, 1, 0),
                selected_color=(0, 1, 0),
                active_color=(0, 1, 0),
            ),
            visible_points=True,
        )
        manager.save(output / "kocher_predicted.mrk.json")
        time_dict["write"]["Intermediate Kocher's Points"] = time() - intermediate_kp_start_time

    logger.info("Planning EVD trajectory.")
    logger.info("Loading ventricles...")
    ventricles_start_time = time()
    ventricles = Mesh.load(str(ventricles), num_samples=10_000_000)
    time_dict["read"]["Ventricles Mesh"] = time() - ventricles_start_time

    logger.info("Finding targets...")
    logger.debug("Left target...")
    target_start_time = time()
    left_tgt, _ = find_target(
        mesh=ventricles,
        origin=left_kp,
        n_steps=256,
        n_iter=3,
        check_radially=True,
        radius=1.5,
        radial_rings=2,
        radial_samples=8,
        objective_distance_weight=0.75,
        thickness_threshold=10.0,
        depth_threshold=80.0,
    )
    logger.debug("Right target...")
    right_tgt, _ = find_target(
        mesh=ventricles,
        origin=right_kp,
        n_steps=256,
        n_iter=3,
        check_radially=True,
        radius=1.5,
        radial_rings=2,
        radial_samples=8,
        objective_distance_weight=0.75,
        thickness_threshold=10.0,
        depth_threshold=80.0,
    )
    time_dict["processing"]["Target Point Search"] = time() - target_start_time

    logger.debug(f"Left target: {left_tgt}.")
    logger.debug(f"Right target: {right_tgt}.")

    logger.info("Creating markups...")
    markup_start_time = time()
    evd_display = DisplaySettings(
        color=(0.35, 0.55, 0.85),
        selected_color=(0.35, 0.55, 0.85),
        active_color=(0.35, 0.55, 0.85),
    )
    evd_display.glyph_size = 3.0
    evd_display.use_glyph_scale = False
    evd_display.text_scale = 2.0
    evd_display.line_thickness = 1.0

    evd_markup = MarkupManager()
    evd_markup.add_line(
        label=("Left Kocher", "Left Target"),
        description=("Left Kocher", "Left Target"),
        position=(left_kp.as_float_list(), left_tgt.as_float_list()),
        display=evd_display,
        visible_points=True,
    )

    evd_markup.add_line(
        label=("Right Kocher", "Right Target"),
        description=("Right Kocher", "Right Target"),
        position=(right_kp.as_float_list(), right_tgt.as_float_list()),
        display=evd_display,
        visible_points=True,
    )
    time_dict["processing"]["Create Markups"] = time() - markup_start_time

    logger.info("Saving markups...")
    markup_save_start_time = time()
    evd_markup.save(output / "EVD.mrk.json")
    time_dict["write"]["Markups"] = time() - markup_save_start_time

    end = time()
    time_dict["total"] = end - start
    logger.info(f"Planning took {end - start:.3f}s.")

    if return_subtimes:
        return time_dict
    return None


@click.command(name="plan")
@click.option(
    "--skin",
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    required=True,
    help="Path to the skin mesh.",
)
@click.option(
    "--skin-model",
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    required=True,
    help="Path to the skin model.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(
        dir_okay=True, file_okay=False, writable=True, resolve_path=True, path_type=Path
    ),
    required=True,
    help="Path to the output directory.",
)
@click.option(
    "--ventricles",
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    required=False,
    help="Path to the ventricles.",
)
@click.option(
    "--gpu-model",
    is_flag=True,
    default=False,
    help="Whether to use GPU model or not.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level. Repeat for more verbosity.",
)
@click.option(
    "--write-intermediate",
    is_flag=True,
    default=False,
    help="Write intermediate results to file.",
)
def cli_plan(
    skin: Path,
    skin_model: Path,
    output: Path,
    ventricles: Path | None = None,
    gpu_model: bool = False,
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
    from evdplanner.cli import set_verbosity

    # Set the verbosity level
    set_verbosity(verbose)

    plan(
        skin=skin,
        skin_model=skin_model,
        output=output,
        ventricles=ventricles,
        gpu_model=gpu_model,
        write_intermediate=write_intermediate,
    )
