from pathlib import Path

import click


@click.command()
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
    "--ventricles-model",
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    required=False,
    help="Path to the ventricles model.",
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
def plan(
    skin: Path,
    skin_model: Path,
    output: Path,
    ventricles: Path | None = None,
    ventricles_model: Path | None = None,
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
    from time import time

    from loguru import logger
    from monai.transforms import Compose
    from torch import device, load, no_grad, transpose

    from evdplanner.cli import set_verbosity
    from evdplanner.generation import generate_landmarks, measure_kocher
    from evdplanner.geometry import Mesh
    from evdplanner.linalg import Vec3
    from evdplanner.markups import DisplaySettings, MarkupManager
    from evdplanner.network.architecture import PointRegressor
    from evdplanner.network.transforms import default_load_transforms
    from evdplanner.rendering import Camera, CameraType, CPURenderer, IntersectionSort
    from evdplanner.rendering.utils import normalize_image

    start = time()

    # Set the verbosity level
    set_verbosity(verbose)

    if not output.exists():
        output.mkdir(parents=True)

    logger.info("Loading skin model...")
    device = device("cuda" if gpu_model else "cpu")
    skin_model: PointRegressor = load(skin_model, map_location=device)
    skin_model.eval()

    logger.debug("Skin model:")
    logger.debug(f"Input shape: {skin_model.in_shape}")
    logger.debug(f"Output shape: {skin_model.out_shape}")
    logger.debug(f"Keypoints: {skin_model.keypoints}")

    # Plan the EVD trajectory
    logger.info("Loading skin mesh...")
    skin = Mesh.load(str(skin), num_samples=10_000_000)

    logger.info("Creating camera...")
    camera = Camera(
        origin=skin.origin,
        forward=Vec3(0.0, -1.0, 0.0),
        up=Vec3(0.0, 0.0, 1.0),
        x_resolution=skin_model.in_shape[0],
        y_resolution=skin_model.in_shape[1],
        camera_type=CameraType.Equirectangular,
    )
    logger.debug(f"Camera: {camera}")

    logger.info("Rendering skin...")
    renderer = CPURenderer(camera, skin)
    skin_render = renderer.render(IntersectionSort.Farthest)
    logger.debug(f"Skin render shape: {skin_render.shape}")

    logger.info("Normalizing skin render...")
    skin_depth = skin_render[..., 0]
    skin_normal = skin_render[..., 1:]
    logger.debug(f"Skin depth shape: {skin_depth.shape}")
    logger.debug(f"Skin normal shape: {skin_normal.shape}")

    skin_depth = normalize_image(skin_depth)

    if write_intermediate:
        from imageio.v3 import imwrite

        logger.info("Writing intermediate results...")
        skin_depth_path = output / "map_skin_depth.png"
        skin_normal_path = output / "map_skin_normal.png"
        imwrite(skin_depth_path, (skin_depth * 65535).astype("uint16"))
        imwrite(skin_normal_path, (((skin_normal + 1) / 2) * 255).astype("uint8"))

    transform = Compose(
        default_load_transforms(
            maps=["depth", "normal"],
            image_key="image",
            include_file_reading=False,
            allow_missing_keys=True,
            input_channel_dim=-1,
        )
    )

    skin_input = transform({"depth": skin_depth[..., None], "normal": skin_normal})["image"][
        None, ...
    ]
    skin_input = transpose(skin_input, -1, -2)
    logger.debug(f"Skin input {skin_input.shape}")

    logger.info("Predicting skin landmarks...")
    with no_grad():
        skin_input = skin_input.to(device)
        skin_landmarks = skin_model(skin_input).squeeze().cpu().numpy()

    for keypoint, pred in zip(skin_model.keypoints, skin_landmarks, strict=True):
        logger.debug(f"{keypoint}: {pred}")

    if write_intermediate:
        logger.info("Writing intermediate results...")
        import json

        skin_landmarks_path = output / "projected_skin.kp.json"
        with skin_landmarks_path.open("w") as f:
            json.dump(skin_landmarks.tolist(), f)

    logger.info("Generating landmarks for skin.")
    logger.info("Creating camera...")
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

    for point, label in zip(lm, skin_model.keypoints, strict=True):
        logger.debug(f"Found {label} at {point}.")

    if write_intermediate:
        logger.info("Writing intermediate results...")
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

    nasion = lm[skin_model.keypoints.index("Nasion")]
    preauricle_left = lm[skin_model.keypoints.index("Pre-Auricle Left")]
    preauricle_right = lm[skin_model.keypoints.index("Pre-Auricle Right")]

    logger.info("Measuring Kocher's points.")
    left_kp, right_kp = measure_kocher(
        mesh=skin,
        nasion=nasion,
        left_ear=preauricle_left,
        right_ear=preauricle_right,
        n_angles=1000,
        debug=False,
    )

    logger.debug(f"Left Kocher's point: {left_kp}.")
    logger.debug(f"Right Kocher's point: {right_kp}.")

    if write_intermediate:
        logger.info("Writing intermediate results...")
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

    end = time()
    logger.info(f"Planning took {end - start:.3f}s.")
    logger.warning("Command implementation not yet finished")
