from pathlib import Path

import click

from evdplanner.network.transforms.keypoint_flip import flip_keypoints


@click.command()
@click.argument(
    "input_path",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
)
@click.option(
    "--landmarks",
    "landmarks_name",
    type=str,
    required=True,
    help="Name of the landmarks file.",
)
@click.option(
    "--mesh",
    "mesh_name",
    type=str,
    required=True,
    help="Name of the mesh file.",
)
@click.option(
    "--model",
    "model_path",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
    ),
    required=True,
    multiple=True,
    help="Model file to validate. Can be multiple for ensembling.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (can be specified multiple times)",
)
def model(
    input_path: Path,
    landmarks_name: str,
    mesh_name: str,
    model_path: list[Path],
    verbose: int = 0,
) -> None:
    """
    Validate model.

    Parameters
    ----------
    input_path : Path
        Path to the input directory.
    model : list[Path]
        Path to the model file. Can be multiple for ensembling.
    verbose : int
        Verbosity level.

    Returns
    -------
    None
    """
    import json
    from time import time

    import numpy as np
    from loguru import logger
    from monai.transforms import Compose
    from torch import load, no_grad, transpose, zeros

    from evdplanner.cli import set_verbosity
    from evdplanner.generation import generate_landmarks
    from evdplanner.geometry import Mesh
    from evdplanner.linalg import Vec3
    from evdplanner.markups import DisplaySettings, MarkupManager
    from evdplanner.network.architecture.point_regressor import PointRegressor
    from evdplanner.network.transforms import default_load_transforms
    from evdplanner.rendering import Camera, CameraType, CPURenderer, IntersectionSort
    from evdplanner.rendering.utils import normalize_image

    set_verbosity(verbose)

    logger.info(f"Validating model(s) {model_path}...")

    errors = []
    times = []

    max_error_sample = {}

    subdirs = [x for x in input_path.iterdir() if x.is_dir()]
    for subdir in subdirs:
        logger.info(subdir.name)

        # We deliberately put the model loading here, to simulate
        # individual predictions in the plan function.
        model_load_start = time()
        models: list[PointRegressor] = [load(x) for x in model_path]
        for m in models:
            m.eval()
        resolution = models[0].in_shape
        keypoints = models[0].keypoints
        for m in models:
            if not m.in_shape == resolution:
                msg = "All models must have the same input resolution."
                logger.error(msg)
                raise ValueError(msg)
            if not m.keypoints == keypoints:
                msg = "All models must have the same keypoints."
                logger.error(msg)
                raise ValueError(msg)
        model_load_end = time()
        model_load = model_load_end - model_load_start
        logger.info(f"Model(s) loaded in {model_load:.2f} seconds.")

        landmarks_path = subdir / landmarks_name
        mesh_path = subdir / mesh_name

        if not landmarks_path.exists() or not mesh_path.exists():
            logger.warning(f"Skipping {subdir.name}.")
            continue

        manager = MarkupManager.load(landmarks_path)

        mesh_load_start = time()
        mesh = Mesh.load(str(mesh_path))
        mesh_load_end = time()
        mesh_load = mesh_load_end - mesh_load_start
        logger.info(f"Mesh loaded in {mesh_load:.2f} seconds.")

        rendering_prep_start = time()
        camera = Camera(
            origin=mesh.origin,
            forward=Vec3(0.0, -1.0, 0.0),
            up=Vec3(0.0, 0.0, 1.0),
            x_resolution=resolution[0],
            y_resolution=resolution[1],
            camera_type=CameraType.Equirectangular,
        )
        renderer = CPURenderer(
            camera=camera,
            mesh=mesh,
        )
        rendering_prep_end = time()
        rendering_prep = rendering_prep_end - rendering_prep_start
        logger.info(f"Rendering prepared in {rendering_prep:.2f} seconds.")

        render_start = time()
        skin_render = renderer.render(IntersectionSort.Farthest)
        render_end = time()
        render_time = render_end - render_start
        logger.info(f"Rendering completed in {render_time:.2f} seconds.")

        normalizing_start = time()
        skin_depth = skin_render[..., 0]
        skin_normal = skin_render[..., 1:]

        skin_depth = normalize_image(skin_depth)

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
        normalizing_end = time()
        normalizing_time = normalizing_end - normalizing_start
        logger.info(f"Normalization completed in {normalizing_time:.2f} seconds.")

        prediction_start = time()
        with no_grad():
            skin_landmarks = zeros((len(models), *models[0].out_shape), device=skin_input.device)
            for idx, m in enumerate(models):
                skin_landmarks[idx] = m(skin_input).squeeze().cpu()

            skin_landmarks = skin_landmarks.numpy()

            for idx, keypoint in enumerate(keypoints):
                kp_set = skin_landmarks[:, idx, :]
                logger.debug(f"Keypoint {keypoint}: {kp_set}.")
                logger.debug(f"Mean: {[np.mean(kp_set[:, i]) for i in range(2)]}.")
                logger.debug(f"Std: {[np.std(kp_set[:, i]) for i in range(2)]}.")
                logger.debug(f"Max: {[np.max(kp_set[:, i]) for i in range(2)]}.")
                logger.debug(f"Min: {[np.min(kp_set[:, i]) for i in range(2)]}.")
                logger.debug(f"Median: {[np.median(kp_set[:, i]) for i in range(2)]}.")
                logger.debug(
                    f"95CI: {[np.percentile(kp_set[:, i], [2.5, 97.5]) for i in range(2)]}."
                )

            skin_landmarks = np.median(skin_landmarks, axis=0)
        prediction_end = time()
        prediction_time = prediction_end - prediction_start
        logger.info(f"Prediction completed in {prediction_time:.2f} seconds.")

        reprojection_start = time()
        camera = Camera(
            origin=mesh.origin,
            forward=Vec3(0.0, -1.0, 0.0),
            up=Vec3(0.0, 0.0, 1.0),
            x_resolution=1,
            y_resolution=1,
            camera_type=CameraType.Equirectangular,
        )
        lm = generate_landmarks(
            projection=[(point[0], point[1]) for point in skin_landmarks],
            mesh=mesh,
            camera=camera,
            intersection_sort=IntersectionSort.Farthest,
        )
        reprojection_end = time()
        reprojection_time = reprojection_end - reprojection_start
        logger.info(f"Reprojection completed in {reprojection_time:.2f} seconds.")

        write_pred_start = time()
        pred_manager = MarkupManager()
        pred_manager.add_fiducial(
            label=keypoints,
            description=keypoints,
            position=[(point.x, point.y, point.z) for point in lm],
            display=DisplaySettings(
                color=(1, 0, 0),
                selected_color=(1, 0, 0),
                active_color=(1, 0, 0),
            ),
            visible_points=True,
        )
        pred_manager.save(subdir / "landmarks_skin_predicted.mrk.json")
        write_pred_end = time()
        write_pred_time = write_pred_end - write_pred_start
        logger.info(f"Predicted landmarks written in {write_pred_time:.2f} seconds.")

        measurements = []
        for point, label in zip(lm, keypoints, strict=True):
            logger.debug(f"Assessing {label} at {point}.")

            gt = Vec3(*manager.find_fiducial(label).position)
            logger.debug(f"Ground truth: {gt}.")

            error = (gt - point).length
            logger.debug(f"Error: {error:.2f}.")

            if label not in max_error_sample or error > max_error_sample[label]["error"]:
                max_error_sample[label] = {
                    "error": error,
                    "subdir": subdir,
                }

            measurements.append(
                {
                    "label": label,
                    "error": error,
                    "predicted": point.as_float_list(),
                    "gt": gt.as_float_list(),
                }
            )
            errors.append(error)

        with open(subdir / "measurements.json", "w") as f:
            json.dump(measurements, f, indent=4)

        with open(subdir / "times.json", "w") as f:
            json.dump(
                {
                    "model_load": model_load,
                    "mesh_load": mesh_load,
                    "rendering_prep": rendering_prep,
                    "render_time": render_time,
                    "normalizing_time": normalizing_time,
                    "prediction_time": prediction_time,
                    "reprojection_time": reprojection_time,
                    "write_pred_time": write_pred_time,
                },
                f,
                indent=4,
            )

        total_time = (
            model_load
            + mesh_load
            + rendering_prep
            + render_time
            + normalizing_time
            + prediction_time
            + reprojection_time
            + write_pred_time
        )
        times.append(total_time)
        logger.info(f"Total time: {total_time:.4f} seconds.")

        del models
        del mesh
        del renderer
        del skin_render
        del skin_depth
        del skin_normal
        del skin_input
        del skin_landmarks
        del camera
        del lm
        del manager
        del pred_manager

    logger.info(f"Mean error:   {np.mean(errors):.4f} mm.")
    logger.info(f"Median error: {np.median(errors):.4f} mm.")
    logger.info(f"Max error:    {np.max(errors):.4f} mm.")
    logger.info(f"Min error:    {np.min(errors):.4f} mm.")
    logger.info(f"Std error:    {np.std(errors):.4f} mm.")
    logger.info(
        f"95CI:        [{np.percentile(errors, 2.5):.4f}, {np.percentile(errors, 97.5):.4f}] mm."
    )

    logger.info(f"Mean time:   {np.mean(times):.4f} seconds.")
    logger.info(f"Median time: {np.median(times):.4f} seconds.")
    logger.info(f"Max time:    {np.max(times):.4f} seconds.")
    logger.info(f"Min time:    {np.min(times):.4f} seconds.")
    logger.info(f"Std time:    {np.std(times):.4f} seconds.")
    logger.info(
        f"95CI:        [{np.percentile(times, 2.5):.4f}, {np.percentile(times, 97.5):.4f}] seconds."
    )

    for label, item in max_error_sample.items():
        logger.info(f"Max error ({item['error']:.3f}) for {label} at {item['subdir']}.")
