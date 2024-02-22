"""
Prediction CLI commands for the CLI.
"""
from pathlib import Path

import click


@click.command(name="predict-skin-mesh")
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True, resolve_path=True, path_type=Path),
    required=True,
    help="Path to the model file.",
)
@click.option(
    "-i",
    "--mesh",
    type=click.Path(exists=True, resolve_path=True, path_type=Path),
    required=True,
    help="Path to the mesh file.",
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(dir_okay=False, file_okay=True, resolve_path=True, path_type=Path),
    required=True,
    help="Path to the output file.",
)
@click.option(
    "--gpu-renderer",
    is_flag=True,
    default=False,
    help="Whether to use GPU rendering or not.",
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
def predict_skin_mesh(
    model_path: Path,
    mesh: Path,
    output_path: Path,
    gpu_renderer: bool = False,
    gpu_model: bool = False,
    verbose: int = 0,
) -> None:
    """
    Predict skin mesh.

    Parameters
    ----------
    model_path : Path
        Path to the model file.
    mesh : Path
        Path to the mesh file.
    output_path : Path
        Path to the output file.
    gpu_renderer : bool, optional
        Whether to use GPU rendering or not.
    gpu_model : bool, optional
        Whether to use GPU model or not.
    verbose : int, optional
        Verbosity level. Repeat for more verbosity.

    Returns
    -------
    None
    """
    import json

    import numpy as np
    import torch
    from loguru import logger
    from imageio.v3 import imwrite
    from monai.transforms import Compose

    from evdplanner.cli import set_verbosity
    from evdplanner.geometry import Mesh
    from evdplanner.linalg import Vec3
    from evdplanner.markups import MarkupManager
    from evdplanner.network import PointRegressor
    from evdplanner.network.training import LightningWrapper
    from evdplanner.network.transforms import default_raw_transforms
    from evdplanner.rendering import Camera, CameraType, IntersectionSort
    from evdplanner.rendering.utils import normalize_image

    set_verbosity(verbose)

    if gpu_renderer:
        logger.info("Using GPU renderer.")
        from evdplanner.rendering.gpu import GPURenderer as Renderer
    else:
        logger.info("Using CPU renderer.")
        from evdplanner.rendering import CPURenderer as Renderer

    logger.info("Loading model...")
    device = torch.device("cuda" if gpu_model else "cpu")
    model: PointRegressor = torch.load(model_path, map_location=device)
    model.eval()

    resolution = model.in_shape

    logger.debug(f"Model resolution: {resolution}")
    logger.info("Loading mesh...")
    mesh = Mesh.load(str(mesh), num_samples=10_000_000)

    logger.info("Creating camera...")
    camera = Camera(
        origin=mesh.origin,
        forward=Vec3(0.0, -1.0, 0.0),
        up=Vec3(0.0, 0.0, 1.0),
        x_resolution=resolution[1],
        y_resolution=resolution[0],
        camera_type=CameraType.Equirectangular,
    )

    logger.info("Creating renderer...")
    renderer = Renderer(
        camera=camera,
        mesh=mesh,
    )

    logger.info("Rendering...")
    image = renderer.render(IntersectionSort.Farthest)
    logger.debug(f"Image shape: {image.shape}")

    logger.info("Saving images...")
    depth_image = image[..., 0]
    normal_image = image[..., 1:]

    logger.debug("Normalizing images...")
    depth_image = normalize_image(depth_image)
    normal_image += 1.0
    normal_image /= 2.0

    logger.debug("Converting images to uint16 and uint8...")
    depth_image = (depth_image * 65535).astype(np.uint16)
    normal_image = (normal_image * 255).astype(np.uint8)

    depth_output = output_path.parent / f"{output_path.stem}_depth.png"
    logger.info(f"Saving to {depth_output}")
    imwrite(depth_output, depth_image)

    normal_output = output_path.parent / f"{output_path.stem}_normal.png"
    logger.info(f"Saving to {normal_output}")
    imwrite(normal_output, normal_image)

    logger.debug(f"Image shape: {image.shape}, dtype: {image.dtype}")

    transforms = default_raw_transforms(
        maps=model.maps,
        keypoints=model.keypoints,
        allow_missing_keys=True,
    )
    transforms = Compose(transforms)

    logger.info("Applying transforms...")
    image = transforms({
        map_name: im for map_name, im in zip(model.maps, [depth_output, normal_output])
    })["image"]
    image = image.to(device)
    logger.debug(f"Transformed image shape: {image.shape}, dtype: {image.dtype}")

    logger.info("Predicting...")
    with torch.no_grad():
        prediction = model(image).squeeze().cpu().numpy()
    logger.debug(f"Prediction: {prediction}")

    keypoints = model.keypoints
    logger.debug(f"Keypoints: {keypoints}")

    intersections = []
    for keypoint, pred in zip(keypoints, prediction):
        logger.debug(f"{keypoint}: {pred}")
        ray = camera.cast_ray(int(pred[0] * resolution[1]), int(pred[1] * resolution[0]))
        logger.debug(f"{keypoint} ray: {ray}")

        intersection = mesh.intersect(ray, IntersectionSort.Farthest)

        if intersection:
            logger.debug(f"{keypoint} intersection: {intersection.position.as_float_list()}")
            intersections.append(intersection.position.as_float_list())

    logger.info("Creating markups...")
    markups = MarkupManager()
    markups.add_fiducial(
        label=keypoints,
        description=keypoints,
        position=intersections,
    )

    logger.info("Saving markups...")
    with open(output_path, "w") as f:
        json.dump(markups.to_dict(), f)
