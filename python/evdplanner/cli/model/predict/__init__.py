"""
Prediction CLI commands for the CLI.
"""
from pathlib import Path

import click


@click.command()
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True, resolve_path=True, path_type=Path),
    required=True,
    help="Path to the model file.",
)
@click.option(
    "-i",
    "--input-images",
    type=(str, click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path)),
    multiple=True,
    required=True,
    help="Input images to predict from.",
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
def predict(
    model_path: Path,
    input_images: list[tuple[str, Path]],
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
    input_images : list[tuple[str, Path]]
        List of tuples containing the image type and the path to the image.
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
    from time import time

    from loguru import logger
    from monai.transforms import Compose
    from torch import device, load, no_grad

    from evdplanner.cli import set_verbosity
    from evdplanner.network.training import LightningWrapper, OptimizableModel
    from evdplanner.network.transforms import default_load_transforms

    set_verbosity(verbose)

    if gpu_renderer:
        logger.info("Using GPU renderer.")
    else:
        logger.info("Using CPU renderer.")

    logger.info("Loading model...")
    device = device("cuda" if gpu_model else "cpu")
    model: OptimizableModel = load(model_path, map_location=device)
    model.eval()
    resolution = model.in_shape

    logger.debug(f"Model resolution: {resolution}")

    logger.info("Loading images...")
    images = {}
    for image_type, image_path in input_images:
        logger.debug(f"Loading {image_type} image from {image_path}.")
        if "depth" in image_type:
            images["map_depth"] = image_path
        elif "normal" in image_type:
            images["map_normal"] = image_path
        else:
            msg = f"Unknown image type: {image_type}."
            logger.error(msg)
            raise ValueError(msg)

    transforms = Compose(
        default_load_transforms(
            maps=list(images.keys()),
            allow_missing_keys=True,
        )
    )
    image = transforms(images)["image"]
    logger.debug(f"Image shape: {image.shape}, dtype: {image.dtype}")
    image = image[None, ...]
    image = image.to(device)
    logger.debug(f"Transformed image shape: {image.shape}, dtype: {image.dtype}")

    logger.info("Predicting...")
    start = time()
    with no_grad():
        prediction = model(image).squeeze().cpu().numpy()
    delta = time() - start
    logger.info(f"Prediction took {delta:.2f}s.")

    keypoints = model.keypoints
    projections = []
    for keypoint, pred in zip(keypoints, prediction, strict=True):
        logger.debug(f"{keypoint}: {pred}")
        projections.append(
            {
                "label": keypoint,
                "position": pred.tolist(),
            }
        )

    logger.info(f"Saving projections to {output_path}.")
    with output_path.open("w") as f:
        json.dump(projections, f, indent=4)
