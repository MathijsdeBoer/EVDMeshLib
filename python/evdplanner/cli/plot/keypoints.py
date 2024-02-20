import json
from pathlib import Path

import click

from evdplanner.cli import set_verbosity


@click.command()
@click.argument(
    "image", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path)
)
@click.argument(
    "keypoints", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path)
)
@click.argument("output", type=click.Path(dir_okay=False, resolve_path=True, path_type=Path))
@click.option("-v", "--verbose", count=True, help="Verbosity level (repeat for more)")
def keypoints(
    image: Path,
    keypoints: Path,
    output: Path,
    verbose: int = 0
):
    import logging
    import matplotlib.pyplot as plt
    import seaborn as sns
    from imageio.v3 import imread

    set_verbosity(verbose)
    logger = logging.getLogger(__name__)

    logger.info(f"Reading keypoints from {keypoints}")
    with keypoints.open("r") as f:
        keypoints: list[dict[str, str | list[float]]] = json.load(f)

    sns.set_theme(context="paper", style="dark")
    sns.despine(left=True, bottom=True)

    logger.debug("Collecting x and y coordinates from keypoints")
    x = [kp["position"][0] for kp in keypoints]
    y = [kp["position"][1] for kp in keypoints]

    logger.debug("Extracting labels")
    labels = [kp["label"] for kp in keypoints]

    logger.info(f"Reading image from {image}")
    image = imread(image)
    offset = 0.0125

    logger.debug("Transforming keypoints to image coordinates")
    x = [i * image.shape[1] for i in x]
    y = [i * image.shape[0] for i in y]

    logger.info("Plotting keypoints on image")
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), dpi=600)
    ax.imshow(image, cmap="viridis")
    ax.scatter(x, y, s=10, c="red", marker="+")

    logger.debug("Adding labels to keypoints")
    offset = min(offset * image.shape[0], offset * image.shape[1])
    for i, label in enumerate(labels):
        logger.debug(f"Adding label {label} to keypoint {i}")
        ax.text(x[i], y[i] - offset, label, fontsize=8, color="red", rotation=60)

    if not output.parent.exists():
        output.parent.mkdir(parents=True)

    logger.info(f"Saving plot to {output}")
    fig.savefig(output)
