"""
Plot keypoints on an image.
"""

import json
from pathlib import Path

import click


@click.command()
@click.option(
    "-i",
    "--image",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
)
@click.option(
    "-k",
    "--keypoints",
    "keypoint_files",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    multiple=True,
)
@click.option("-o", "output", type=click.Path(dir_okay=False, resolve_path=True, path_type=Path))
@click.option(
    "-l",
    "--label",
    type=bool,
    is_flag=True,
    default=False,
    help="Whether to label keypoints or not.",
)
@click.option("-v", "--verbose", count=True, help="Verbosity level (repeat for more)")
def keypoints(
    image: Path, keypoint_files: list[Path], output: Path, label: bool = False, verbose: int = 0
) -> None:
    """
    Plot keypoints on an image.

    Parameters
    ----------
    image : Path
        Path to the image file.
    keypoint_files : list[Path]
        List of paths to the keypoint files.
    output : Path
        Path to the output file.
    label : bool
        Whether to label keypoints or not.
    verbose : int
        Verbosity level.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from imageio.v3 import imread
    from loguru import logger

    from evdplanner.cli import set_verbosity

    set_verbosity(verbose)

    sns.set_theme(context="paper", style="dark")
    sns.despine(left=True, bottom=True)

    kps = {}
    colormap = sns.color_palette("tab10", len(keypoint_files))

    for file in keypoint_files:
        logger.info(f"Reading keypoints from {file}")
        with file.open("r") as f:
            k = json.load(f)
            for kp in k:
                filename = file.name.split(".")[0]
                filename = " ".join(filename.split("_")[1:])

                if filename not in kps:
                    kps[filename] = {"label": [], "position": []}

                kps[filename]["label"].append(kp["label"])
                kps[filename]["position"].append(kp["position"])
                logger.debug(f"Adding keypoint {kp['label']} at {kp['position']} from {filename}")

    logger.info(f"Reading image from {image}")
    image = imread(image)
    logger.debug(f"Image shape: {image.shape}")
    relative_offset = 0.0125

    logger.info("Plotting keypoints on image")
    fig, axs = plt.subplots(2, 1, figsize=(9, 9), dpi=600)
    axs[0].imshow(image, cmap="gray")
    axs[0].axis("off")

    for idx, k in enumerate(kps.values()):
        labels = k["label"]
        positions = k["position"]
        x = [p[0] * image.shape[1] for p in positions]
        y = [p[1] * image.shape[0] for p in positions]

        axs[0].scatter(x, y, s=10, color=colormap[idx], marker="+")

        if label:
            logger.debug("Adding labels to keypoints")
            offset = min(relative_offset * image.shape[0], relative_offset * image.shape[1])
            for i, l in enumerate(labels):
                logger.debug(f"Adding label {l} to keypoint {i}")
                axs[0].text(
                    x[i],
                    y[i] - offset,
                    l,
                    fontsize=8,
                    color=colormap[idx],
                    rotation=(60 + idx * 15),
                )

    axs[0].legend(kps.keys(), loc="upper right", fontsize=8)

    axs[1].imshow(np.flip(image, axis=-1), cmap="gray")
    axs[1].axis("off")

    for idx, k in enumerate(kps.values()):
        labels = k["label"]
        positions = k["position"]
        x = [(1.0 - p[0]) * image.shape[1] for p in positions]
        y = [p[1] * image.shape[0] for p in positions]

        pairs = ((1, 2), (3, 4), (5, 6))
        for pair in pairs:
            x[pair[0]], x[pair[1]] = x[pair[1]], x[pair[0]]
            y[pair[0]], y[pair[1]] = y[pair[1]], y[pair[0]]

        axs[1].scatter(x, y, s=10, color=colormap[idx], marker="+")

        if label:
            logger.debug("Adding labels to keypoints")
            offset = min(relative_offset * image.shape[0], relative_offset * image.shape[1])
            for i, l in enumerate(labels):
                logger.debug(f"Adding label {l} to keypoint {i}")
                axs[1].text(
                    x[i],
                    y[i] - offset,
                    l,
                    fontsize=8,
                    color=colormap[idx],
                    rotation=(60 + idx * 15),
                )

    axs[1].legend(kps.keys(), loc="upper right", fontsize=8)

    if not output.parent.exists():
        output.parent.mkdir(parents=True)

    logger.info(f"Saving plot to {output}")
    plt.tight_layout()
    fig.savefig(output)
