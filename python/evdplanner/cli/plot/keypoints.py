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
    verbose : int
        Verbosity level.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
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
    offset = 0.0125

    logger.info("Plotting keypoints on image")
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), dpi=600)
    ax.imshow(image, cmap="viridis")

    for idx, (filename, k) in enumerate(kps.items()):
        labels = k["label"]
        positions = k["position"]
        x = [p[0] * image.shape[1] for p in positions]
        y = [p[1] * image.shape[0] for p in positions]

        ax.scatter(x, y, s=10, color=colormap[idx], marker="+")

        if label:
            logger.debug("Adding labels to keypoints")
            offset = min(offset * image.shape[0], offset * image.shape[1])
            for i, label in enumerate(labels):
                logger.debug(f"Adding label {label} to keypoint {i}")
                ax.text(
                    x[i],
                    y[i] - offset,
                    label,
                    fontsize=8,
                    color=colormap[idx],
                    rotation=(60 + idx * 15),
                )

    ax.legend(kps.keys(), loc="upper right", fontsize=8)

    if not output.parent.exists():
        output.parent.mkdir(parents=True)

    logger.info(f"Saving plot to {output}")
    fig.savefig(output)
