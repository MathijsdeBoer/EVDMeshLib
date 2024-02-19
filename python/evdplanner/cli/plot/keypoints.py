import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import seaborn as sns
from imageio.v3 import imread


@click.command()
@click.argument(
    "image", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path)
)
@click.argument(
    "keypoints", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path)
)
@click.argument("output", type=click.Path(dir_okay=False, resolve_path=True, path_type=Path))
def keypoints(
    image: Path,
    keypoints: Path,
    output: Path,
):
    with keypoints.open("r") as f:
        keypoints: list[dict[str, str | list[float]]] = json.load(f)

    sns.set_theme(context="paper", style="dark")
    sns.despine(left=True, bottom=True)

    x = [kp["position"][0] for kp in keypoints]
    y = [kp["position"][1] for kp in keypoints]
    labels = [kp["label"] for kp in keypoints]
    image = imread(image)
    offset = 0.0125

    x = [i * image.shape[1] for i in x]
    y = [i * image.shape[0] for i in y]

    fig, ax = plt.subplots(1, 1, figsize=(9, 9), dpi=600)
    ax.imshow(image, cmap="viridis")
    ax.scatter(x, y, s=10, c="red", marker="+")

    offset = min(offset * image.shape[0], offset * image.shape[1])
    for i, label in enumerate(labels):
        ax.text(x[i], y[i] - offset, label, fontsize=8, color="red", rotation=60)

    if not output.parent.exists():
        output.parent.mkdir(parents=True)

    fig.savefig(output)
