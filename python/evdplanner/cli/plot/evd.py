from pathlib import Path

import click
import matplotlib.pyplot as plt
import seaborn as sns

from evdplanner.markups import MarkupManager


@click.group
def evd() -> None:
    pass


@evd.command()
@click.argument(
    "dataset",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        readable=True,
        path_type=Path,
    ),
)
@click.option(
    "-f",
    "--filename",
    "evd_filename",
    type=str,
    default="EVD.mrk.json",
    help="The filename of the EVD markup.",
)
def lengths(
    dataset: Path,
    evd_filename: str = "EVD.mrk.json",
) -> None:
    """
    Plot the lengths of the EVDs in the dataset.

    Parameters
    ----------
    dataset : Path
        The path to the dataset.
    evd_filename : str, optional
        The filename of the EVD markup, by default "EVD.mrk.json".

    Returns
    -------
    None
    """
    from tqdm import tqdm

    from evdplanner.linalg import Vec3

    patients = [x.resolve() for x in dataset.iterdir() if x.is_dir()]
    evd_lengths = []
    for patient in tqdm(patients, desc="Processing patients", total=len(patients)):
        if not (patient / evd_filename).exists():
            continue

        markup = MarkupManager.load(patient / evd_filename)
        left_kp = Vec3(*markup.find_fiducial("Left Kocher").position)
        left_tp = Vec3(*markup.find_fiducial("Left Target").position)

        right_kp = Vec3(*markup.find_fiducial("Right Kocher").position)
        right_tp = Vec3(*markup.find_fiducial("Right Target").position)

        evd_lengths.append((left_kp - left_tp).length)
        evd_lengths.append((right_kp - right_tp).length)

    p = sns.histplot(
        evd_lengths,
        stat="frequency",
        kde=True,
        binwidth=5,
    )

    p.set_title("EVD Lengths")
    p.set_xlabel("Length (mm)")
    p.set_ylabel("Count")

    plt.show()
