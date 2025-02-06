from pathlib import Path

import click


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
@click.option(
    "-o",
    "--output",
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    required=False,
    help="Path to save the plot image.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity.",
)
def lengths(
    dataset: Path,
    evd_filename: str = "EVD.mrk.json",
    output: Path | None = None,
    verbose: int = 0,
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
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from loguru import logger
    from tqdm import tqdm

    from evdplanner.cli import set_verbosity
    from evdplanner.linalg import Vec3
    from evdplanner.markups import MarkupManager

    set_verbosity(verbose)

    logger.debug(f"Processing dataset: {dataset}")
    patients = [x.resolve() for x in dataset.iterdir() if x.is_dir()]
    logger.debug(f"Found {len(patients)} patients.")

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
    
    evd_lengths = np.array(evd_lengths)

    logger.info(f"Found {len(evd_lengths)} EVDs.")
    logger.info(f"Min: {evd_lengths.min()}")
    logger.info(f"Max: {evd_lengths.max()}")
    logger.info(f"Mean: {evd_lengths.mean()}")
    logger.info(f"Std: {evd_lengths.std()}")

    logger.info(f"Median: {np.median(evd_lengths)}")
    logger.info(f"95CI: {np.percentile(evd_lengths, 2.5)} - {np.percentile(evd_lengths, 97.5)}")

    logger.info("Plotting histogram.")
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette="colorblind",
        rc={
            "figure.figsize": (8, 4),
            "figure.dpi": 600,
        },
    )
    p = sns.histplot(
        evd_lengths,
        stat="count",
        kde=True,
        binwidth=5,
    )

    p.axvline(80.0, color="red", linestyle=":", label="80mm threshold", alpha=0.5)

    p.axvline(np.median(evd_lengths), color="black", linestyle="--", label="Median", alpha=0.5)
    p.axvline(np.percentile(evd_lengths, 2.5), color="black", linestyle="--", alpha=0.5)
    p.axvline(np.percentile(evd_lengths, 97.5), color="black", linestyle="--", alpha=0.5)

    p.axvline(np.mean(evd_lengths), color="blue", linestyle="-.", label="Mean", alpha=0.5)
    p.axvline(np.mean(evd_lengths) + np.std(evd_lengths), color="blue", linestyle="-.", alpha=0.5)
    p.axvline(np.mean(evd_lengths) - np.std(evd_lengths), color="blue", linestyle="-.", alpha=0.5)

    p.legend()

    p.set_title("EVD Lengths")
    p.set_xlabel("Length (mm)")
    p.set_ylabel("Count")

    if output is not None:
        logger.info(f"Saving plot to {output}.")
        plt.savefig(output)
    else:
        plt.show()
