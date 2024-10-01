from pathlib import Path

import click


@click.command()
@click.option(
    "-d",
    "--dataset",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        readable=True,
        path_type=Path,
    ),
    required=True,
    multiple=True,
    help="Path to the dataset(s).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    required=True,
    help="Path to save the CSV file with the timings.",
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
    "--skin-mesh-name",
    type=str,
    default="mesh_skin.stl",
    help="The filename of the skin mesh.",
)
@click.option(
    "--ventricle-mesh-name",
    type=str,
    default="mesh_ventricles.stl",
    help="The filename of the ventricle mesh.",
)
@click.option(
    "--use-gpu",
    is_flag=True,
    default=False,
    help="Whether to use the GPU to predict or not.",
)
@click.option(
    "-n",
    "--n-runs",
    type=int,
    default=1,
    help="Number of times to run the EVD planning.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level. Repeat for more verbosity.",
)
def times(
    dataset: list[Path],
    output: Path,
    skin_model: Path,
    skin_mesh_name: str = "mesh_skin.stl",
    ventricle_mesh_name: str = "mesh_ventricles.stl",
    use_gpu: bool = False,
    n_runs: int = 1,
    verbose: int = 0,
) -> None:
    """
    Collect the timings of EVD planning in the dataset.

    Parameters
    ----------
    dataset : list[Path]
        The path to the dataset(s).
    output : Path
        The path to save the CSV file with the timings.
    skin_model : Path
        The path to the skin model.
    skin_mesh_name : str, optional
        The filename of the skin mesh, by default "mesh_skin.stl".
    ventricle_mesh_name : str, optional
        The filename of the ventricle mesh, by default "mesh_ventricles.stl".
    use_gpu : bool, optional
        Whether to use the GPU to predict or not, by default False
    n_runs : int, optional
        Number of times to run the EVD planning, by default 1
    verbose : int, optional
        Verbosity level. Repeat for more verbosity, by default 0

    Returns
    -------
    None
    """
    import tempfile
    from shutil import rmtree

    import numpy as np
    import pandas as pd
    from loguru import logger

    from evdplanner.cli import set_verbosity
    from evdplanner.cli.plan import plan

    set_verbosity(verbose)

    patients = []
    for data in dataset:
        patients += [x.resolve() for x in data.iterdir() if x.is_dir()]
    evd_times = []
    df = []

    logger.info(f"Running EVD planning on {len(patients)} patients, {n_runs} times.")
    for run in range(n_runs):
        for idx, patient in enumerate(patients):
            logger.info(f"{patient.name} ({idx + 1}/{len(patients)}), run {run + 1}/{n_runs}")
            output_tmp = Path(tempfile.mkdtemp())

            skin_mesh = patient / skin_mesh_name
            if not skin_mesh.exists():
                logger.warning(f"Skin mesh {skin_mesh} does not exist.")
                continue

            ventricle_mesh = patient / ventricle_mesh_name
            if not ventricle_mesh.exists():
                logger.warning(f"Ventricle mesh {ventricle_mesh} does not exist.")
                continue

            time_dict = plan(
                skin_mesh,
                skin_model,
                output_tmp,
                ventricle_mesh,
                gpu_model=use_gpu,
                write_intermediate=False,
                return_subtimes=True
            )
            evd_times.append(time_dict)
            logger.info(f"Time: {time_dict['total']:.2f}s")

            df.append({
                "patient": patient.name,
                "main stage": "common",
                "substage": "total",
                "time": time_dict["total"],
            })
            for key, value in time_dict.items():
                if key == "total":
                    continue
                for subkey, subvalue in value.items():
                    df.append({
                        "patient": f"{patient.name} run {run}",
                        "main stage": key,
                        "substage": subkey,
                        "time": subvalue,
                    })

            # Clean up the temporary directory
            rmtree(output_tmp)

    logger.info(f"Creating CSV file with timings. {len(df)} entries.")
    df = pd.DataFrame(df)

    logger.info(f"Saving CSV file with timings.")
    df.to_csv(output, index=False)

    total_times = np.array([x["total"] for x in evd_times])
    logger.info(f"Mean time: {total_times.mean():.2f}s")
    logger.info(f"Std time: {total_times.std():.2f}s")
    logger.info(f"Max time: {total_times.max():.2f}s")
    logger.info(f"Min time: {total_times.min():.2f}s")
