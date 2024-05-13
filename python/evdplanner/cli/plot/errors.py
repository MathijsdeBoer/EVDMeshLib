from pathlib import Path

import click


@click.command()
@click.argument(
    "input_path",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
)
@click.option(
    "--measurements",
    "measurements_name",
    type=str,
    required=False,
    default="measurements.json",
    help="Name of the measurements file.",
)
@click.option(
    "--times",
    "times_name",
    type=str,
    required=False,
    default="times.json",
    help="Name of the times file.",
)
@click.option(
    "--mesh",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
    ),
    required=False,
    default=None,
    help="Path to the mesh file.",
)
@click.option(
    "--reference-landmarks",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
    ),
    required=False,
    default=None,
    help="Path to the reference landmarks file.",
)
@click.option(
    "--origin-shift",
    type=float,
    nargs=3,
    required=False,
    default=(0.0, 0.0, 0.0),
    help="Origin shift.",
)
def errors(
    input_path: Path,
    measurements_name: str = "measurements.json",
    times_name: str = "times.json",
    mesh: Path = None,
    reference_landmarks: Path = None,
    origin_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """
    Plot errors.

    Parameters
    ----------
    input_path : Path
        Path to the input directory.
    measurements_name : str
        Name of the measurements file.
    """
    import json

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from scipy.stats import mannwhitneyu, shapiro, ttest_ind

    from evdplanner.geometry import Mesh
    from evdplanner.linalg import Vec3
    from evdplanner.markups import MarkupManager
    from evdplanner.rendering import Camera, CameraType, Renderer, IntersectionSort
    from evdplanner.rendering.utils import normalize_image

    subdirs = [x for x in input_path.iterdir() if x.is_dir()]
    errors = {
        "value": [],
        "label": [],
        "modality": [],
        "axis": [],
    }
    delta_errors = {
        "value": [],
        "label": [],
    }
    timings = {
        "value": [],
        "part": [],
    }

    for subdir in subdirs:
        measurements = subdir / measurements_name
        if not measurements.exists():
            continue

        times = subdir / times_name
        if not times.exists():
            continue

        with measurements.open("r") as file:
            data = json.load(file)

            for element in data:
                errors["label"].append(element["label"])
                errors["value"].append(element["error"])
                errors["modality"].append("CT" if "augmedit" in subdir.name.lower() else "MRI")
                errors["axis"].append("all")

                pred = element["predicted"]
                gt = element["gt"]

                for i in range(3):
                    errors["label"].append(element["label"])
                    errors["value"].append(pred[i] - gt[i])
                    errors["modality"].append("CT" if "augmedit" in subdir.name.lower() else "MRI")
                    errors["axis"].append(["x", "y", "z"][i])

                delta_errors["label"].append(element["label"])
                delta_errors["value"].append([pred[i] - gt[i] for i in range(3)])

        with times.open("r") as file:
            data = json.load(file)

            for k, v in data.items():
                timings["value"].append(v)
                timings["part"].append(k)

    error_df = pd.DataFrame(errors)
    delta_error_df = pd.DataFrame(delta_errors)
    timing_df = pd.DataFrame(timings)

    mri_errors = error_df.loc[error_df["modality"] == "MRI", "value"]
    ct_errors = error_df.loc[error_df["modality"] == "CT", "value"]

    # Normality tests
    print("Normality tests:")
    print("MRI errors:")
    print(shapiro(mri_errors.to_numpy()))
    print("CT errors:")
    print(shapiro(ct_errors.to_numpy()))

    stat = ttest_ind(mri_errors, ct_errors)
    print("T-test between MRI and CT errors:")
    print(f"\tt: {stat.statistic}, p: {stat.pvalue}")

    stat = mannwhitneyu(mri_errors, ct_errors)
    print("Mann-Whitney U test between MRI and CT errors:")
    print(f"\tU: {stat.statistic}, p: {stat.pvalue}")

    significance = {}
    significance_level = 0.05

    # print("Mann-Whitney U test between MRI and CT errors for each label:")
    for label in error_df["label"].unique():
        mri_label_errors = error_df.loc[
            (error_df["modality"] == "MRI")
            & (error_df["label"] == label)
            & (error_df["axis"] == "all"),
            "value",
        ]
        ct_label_errors = error_df.loc[
            (error_df["modality"] == "CT")
            & (error_df["label"] == label)
            & (error_df["axis"] == "all"),
            "value",
        ]

        stat = mannwhitneyu(
            mri_label_errors,
            ct_label_errors,
        )
        print(f"\t{label}:")
        print(f"\t\tU: {stat.statistic}, p: {stat.pvalue}")
        significance[label] = stat.pvalue

        stat = ttest_ind(mri_label_errors, ct_label_errors)
        print(f"\tt: {stat.statistic}, p: {stat.pvalue}")

    # Sidak correction
    significance_level = 1 - (1 - significance_level) ** (1 / len(error_df["label"].unique()))

    sns.set_style("darkgrid")
    sns.set_theme(rc={"figure.figsize": (12, 6)})
    sns.set_context("paper", font_scale=1.5)

    p = sns.boxenplot(
        data=error_df[error_df["axis"] == "all"],
        x="label",
        y="value",
        hue="modality",
        k_depth="trustworthy",
        trust_alpha=0.05,
    )
    p.set_title("Euclidian Distance")
    p.set_ylabel("Error (mm)")
    p.set_xlabel("Label")
    p.set_xticklabels(p.get_xticklabels(), rotation=30, ha="right")

    # Add significance stars
    print(f"Significance level: {significance_level}")
    for i, label in enumerate(p.get_xticklabels()):
        p_value = significance[error_df["label"].unique()[i]]
        mr_vals = error_df.loc[
            (error_df["label"] == label.get_text())
            & (error_df["modality"] == "MRI")
            & (error_df["axis"] == "all"),
            "value",
        ]
        ct_vals = error_df.loc[
            (error_df["label"] == label.get_text())
            & (error_df["modality"] == "CT")
            & (error_df["axis"] == "all"),
            "value",
        ]
        mean_mr = mr_vals.mean()
        mean_ct = ct_vals.mean()

        plt.scatter(
            x=[i - 0.2, i + 0.2],
            y=[mean_mr, mean_ct],
            color=["blue", "orange"],
            marker="*",
            s=100,
        )

        print(f"{p_value}")

        print(f"Means for label {label.get_text()}: MRI: {mean_mr}, CT: {mean_ct}")
        if significance[error_df["label"].unique()[i]] < significance_level:
            p.text(
                i,
                error_df.loc[error_df["label"] == label.get_text(), "value"].max() + 1.0,
                f"{p_value:.3f}*",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12,
                fontweight="bold",
            )
        else:
            p.text(
                i,
                error_df.loc[error_df["label"] == label.get_text(), "value"].max() + 1.0,
                f"{p_value:.3f}",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12,
            )

        p.add_line(
            plt.Line2D(
                [i - 0.2, i + 0.2],
                [error_df.loc[error_df["label"] == label.get_text(), "value"].max() + 0.5] * 2,
                color="k",
                linewidth=1.5,
            )
        )

    p.set_ybound(0, error_df["value"].max() + 3.0)

    plt.tight_layout()
    plt.show()

    if mesh:
        if not reference_landmarks:
            raise ValueError("Reference landmarks must be provided when mesh is provided.")

        manager = MarkupManager().load(reference_landmarks)
        control_points: list[tuple[str, Vec3]] = []
        for markup in manager.markups:
            for control_point in markup.control_points:
                control_points.append((control_point.label, Vec3(*control_point.position)))

        mesh = Mesh.load(str(mesh))
        camera = Camera(
            origin=mesh.origin + Vec3(*origin_shift),
            forward=Vec3(0.0, -1.0, 0.0),
            up=Vec3(0.0, 0.0, 1.0),
            x_resolution=8192,
            y_resolution=4096,
            camera_type=CameraType.Equirectangular,
        )
        renderer = Renderer(camera, mesh)

        render = renderer.render(IntersectionSort.Farthest)[..., 0]
        render = normalize_image(render, lower_percentile=1.0, upper_percentile=85.0)

        sns.set_style("dark")
        sns.set_context("paper", font_scale=1.0)
        fig, ax = plt.subplots(1, 1, figsize=(9, 4.5), dpi=300)
        ax.imshow(render, cmap="gray")
        ax.axis("off")

        delta_err_df = {
            "x": [],
            "y": [],
            "label": [],
        }
        for label, position in control_points:
            curr_deltas = delta_error_df[delta_error_df["label"] == label]["value"].values
            for delta in curr_deltas:
                error_pos = Vec3(*delta) + position
                x, y = camera.project_back(error_pos)

                delta_err_df["x"].append(x)
                delta_err_df["y"].append(y)
                delta_err_df["label"].append(label)

        delta_error_df = pd.DataFrame(delta_err_df)

        sns.kdeplot(
            data=delta_error_df,
            x="x",
            y="y",
            hue="label",
            fill=True,
            levels=16,
            thresh=0.05,
            common_norm=False,
            ax=ax,
        )

        plt.tight_layout()
        plt.show()
