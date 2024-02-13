from pathlib import Path


def get_data(
    root: Path,
    anatomy: str,
    image_files: tuple[str] = ("map_{anatomy}_depth.png", "map_{anatomy}_normal.png"),
    label_file: str = "landmarks_{anatomy}.mrk.json",
) -> list[dict[str, Path]]:
    data = []

    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue

        image_files = [subdir / file.format(anatomy=anatomy) for file in image_files]
        label_file = subdir / label_file.format(anatomy=anatomy)

        if not all([file.exists() for file in image_files]) or not label_file.exists():
            continue

        data.append(
            {
                "map_depth": image_files[0],
                "map_normal": image_files[1],
                "label": label_file,
            }
        )

    return data


def verify_data(data: list[dict[str, Path]]) -> None:
    for sample in data:
        for file in sample.values():
            if not file.exists():
                raise FileNotFoundError(f"File {file} does not exist.")
