from pathlib import Path

import numpy as np
from SimpleITK import ImageFileReader, GetArrayFromImage


root = Path(r"S:\E_ResearchData\evdplanner\Samples")
subdirs = [x for x in root.iterdir() if x.is_dir()]

matches = []

for subdir in subdirs:
    print(subdir.name)

    image = subdir / "image.nii.gz"
    reader = ImageFileReader()
    reader.SetFileName(str(image))
    reader.ReadImageInformation()

    for other in subdirs:
        if subdir == other:
            continue

        other_image = other / "image.nii.gz"

        other_reader = ImageFileReader()
        other_reader.SetFileName(str(other_image))
        other_reader.ReadImageInformation()

        if reader.GetSize() != other_reader.GetSize():
            continue
        if reader.GetSpacing() != other_reader.GetSpacing():
            continue
        if reader.GetOrigin() != other_reader.GetOrigin():
            continue
        if reader.GetDirection() != other_reader.GetDirection():
            continue

        print(f"Partial match for {subdir.name} and {other.name}")
        image = reader.Execute()
        other_image = other_reader.Execute()

        image_data = GetArrayFromImage(image)
        other_image_data = GetArrayFromImage(other_image)

        if image_data.shape != other_image_data.shape:
            continue

        if np.all(image_data == other_image_data):
            print(f"    {other.name} is a duplicate of {subdir.name}")
            matches.append((subdir, other))

print("Matches:")
for subdir, other in matches:
    print(subdir.name, other.name)

print("Done")
