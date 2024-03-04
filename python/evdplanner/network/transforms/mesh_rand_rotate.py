import random
from pathlib import Path
from typing import Hashable, Mapping

import monai.transforms as mt

from evdplanner.geometry import Mesh
from evdplanner.linalg import Vec3, Mat4


class MeshRandRotated(mt.MapTransform):
    def __init__(
            self,
            keys: list[str],
            mesh_key: str,
            landmarks_key: str,
            rotation_range: float | tuple[float, float] = 1.0,
            allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

        self.mesh_key = mesh_key
        self.landmarks_key = landmarks_key

        if isinstance(rotation_range, float):
            self.rotation_range = (-rotation_range, rotation_range)
        else:
            self.rotation_range = tuple(rotation_range)

    def __call__(self, data: Mapping[Hashable, Path | Mesh]) -> dict:
        d = dict(data)
        matrix = None

        if self.mesh_key not in d.keys():
            if not self.allow_missing_keys:
                msg = f"Missing key: {self.mesh_key}."
                raise KeyError(msg)
        else:
            mesh = d[self.mesh_key]
            if not isinstance(mesh, Mesh):
                msg = f"Expected a Mesh, but got {type(mesh)}."
                raise ValueError(msg)

            random_axis = Vec3(
                random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)
            ).unit_vector
            random_angle = random.uniform(*self.rotation_range)

            origin = mesh.origin
            matrix = (
                    Mat4.translation(-origin.x, -origin.y, -origin.z)
                    * Mat4.rotation(random_axis, random_angle)
                    * Mat4.translation(origin.x, origin.y, origin.z)
            )

            mesh.transform(matrix)
            d[self.mesh_key] = mesh

        if self.landmarks_key not in d.keys():
            if not self.allow_missing_keys:
                msg = f"Missing key: {self.landmarks_key}."
                raise KeyError(msg)
        else:
            landmarks, names = d[self.landmarks_key]

            if matrix is not None:
                for i in range(len(landmarks)):
                    landmarks[i] = landmarks[i] @ matrix
                d[self.landmarks_key] = landmarks, names
            else:
                msg = f"Matrix is None."
                raise ValueError(msg)

        return d
