from math import pi
from pathlib import Path
from typing import Any, Hashable, Mapping

import monai.transforms as mt
import numpy as np
from monai.transforms import Randomizable

from evdplanner.geometry import Mesh
from evdplanner.linalg import Mat4, Vec3


class MeshRandRotated(mt.RandomizableTransform, mt.MapTransform):
    def __init__(
        self,
        keys: list[str],
        mesh_key: str,
        landmarks_key: str,
        rotation_range: float | tuple[float, float] = pi,
        allow_missing_keys: bool = False,
        prob: float = 1.0,
    ) -> None:
        mt.RandomizableTransform.__init__(self, prob=prob)
        mt.MapTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)

        self.mesh_key = mesh_key
        self.landmarks_key = landmarks_key

        if isinstance(rotation_range, float):
            self.rotation_range = (-rotation_range, rotation_range)
        else:
            self.rotation_range = tuple(rotation_range)

        self.rotation_axis = None
        self.rotation_angle = None

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> Randomizable:
        super().set_random_state(seed, state)
        return self

    def randomize(self, data: Any) -> None:
        super().randomize(data)
        self.rotation_axis = Vec3(
            self.R.uniform(-1.0, 1.0), self.R.uniform(-1.0, 1.0), self.R.uniform(-1.0, 1.0)
        ).unit_vector
        self.rotation_angle = self.R.uniform(*self.rotation_range)

    def __call__(self, data: Mapping[Hashable, Path | Mesh]) -> dict:
        d = dict(data)
        matrix = None
        self.randomize(None)

        if not self._do_transform:
            return d

        if self.mesh_key not in d.keys():
            if not self.allow_missing_keys:
                msg = f"Missing key: {self.mesh_key}."
                raise KeyError(msg)
        else:
            mesh = d[self.mesh_key]
            if not isinstance(mesh, Mesh):
                msg = f"Expected a Mesh, but got {type(mesh)}."
                raise ValueError(msg)

            origin = mesh.origin
            matrix = (
                Mat4.translation(-origin.x, -origin.y, -origin.z)
                * Mat4.rotation(self.rotation_axis, self.rotation_angle)
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
