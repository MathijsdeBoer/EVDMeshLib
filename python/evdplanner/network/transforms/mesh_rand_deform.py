import random
from pathlib import Path
from typing import Hashable, Mapping

import monai.transforms as mt

from evdplanner.geometry import Mesh
from evdplanner.rs import Deformer


class MeshRandDeformed(mt.MapTransform):
    def __init__(
            self,
            mesh_key: str,
            landmarks_key: str,
            amplitude: float = 1.0,
            scale: float = 15.0,
            frequency: float = 0.0025,
            octaves: int = 6,
            persistence: float = 0.5,
            lacunarity: float = 2.0,
            seed: int = 42,
            allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys=[mesh_key, landmarks_key], allow_missing_keys=allow_missing_keys)

        self.mesh_key = mesh_key
        self.landmarks_key = landmarks_key

        self.amplitude = amplitude
        self.scale = scale
        self.frequency = frequency
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity

        self.seed = seed
        self.deformer_seed_generator = random.Random(seed)

    def __call__(self, data: Mapping[Hashable, Path | Mesh]) -> dict:
        d = dict(data)

        deformer = Deformer(
            scale=self.scale,
            amplitude=self.amplitude,
            frequency=self.frequency,
            octaves=self.octaves,
            persistence=self.persistence,
            lacunarity=self.lacunarity,
            seed=self.deformer_seed_generator.randint(0, 2**32 - 1)
        )

        if self.mesh_key not in d.keys():
            if not self.allow_missing_keys:
                msg = f"Missing key: {self.mesh_key}."
                raise KeyError(msg)
        else:
            mesh = d[self.mesh_key]
            if not isinstance(mesh, Mesh):
                msg = f"Expected a Mesh, but got {type(mesh)}."
                raise ValueError(msg)

            mesh.deform(deformer)
            d[self.mesh_key] = mesh

        if self.landmarks_key not in d.keys():
            if not self.allow_missing_keys:
                msg = f"Missing key: {self.landmarks_key}."
                raise KeyError(msg)
        else:
            landmarks, names = d[self.landmarks_key]

            for i in range(len(landmarks)):
                landmarks[i] = deformer.deform_vertex(landmarks[i])
            d[self.landmarks_key] = landmarks, names

        return d
