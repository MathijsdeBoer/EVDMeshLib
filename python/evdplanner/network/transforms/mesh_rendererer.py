from pathlib import Path
from typing import Hashable, Mapping

import monai.transforms as mt
import numpy as np
from monai.data import MetaTensor
from monai.utils import MetaKeys

from evdplanner.geometry import Mesh
from evdplanner.linalg import Vec3
from evdplanner.rendering import CPURenderer, Camera, CameraType, IntersectionSort


class MeshRenderd(mt.MapTransform):
    def __init__(
            self,
            keys: list[str],
            x_resolution: int,
            y_resolution: int,
            render_key: str = "image",
            intersection_sort: IntersectionSort = IntersectionSort.Farthest,
            allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

        self.render_key = render_key

        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.intersection_sort = intersection_sort

    def __call__(self, data: Mapping[Hashable, Path | Mesh]) -> dict:
        d = dict(data)

        if self.mesh_key not in d.keys():
            if not self.allow_missing_keys:
                msg = f"Missing key: {self.mesh_key}."
                raise KeyError(msg)
        else:
            mesh = d[self.mesh_key]
            if not isinstance(mesh, Mesh):
                msg = f"Expected a Mesh, but got {type(mesh)}."
                raise ValueError(msg)

            renderer = CPURenderer(
                camera=Camera(
                    origin=mesh.origin,
                    forward=Vec3(0.0, -1.0, 0.0),
                    up=Vec3(0.0, 0.0, 1.0),
                    camera_type=CameraType.Equirectangular,
                    x_resolution=self.x_resolution,
                    y_resolution=self.y_resolution,
                ),
                mesh=mesh,
            )

            render = renderer.render(self.intersection_sort)
            render = np.transpose(render, (2, 1, 0))

            d[self.render_key] = MetaTensor(
                x=render,
                meta={
                    "origin": mesh.origin,
                    MetaKeys.ORIGINAL_CHANNEL_DIM: 0,
                    MetaKeys.SPATIAL_SHAPE: render.shape[1:],
                }
            )

        return d
