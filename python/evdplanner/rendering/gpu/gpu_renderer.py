import logging

import numpy as np
import taichi as ti
import taichi.math as tm

from evdplanner.geometry import Mesh
from evdplanner.rendering import Camera, CameraType, CPURenderer, IntersectionSort


_logger = logging.getLogger(__name__)


@ti.data_oriented
class GPURenderer(CPURenderer):
    def __init__(
        self,
        camera: Camera,
        mesh: Mesh,
    ) -> None:
        super().__init__()

        self.camera = camera
        self.mesh = mesh

        self.x_resolution = self.camera.x_resolution
        self.y_resolution = self.camera.y_resolution

    def render(self, intersection_mode: IntersectionSort, epsilon: float = 1e-8) -> np.ndarray:
        ti.init(arch=ti.gpu, default_fp=ti.f64, kernel_profiler=True, enable_fallback=False)

        _logger.debug(f"Initializing pixel field with {self.x_resolution=}, {self.y_resolution=}")
        pixels = ti.Vector.field(4, dtype=ti.f64, shape=(self.y_resolution, self.x_resolution))
        pixels.fill(-1.0)

        _logger.debug(f"Initializing ray directions field with {self.x_resolution=}, {self.y_resolution=}")
        ray_directions = ti.Vector.field(
            3, dtype=ti.f64, shape=(self.y_resolution, self.x_resolution)
        )

        _logger.debug(f"Initializing camera origin with {self.camera.origin=}")
        camera_origin = ti.Vector(self.camera.origin.as_float_list())

        _logger.debug(f"Generating triangles from mesh")
        tris = self.mesh.triangles_as_vertex_array()
        triangles = ti.Vector.field(3, dtype=ti.f64, shape=tris.shape[:-1])
        triangles.from_numpy(tris)
        _logger.debug(f"{triangles.shape=}")

        _logger.debug(f"Generating normals from mesh")
        norms = [x.normal.as_float_list() for x in self.mesh.triangles]
        normals = ti.Vector.field(3, dtype=ti.f64, shape=(len(norms)))
        normals.from_numpy(np.array(norms))

        @ti.func
        def spherical_to_cartesian(rho, theta, phi):
            x = rho * tm.cos(theta) * tm.sin(phi)
            y = rho * tm.sin(theta) * tm.sin(phi)
            z = rho * tm.cos(phi)
            return ti.Vector([x, y, z])

        def generate_ray_directions():
            if self.camera.camera_type == CameraType.Equirectangular:

                @ti.kernel
                def generate_ray_directions_kernel():
                    for pix_y, pix_x in pixels:
                        theta = (
                            2.0 * tm.pi * (pix_x / self.x_resolution) + self.camera.theta_offset
                        )
                        phi = tm.pi * (pix_y / self.y_resolution)
                        ray_directions[pix_y, pix_x] = spherical_to_cartesian(
                            1.0, theta, phi
                        ).normalized()

            elif self.camera.camera_type == CameraType.Perspective:
                msg = "Perspective camera not implemented yet"
                _logger.error(msg)
                raise NotImplementedError(msg)
            elif self.camera.camera_type == CameraType.Orthographic:
                msg = "Orthographic camera not implemented yet"
                _logger.error(msg)
                raise NotImplementedError(msg)
            else:
                msg = f"Unknown camera type {self.camera.camera_type}"
                _logger.error(msg)
                raise ValueError(msg)

            generate_ray_directions_kernel()

        @ti.func
        def ray_triangle_intersect(
            ray_origin,
            ray_dir,
            a,
            b,
            c,
            normal,
        ) -> tuple[bool, float]:
            intersect = False
            t = -1.0

            # Check if ray intersects plane
            if normal.dot(ray_dir) > epsilon:
                d = -normal.dot(a)
                t = -(normal.dot(ray_origin) + d) / normal.dot(ray_dir)

                if t > 0.0:
                    p = ray_origin + t * ray_dir

                    # Check if point is inside triangle
                    ab = b - a
                    bc = c - b
                    ca = a - c

                    ap = p - a
                    bp = p - b
                    cp = p - c

                    if (
                        normal.dot(ab.cross(ap)) > 0.0
                        and normal.dot(bc.cross(bp)) > 0.0
                        and normal.dot(ca.cross(cp)) > 0.0
                    ):
                        intersect = True

            return intersect, t

        @ti.kernel
        def flatten_mesh_kernel():
            ti.loop_config(
                block_dim=128,
            )
            for pix_y, pix_x in pixels:
                ray_dir = ray_directions[pix_y, pix_x]

                for k in range(triangles.shape[0]):
                    a = triangles[k, 0]
                    b = triangles[k, 1]
                    c = triangles[k, 2]
                    normal = normals[k]

                    intersect, t = ray_triangle_intersect(camera_origin, ray_dir, a, b, c, normal)

                    if intersection_mode == IntersectionSort.Nearest:
                        if intersect and t < pixels[pix_y, pix_x][0]:
                            pixels[pix_y, pix_x][0] = t
                            pixels[pix_y, pix_x][1] = normal[0]
                            pixels[pix_y, pix_x][2] = normal[1]
                            pixels[pix_y, pix_x][3] = normal[2]
                    elif intersection_mode == IntersectionSort.Farthest:
                        if intersect and t > pixels[pix_y, pix_x][0]:
                            pixels[pix_y, pix_x][0] = t
                            pixels[pix_y, pix_x][1] = normal[0]
                            pixels[pix_y, pix_x][2] = normal[1]
                            pixels[pix_y, pix_x][3] = normal[2]

        _logger.debug("Generating ray directions")
        generate_ray_directions()

        _logger.debug("Flattening mesh")
        flatten_mesh_kernel()

        return pixels.to_numpy()
