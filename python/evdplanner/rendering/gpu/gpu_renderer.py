import numpy as np
import taichi as ti
import taichi.math as tm
from evdplanner.rs import Mesh, Vec3
from evdplanner.rs.rendering import Camera, CPURenderer, IntersectionSort


@ti.data_oriented
class GPURenderer(CPURenderer):
    def __init__(
        self,
        camera: Camera,
        x_resolution: int,
        y_resolution: int,
        mesh: Mesh,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()

        self.camera = camera
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution

        self.mesh = mesh

        self.epsilon = epsilon

    def render(self, intersection_mode: IntersectionSort) -> np.ndarray:
        ti.init(arch=ti.gpu, default_fp=ti.f64, kernel_profiler=True)
        pixels = ti.Vector.field(4, dtype=ti.f64, shape=(self.y_resolution, self.x_resolution))
        pixels.fill(0.0)

        ray_dirs = [[Vec3.zero()] * self.x_resolution] * self.y_resolution
        for y in range(self.y_resolution):
            for x in range(self.x_resolution):
                ray_dirs[y][x] = self.camera.cast_ray(
                    x / self.x_resolution, y / self.y_resolution
                ).direction.as_float_list()

                print(f"{x=}, {y=}")
                print(f"{x / self.x_resolution=}, {y / self.y_resolution=}")
                print(f"{ray_dirs[y][x]=}")

        ray_directions = ti.Vector.field(3, dtype=ti.f64, shape=(self.y_resolution, self.x_resolution))
        ray_directions.from_numpy(np.array(ray_dirs))

        camera_origin = ti.Vector(self.camera.origin.as_float_list())

        tris = self.mesh.triangles_as_vertex_array()
        triangles = ti.Vector.field(3, dtype=ti.f64, shape=tris.shape[:-1])
        triangles.from_numpy(tris)

        print(f"{triangles.shape=}")

        norms = [x.as_float_list() for x in self.mesh.normals]
        normals = ti.Vector.field(3, dtype=ti.f64, shape=(len(norms)))
        normals.from_numpy(np.array(norms))

        @ti.func
        def _ray_triangle_intersection(
                ray_origin,
                ray_direction,
                a,
                b,
                c,
                normal,
        ) -> tuple[bool, ti.f64]:
            intersection = False
            t = -1.0

            # Check if ray intersects plane
            if normal.dot(ray_direction) > self.epsilon:
                d = -normal.dot(a)
                t = -(normal.dot(ray_origin) + d) / normal.dot(ray_direction)

                if t > 0.0:
                    p = ray_origin + t * ray_direction

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
                        intersection = True

            return intersection, t

        @ti.kernel
        def _flatten_mesh_kernel():
            ti.loop_config(
                block_dim=128,
            )
            for y, x in pixels:
                for k in range(triangles.shape[0]):
                    a = triangles[k, 0]
                    b = triangles[k, 1]
                    c = triangles[k, 2]
                    normal = normals[k]

                    direction = ray_directions[y, x]

                    intersect, t = _ray_triangle_intersection(
                        camera_origin, direction, a, b, c, normal
                    )
                    if intersection_mode == IntersectionSort.Nearest:
                        if intersect and t < pixels[y, x][0]:
                            pixels[y, x][0] = t
                            pixels[y, x][1] = normal[0]
                            pixels[y, x][2] = normal[1]
                            pixels[y, x][3] = normal[2]
                    elif intersection_mode == IntersectionSort.Farthest:
                        if intersect and t > pixels[y, x][0]:
                            pixels[y, x][0] = t
                            pixels[y, x][1] = normal[0]
                            pixels[y, x][2] = normal[1]
                            pixels[y, x][3] = normal[2]

        return pixels.to_numpy()
