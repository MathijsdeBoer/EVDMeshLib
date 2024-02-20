"""
The GPURenderer class, which is a GPU-accelerated implementation of the CPURenderer class.
"""
import numpy as np
import taichi as ti
import taichi.math as tm
from loguru import logger

from evdplanner.geometry import Mesh
from evdplanner.rendering import Camera, CameraType, CPURenderer, IntersectionSort


@ti.data_oriented
class GPURenderer(CPURenderer):
    """
    A GPU-accelerated implementation of the CPURenderer class.
    """

    def __init__(
        self,
        camera: Camera,
        mesh: Mesh,
    ) -> None:
        """
        Initialize the GPURenderer.

        Parameters
        ----------
        camera : Camera
            The camera to use for rendering
        mesh : Mesh
            The mesh to render
        """
        super().__init__()

        self.camera = camera
        self.mesh = mesh

        self.x_resolution = self.camera.x_resolution
        self.y_resolution = self.camera.y_resolution

    def render(self, intersection_mode: IntersectionSort, epsilon: float = 1e-8) -> np.ndarray:
        """
        Render the scene using the given intersection mode and epsilon value.

        Parameters
        ----------
        intersection_mode : IntersectionSort
            The intersection mode to use
        epsilon : float
            The epsilon value to use for floating point comparisons

        Returns
        -------
        np.ndarray
            An array of pixels
        """
        ti.init(arch=ti.gpu, default_fp=ti.f64, kernel_profiler=True, enable_fallback=False)

        logger.debug(f"Initializing pixel field with {self.x_resolution=}, {self.y_resolution=}")
        pixels = ti.Vector.field(4, dtype=ti.f64, shape=(self.y_resolution, self.x_resolution))
        pixels.fill(-1.0)

        logger.debug(
            f"Initializing ray directions field with {self.x_resolution=}, {self.y_resolution=}"
        )
        ray_directions = ti.Vector.field(
            3, dtype=ti.f64, shape=(self.y_resolution, self.x_resolution)
        )

        logger.debug(f"Initializing camera origin with {self.camera.origin=}")
        camera_origin = ti.Vector(self.camera.origin.as_float_list())

        logger.debug("Generating triangles from mesh")
        tris = self.mesh.triangles_as_vertex_array()
        triangles = ti.Vector.field(3, dtype=ti.f64, shape=tris.shape[:-1])
        triangles.from_numpy(tris)
        logger.debug(f"{triangles.shape=}")

        logger.debug("Generating normals from mesh")
        norms = [x.normal.as_float_list() for x in self.mesh.triangles]
        normals = ti.Vector.field(3, dtype=ti.f64, shape=(len(norms)))
        normals.from_numpy(np.array(norms))

        @ti.func
        def spherical_to_cartesian(rho: float, theta: float, phi: float) -> ti.Vector:
            """
            Convert spherical coordinates to cartesian coordinates.

            Parameters
            ----------
            rho : float
                The radius
            theta : float
                The azimuthal angle
            phi : float
                The polar angle

            Returns
            -------
            ti.Vector
                The cartesian coordinates
            """
            x = rho * tm.cos(theta) * tm.sin(phi)
            y = rho * tm.sin(theta) * tm.sin(phi)
            z = rho * tm.cos(phi)
            return ti.Vector([x, y, z])

        def generate_ray_directions() -> None:
            """
            Generate the ray directions for the given camera.

            Returns
            -------
            None
            """
            if self.camera.camera_type == CameraType.Equirectangular:

                @ti.kernel
                def generate_ray_directions_kernel() -> None:
                    """
                    Generate the ray directions for the equirectangular camera.

                    Returns
                    -------
                    None
                    """
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
                logger.error(msg)
                raise NotImplementedError(msg)
            elif self.camera.camera_type == CameraType.Orthographic:
                msg = "Orthographic camera not implemented yet"
                logger.error(msg)
                raise NotImplementedError(msg)
            else:
                msg = f"Unknown camera type {self.camera.camera_type}"
                logger.error(msg)
                raise ValueError(msg)

            generate_ray_directions_kernel()

        @ti.func
        def ray_triangle_intersect(
            ray_origin: ti.Vector,
            ray_dir: ti.Vector,
            a: ti.Vector,
            b: ti.Vector,
            c: ti.Vector,
            normal: ti.Vector,
        ) -> tuple[bool, float]:
            """
            Check if a ray intersects a triangle.

            Parameters
            ----------
            ray_origin : ti.Vector
                The origin of the ray
            ray_dir : ti.Vector
                The direction of the ray
            a : ti.Vector
                The first vertex of the triangle
            b : ti.Vector
                The second vertex of the triangle
            c : ti.Vector
                The third vertex of the triangle
            normal : ti.Vector
                The normal of the triangle

            Returns
            -------
            tuple[bool, float]
                A tuple containing a boolean indicating whether the ray intersects the triangle and
                the distance to the intersection point
            """
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
        def flatten_mesh_kernel() -> None:
            """
            Flatten the mesh into a 2D array.

            Returns
            -------
            None
            """
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

        logger.debug("Generating ray directions")
        generate_ray_directions()

        logger.debug("Flattening mesh")
        flatten_mesh_kernel()

        return pixels.to_numpy()
