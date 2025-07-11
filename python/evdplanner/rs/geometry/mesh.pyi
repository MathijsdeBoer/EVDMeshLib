"""
Mesh class for representing 3D triangle meshes.
"""

import numpy as np

from evdplanner.rs import Deformer, Intersection, IntersectionSort, Mat4, Ray, Vec3

class Triangle:
    """
    A triangle in 3D space.

    Attributes
    ----------
    a : int
        Index of the first vertex.
    b : int
        Index of the second vertex.
    c : int
        Index of the third vertex.
    normal : Vec3
        Normal vector of the triangle.
    area : float
        Area of the triangle.
    """

    a: Vec3
    b: Vec3
    c: Vec3
    normal: Vec3
    area: float

class Mesh:
    """
    A 3D triangle mesh.

    Attributes
    ----------
    origin : Vec3
        Origin of the mesh.
    vertices : list[Vec3]
        List of vertices.
    triangles : list[Triangle]
        List of triangles.
    """

    origin: Vec3
    triangles: list[Triangle]

    def __init__(
        self, origin: Vec3, vertices: list[Vec3], triangles: list[tuple[int, int, int]]
    ) -> None:
        """
        Initialize a mesh.

        Parameters
        ----------
        origin : Vec3
            Origin of the mesh.
        vertices : list[Vec3]
            List of vertices.
        triangles : list[tuple[int, int, int]]
            List of triangles, each represented by a tuple of three vertex indices.
        """

    @staticmethod
    def load(path: str, num_samples: int = 10_000) -> Mesh:
        """
        Load a mesh from a file.

        Parameters
        ----------
        path : str
            Path to the file.
        num_samples : int, optional
            Number of samples to use for origin calculation, by default 10_000.

        Returns
        -------
        Mesh
            The loaded mesh.
        """

    def save(self, path: str) -> None:
        """
        Save the mesh to a file.

        Parameters
        ----------
        path : str
            Path to the file.

        Returns
        -------
        None
        """

    def transform(self, transform: Mat4) -> None:
        """
        Transform the mesh.

        Parameters
        ----------
        transform : Mat4
            Transformation matrix.

        Returns
        -------
        None
        """

    def deform(self, deform: Deformer) -> None:
        """
        Deform the mesh.

        Parameters
        ----------
        deform : Deformer
            Deformer that defines the noise characteristics.

        Returns
        -------
        None
        """

    def recalculate_normals(self) -> None:
        """
        Recalculate the normals of the triangles.

        Returns
        -------
        None
        """

    def recalculate_areas(self) -> None:
        """
        Recalculate the areas of the triangles.

        Returns
        -------
        None
        """

    def recalculate_origin(self, num_samples: int = 1_000_000) -> None:
        """
        Recalculate the origin of the mesh.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to use for origin calculation, by default 1_000_000.

        Returns
        -------
        None
        """

    def recalculate(self, num_samples: int = 1_000_000) -> None:
        """
        Rebuild the BVH and recalculate the normals, areas, and origin of the mesh.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to use for origin calculation, by default 1_000_000.

        Returns
        -------
        None
        """

    def update_triangles(self) -> None:
        """
        Update the triangles of the mesh.

        Returns
        -------
        None
        """

    def rebuild_bvh(self) -> None:
        """
        Rebuild the BVH.

        Returns
        -------
        None
        """

    def flatten_bvh(self) -> None:
        """
        Flatten the BVH.

        Returns
        -------
        None
        """

    def print_bvh(self) -> None:
        """
        Print the BVH.

        Returns
        -------
        None
        """

    def intersect(
        self, ray: Ray, sorting: IntersectionSort, epsilon: float = 1e-8
    ) -> Intersection | None:
        """
        Intersect the mesh with a ray.

        Parameters
        ----------
        ray : Ray
            The ray to intersect with.
        sorting : IntersectionSort
            Sorting method for the intersections.
        epsilon : float, optional
            Epsilon value for intersection calculations, by default 1e-8.

        Returns
        -------
        Intersection | None
            The intersection, or None if no intersection was found.
        """

    def triangles_as_vertex_array(self) -> np.ndarray:
        """
        Get the triangles as a vertex array.

        Returns
        -------
        np.ndarray
            The vertex array.
        """

    def laplacian_smooth(self, iterations: int = 10, smoothing_factor: float = 0.5) -> None:
        """
        Smooth the mesh using the Laplacian smoothing algorithm.

        Parameters
        ----------
        iterations : int, optional
            Number of iterations, by default 10.
        smoothing_factor : float, optional
            Smoothing factor, by default 0.5.

        Returns
        -------
        None
        """

    @property
    def num_triangles(self) -> int:
        """
        Number of triangles in the mesh.

        Returns
        -------
        int
            Number of triangles.
        """

    @property
    def surface_area(self) -> float:
        """
        Surface area of the mesh.

        Returns
        -------
        float
            Surface area.
        """

    @property
    def volume(self) -> float:
        """
        Volume of the mesh.

        Returns
        -------
        float
            Volume.
        """
