"""
The CPURenderer class, which is used to render a mesh from a given camera.
"""
import numpy as np

from evdplanner.rs import Camera, Intersection, IntersectionSort, Mesh

class CPURenderer:
    """
    A class to render a mesh from a given camera.

    Attributes
    ----------
    camera : Camera
        The camera to render the mesh from.
    mesh : Mesh
        The mesh to render.
    """

    camera: Camera
    mesh: Mesh
    def __init__(self, camera: Camera, mesh: Mesh) -> None:
        """
        Initializes the CPURenderer.

        Parameters
        ----------
        camera : Camera
            The camera to render the mesh from.
        mesh : Mesh
            The mesh to render.
        """
    def render(self, intersection_mode: IntersectionSort, epsilon: float = 1e-8) -> np.ndarray:
        """
        Renders the mesh from the camera.

        Parameters
        ----------
        intersection_mode : IntersectionSort
            The mode to use for sorting the intersections.
        epsilon : float, optional
            The epsilon value to use for floating point comparisons.

        Returns
        -------
        np.ndarray
            The rendered image.
        """
    def generate_intersections(
        self, intersection_mode: IntersectionSort, epsilon: float = 1e-8
    ) -> list[tuple[tuple[int, int], Intersection]]: ...
