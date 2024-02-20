"""
Axis-aligned bounding box.
"""
from evdplanner.rs import Ray, Vec3

class Aabb:
    """
    Axis-aligned bounding box.

    Attributes
    ----------
    min : Vec3
        Minimum corner of the box
    max : Vec3
        Maximum corner of the box
    """

    min: Vec3
    max: Vec3

    def __init__(self) -> None:
        """
        Create an empty bounding box.
        """
    def intersect(self, ray: Ray, epsilon: float = 1e-8) -> bool:
        """
        Check if a ray intersects the box.

        Parameters
        ----------
        ray : Ray
            Ray to check for intersection
        epsilon : float, optional
            Small value to avoid self-intersection

        Returns
        -------
        bool
            True if the ray intersects the box
        """
    def grow(self, v: Vec3) -> None:
        """
        Grow the box to include a point.

        Parameters
        ----------
        v : Vec3
            Point to include in the box

        Returns
        -------
        None
        """
    @property
    def longest_axis(self) -> int:
        """
        Get the index of the longest axis.

        Returns
        -------
        int
            Index of the longest axis

        Returns
        -------
        int
            Index of the longest axis
        """
