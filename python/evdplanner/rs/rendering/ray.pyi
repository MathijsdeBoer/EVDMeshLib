"""
The Ray class, which is used to represent a ray in 3D space.
"""
from evdplanner.rs import Vec3

class Ray:
    """
    A class to represent a ray in 3D space.

    Attributes
    ----------
    origin : Vec3
        The origin of the ray.
    direction : Vec3
        The direction of the ray.
    """

    origin: Vec3
    direction: Vec3
    def __init__(self, origin: Vec3, direction: Vec3) -> None:
        """
        Constructs a new Ray object.

        Parameters
        ----------
        origin : Vec3
            The origin of the ray.
        direction : Vec3
            The direction of the ray.
        """
    def at(self, t: float) -> Vec3:
        """
        Returns the point at a given distance along the ray.

        Parameters
        ----------
        t : float
            The distance along the ray.

        Returns
        -------
        Vec3
            The point at the given distance along the ray.
        """
    def __repr__(self) -> str:
        """
        Returns a string representation of the ray.

        Returns
        -------
        str
            A string representation of the ray.
        """
    def __str__(self) -> str:
        """
        Returns a string representation of the ray.

        Returns
        -------
        str
            A string representation of the ray.
        """
