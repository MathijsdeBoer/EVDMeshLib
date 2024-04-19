"""
Intersection and IntersectionSort classes for use in the EVDPlanner class.
"""
from evdplanner.rs import Vec3

class IntersectionSort:
    """
    A class to select the nearest or farthest intersection from a list of intersections.
    """

    Nearest: IntersectionSort
    Farthest: IntersectionSort

class Intersection:
    """
    A class to represent an intersection between a ray and a surface.

    Attributes
    ----------
    distance : float
        The distance from the ray origin to the intersection point.
    position : Vec3
        The position of the intersection point.
    normal : Vec3
        The normal of the surface at the intersection point.
    """

    distance: float
    position: Vec3
    normal: Vec3

    def __init__(self, distance: float, position: Vec3, normal: Vec3) -> None:
        """
        Initialize an Intersection object.

        Parameters
        ----------
        distance : float
            The distance from the ray origin to the intersection point.
        position : Vec3
            The position of the intersection point.
        normal : Vec3
            The normal of the surface at the intersection point.
        """
