"""
Bounding Volume Hierarchy (BVH) for ray tracing acceleration.
"""
from evdplanner.rs import Ray
from evdplanner.rs.geometry import Aabb

class BvhNode:
    """
    A node in the BVH tree.

    Attributes
    ----------
    aabb : Aabb
        The axis-aligned bounding box of the node.
    left : int, optional
        The index of the left child node in the BVH tree.
    right : int, optional
        The index of the right child node in the BVH tree.
    triangles : list[int]
        The indices of the triangles in the node.
    """

    aabb: Aabb
    left: int | None
    right: int | None
    triangles: list[int]

class Bvh:
    """
    A bounding volume hierarchy (BVH) for ray tracing acceleration.

    Attributes
    ----------
    nodes : list[BvhNode]
        The nodes in the BVH tree.
    """

    nodes: list[BvhNode]
    def intersect(self, ray: Ray, epsilon: float = 1e-8) -> list[int]:
        """
        Intersect the BVH with a ray.

        Parameters
        ----------
        ray : Ray
            The ray to intersect with the BVH.
        epsilon : float, optional
            The epsilon value to use for intersection tests.

        Returns
        -------
        list[int]
            The indices of the triangles intersected by the ray.
        """
    def trim(self) -> None:
        """
        Trim the BVH tree to remove nodes without triangles.

        Returns
        -------
        None
        """
    @property
    def num_nodes(self) -> int:
        """
        The number of nodes in the BVH tree.
        """
    @property
    def num_levels(self) -> int:
        """
        The number of levels in the BVH tree.
        """
