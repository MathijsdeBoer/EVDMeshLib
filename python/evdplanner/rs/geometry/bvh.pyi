from evdplanner.rs import Ray
from evdplanner.rs.geometry import Aabb

class BvhNode:
    aabb: Aabb
    left: int | None
    right: int | None
    triangles: list[int]

class Bvh:
    nodes: list[BvhNode]
    def intersect(self, ray: Ray, epsilon: float = 1e-8) -> list[int]: ...
    def trim(self): ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_levels(self) -> int: ...
