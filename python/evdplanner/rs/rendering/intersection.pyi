from evdplanner.rs import Vec3

class IntersectionSort:
    Nearest: "IntersectionSort"
    Farthest: "IntersectionSort"

class Intersection:
    distance: float
    position: Vec3
    normal: Vec3

    def __init__(self, distance: float, position: Vec3, normal: Vec3) -> None: ...
