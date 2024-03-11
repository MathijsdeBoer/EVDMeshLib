from evdplanner.geometry import Mesh
from evdplanner.linalg import Vec3
from evdplanner.rendering import Camera, IntersectionSort


def generate_landmarks(
    projection: list[tuple[float, float]],
    mesh: Mesh,
    camera: Camera,
    intersection_sort: IntersectionSort,
) -> list[Vec3]:
    points = []
    for u, v in projection:
        ray = camera.cast_ray(u, v)
        point = mesh.intersect(ray, intersection_sort)
        points.append(point.position)
    return points
