import numpy as np

from evdplanner.geometry import Mesh
from evdplanner.linalg import Vec3
from evdplanner.rendering import Camera, CameraType, IntersectionSort, Renderer


def find_closest_intersection(
    mesh: Mesh,
    point: Vec3,
    iterations: int = 3,
    n_samples: int = 256,
    return_camera: bool = False,
) -> Vec3 | tuple[Vec3, Camera]:
    spread = 0.5

    camera = Camera(
        origin=point,
        forward=(mesh.origin - point).unit_vector,
        up=Vec3(0.0, 0.0, 1.0),
        x_resolution=1,
        y_resolution=1,
        camera_type=CameraType.Equirectangular,
    )

    min_distance = float("inf")
    closest_point = None
    closest_x = 0.5
    closest_y = 0.5

    for i in range(iterations):
        for x in np.linspace(closest_x - spread, closest_y + spread, n_samples):
            for y in np.linspace(closest_y - spread, closest_y + spread, n_samples):
                ray = camera.cast_ray(x, y)
                intersection = mesh.intersect(ray, IntersectionSort.Nearest)

                if intersection:
                    distance = intersection.distance

                    if distance < min_distance:
                        min_distance = distance
                        closest_point = intersection.position
                        closest_x = x
                        closest_y = y

        spread /= 8

    if return_camera:
        return closest_point, camera
    return closest_point


def find_closest_intersection_percentiles(
    mesh: Mesh,
    point: Vec3,
    resolution: int = 256,
    percentile: float = 0.05,
    return_mean: bool = False,
) -> list[Vec3] | Vec3:
    camera = Camera(
        origin=point,
        forward=(mesh.origin - point).unit_vector,
        up=Vec3(0.0, 0.0, 1.0),
        x_resolution=resolution,
        y_resolution=resolution,
        camera_type=CameraType.Equirectangular,
    )
    renderer = Renderer(
        mesh=mesh,
        camera=camera,
    )

    render = renderer.render(intersection_mode=IntersectionSort.Nearest)
    depth = render[..., 0]

    indices = np.argwhere(depth < np.percentile(depth, percentile))
    points = []
    for index in indices:
        x, y = index
        intersection = mesh.intersect(camera.cast_ray(x, y), IntersectionSort.Nearest)
        if intersection:
            points.append(intersection.position)

    if return_mean:
        v = Vec3.zero()
        for point in points:
            v += point
        return v / len(points)
    return points
