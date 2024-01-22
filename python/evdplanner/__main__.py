from itertools import product
from math import pi

import numpy as np
from evdplanner.rendering.gpu import GPURenderer
from evdplanner.rs import Camera, CameraType, CPURenderer, IntersectionSort, Mesh, Vec3
from imageio import imwrite


def _normalize_map(
    map: np.ndarray, lower_percentile: float = 0.5, upper_percentile: float = 99.5
) -> np.ndarray:
    """Normalize the map to 0-1 range."""
    lower = np.percentile(map, lower_percentile)
    upper = np.percentile(map, upper_percentile)

    return np.clip((map - lower) / (upper - lower), 0, 1)


if __name__ == "__main__":
    mesh = Mesh.from_file(
        r"D:\HANARTH\EVDKeypoints\Samples\Pierre 274850\mesh_skin_decimated.stl", 10_000_000
    )
    resolution = 64

    x_max = 8
    y_max = 4
    for (x, y) in product(range(x_max), range(y_max)):
        print(f"{x}, {y}")
        phi = pi * (y / y_max)
        theta = 2 * pi * (x / x_max) + (0.5 * pi)
        v = Vec3(1.0, theta, phi)
        print(f"{v=} ({v.length})")
        direction = Vec3.spherical_to_cartesian(v)
        print(f"{direction=} ({direction.length})")

    # camera = Camera(
    #     mesh.origin,
    #     forward=Vec3(0, -1, 0),
    #     up=Vec3(0, 0, 1),
    #     x_resolution=resolution,
    #     y_resolution=resolution // 2,
    #     camera_type=CameraType.Equirectangular,
    #     theta_offset=0.5 * pi,
    # )
    #
    # renderer = CPURenderer(camera, mesh)
    # render = renderer.render(IntersectionSort.Nearest)
    #
    # depth_image = render[..., 0]
    # normal_image = render[..., 1:]
    # depth_image = _normalize_map(depth_image)
    #
    # print(depth_image.min(), depth_image.max())
    #
    # depth_image = (depth_image * 65535).astype(np.uint16)
    # normal_image = (normal_image * 255).astype(np.uint8)
    #
    # imwrite("depth.png", depth_image)
    # imwrite("normal.png", normal_image)
