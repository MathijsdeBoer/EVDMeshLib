from typing import Optional

from evdplanner.rs import Ray
from evdplanner.rs.linalg.vec3 import Vec3

class CameraType:
    Perspective: "CameraType"
    Orthographic: "CameraType"
    Equirectangular: "CameraType"

class Camera:
    origin: Vec3
    forward: Vec3
    up: Vec3
    right: Vec3
    x_resolution: int
    y_resolution: int
    camera_type: CameraType
    fov: Optional[float]
    aspect_ratio: Optional[float]
    size: Optional[float]
    theta_offset: Optional[float]

    def __init__(
        self,
        origin: Vec3,
        forward: Vec3,
        up: Vec3,
        x_resolution: int,
        y_resolution: int,
        camera_type: CameraType,
        fov: Optional[float] = None,
        aspect_ratio: Optional[float] = None,
        size: Optional[float] = None,
        theta_offset: Optional[float] = None,
    ) -> None: ...
    def cast_ray(self, x: float, y: float) -> Ray: ...
    def translate(self, vec: Vec3) -> None: ...
    def rotate(self, axis: Vec3, angle: float) -> None: ...
