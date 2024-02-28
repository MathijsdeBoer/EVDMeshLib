"""
The Camera class, which is used to represent a camera in a 3D scene.
"""
from typing import Optional

from evdplanner.rs import Ray
from evdplanner.rs.linalg.vec3 import Vec3

class CameraType:
    """
    An enum representing the different types of cameras.

    Attributes
    ----------
    Perspective: "CameraType"
    Orthographic: "CameraType"
    Equirectangular: "CameraType"
    """

    Perspective: "CameraType"
    Orthographic: "CameraType"
    Equirectangular: "CameraType"

class Camera:
    """
    A class representing a camera in a 3D scene.

    Attributes
    ----------
    origin: Vec3
        The position of the camera in 3D space.
    forward: Vec3
        The direction the camera is facing.
    up: Vec3
        The direction that is considered "up" for the camera.
    right: Vec3
        The direction that is considered "right" for the camera.
    x_resolution: int
        The horizontal resolution of the camera.
    y_resolution: int
        The vertical resolution of the camera.
    camera_type: CameraType
        The type of camera.
    fov: Optional[float]
        The field of view of the camera, in degrees. Only used for perspective cameras.
    aspect_ratio: Optional[float]
        The aspect ratio of the camera. Only used for perspective cameras.
    size: Optional[float]
        The size of the camera. Only used for orthographic cameras.
    """

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
    ) -> None:
        """
        Initialize the Camera object.

        Parameters
        ----------
        origin : Vec3
            The position of the camera in 3D space.
        forward : Vec3
            The direction the camera is facing.
        up : Vec3
            The direction that is considered "up" for the camera.
        x_resolution : int
            The horizontal resolution of the camera.
        y_resolution : int
            The vertical resolution of the camera.
        camera_type : CameraType
            The type of camera.
        fov : float, optional
            The field of view of the camera, in degrees. Only used for perspective cameras.
        aspect_ratio : float, optional
            The aspect ratio of the camera. Only used for perspective cameras.
        size : float, optional
            The size of the camera. Only used for orthographic cameras.
        """
    def cast_ray(self, x: float, y: float) -> Ray:
        """
        Cast a ray from the camera through the specified pixel.

        Parameters
        ----------
        x : int
            The x-coordinate of the pixel.
        y : int
            The y-coordinate of the pixel.

        Returns
        -------
        Ray
            The ray that was cast from the camera through the specified pixel.
        """
    def project_back(self, point: Vec3, normalized: bool = True) -> tuple[float, float]:
        """
        Project a 3D point back onto the camera's image plane.

        Parameters
        ----------
        point : Vec3
            The 3D point to project back onto the camera's image plane.
        normalized : bool, optional
            Whether to return the coordinates in normalized space (between 0 and 1) or not.

        Returns
        -------
        tuple[float, float]
            The x and y coordinates of the projected point on the camera's image plane.
        """
    def translate(self, vec: Vec3) -> None:
        """
        Translate the camera by the specified vector.

        Parameters
        ----------
        vec : Vec3
            The vector by which to translate the camera.
        """
    def rotate(self, axis: Vec3, angle: float) -> None:
        """
        Rotate the camera around the specified axis by the specified angle.

        Parameters
        ----------
        axis : Vec3
            The axis around which to rotate the camera.
        angle : float
            The angle by which to rotate the camera, in radians.
        """
