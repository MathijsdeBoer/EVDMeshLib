"""
Utility functions for rendering and normalizing images.
"""

import numpy as np
from loguru import logger

from evdplanner.geometry import Mesh
from evdplanner.linalg import Vec3
from evdplanner.rendering import Camera, CameraType, IntersectionSort, Renderer


def normalize_image(
    image: np.ndarray, lower_percentile: float = 0.5, upper_percentile: float = 99.5
) -> np.ndarray:
    """
    Normalize the image to 0-1 range.

    Parameters
    ----------
    image : np.ndarray
        Image to normalize.
    lower_percentile : float, optional
        Lower percentile for normalization, by default 0.5
    upper_percentile : float, optional
        Upper percentile for normalization, by default 99.5

    Returns
    -------
    np.ndarray
        Normalized image.
    """
    lower = np.percentile(image, lower_percentile)
    upper = np.percentile(image, upper_percentile)

    return np.clip((image - lower) / (upper - lower), 0, 1)


def spherical_project(
    mesh: Mesh,
    resolution: tuple[int, int],
    forward: Vec3 = Vec3(0, -1, 0),
    up: Vec3 = Vec3(0, 0, 1),
) -> tuple[np.ndarray, np.ndarray]:
    camera = Camera(
        origin=mesh.origin,
        forward=forward,
        up=up,
        x_resolution=resolution[0],
        y_resolution=resolution[1],
        camera_type=CameraType.Equirectangular,
    )

    logger.debug("Rendering spherical projection...")
    renderer = Renderer(camera, mesh)
    skin_render = renderer.render(IntersectionSort.Farthest)
    logger.debug(f"Rendered image shape: {skin_render.shape}")

    skin_depth = skin_render[..., 0]
    skin_normal = skin_render[..., 1:]

    return skin_depth, skin_normal
