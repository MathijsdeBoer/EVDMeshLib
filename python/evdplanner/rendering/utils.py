"""
Utility functions for rendering and normalizing images.
"""
import numpy as np


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
