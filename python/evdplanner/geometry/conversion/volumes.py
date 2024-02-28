"""
Functions for converting volumes to meshes.
"""

import mcubes
import numpy as np
import SimpleITK as sitk
from loguru import logger

from evdplanner.geometry import Mesh
from evdplanner.linalg import Vec3


def volume_to_mesh(
    volume: sitk.Image,
) -> Mesh:
    """
    Convert a volume to a mesh.

    Parameters
    ----------
    volume : np.ndarray
        The volume to convert to a mesh.
    num_samples : int
        The number of samples to use when recalculating the origin of the mesh.

    Returns
    -------
    Mesh
        The mesh generated from the volume.
    """
    array = sitk.GetArrayFromImage(volume)

    # Pad the volume with zeros to avoid edge effects
    array = np.pad(array, 1, mode="constant")
    logger.debug(f"Padded volume shape: {array.shape}")

    # SimpleITK axis order is (x, y, z), after conversion to numpy it becomes (z, y, x)
    # We need to swap the axes to get the correct order for the indexing later on
    array = np.swapaxes(array, 0, 2)

    # Extract the mesh from the volume
    logger.info("Extracting mesh from volume...")
    vertices, triangles = mcubes.marching_cubes(mcubes.smooth(array), 0.0)

    logger.debug(f"Number of vertices: {len(vertices)}")
    logger.debug(f"Number of triangles: {len(triangles)}")

    # Convert the vertices to the correct format
    logger.info("Converting vertices to the correct format...")
    vertices = [
        Vec3(
            *volume.TransformContinuousIndexToPhysicalPoint(vertex),
        )
        for vertex in vertices
    ]
    logger.debug(f"Vertices: {vertices[:5]}...")
    logger.info("Converting triangles to the correct format...")
    faces = [tuple(triangle) for triangle in triangles]

    # Convert the mesh to the correct format
    logger.info("Converting mesh to the correct format...")
    mesh = Mesh(
        Vec3.zero(),
        vertices,
        faces,
    )

    return mesh
