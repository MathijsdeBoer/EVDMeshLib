import mcubes
import numpy as np
from evdplanner.geometry import Mesh
from evdplanner.linalg import Vec3


def volume_to_mesh(
    volume: np.ndarray,
    origin: Vec3 | tuple[float, float, float] = Vec3.zero(),
    spacing: Vec3 | tuple[float, float, float] = Vec3.one(),
    num_samples: int = 1_000_000,
) -> Mesh:
    """Convert a volume to a mesh."""
    if isinstance(origin, tuple):
        origin = Vec3(*origin)
    if isinstance(spacing, tuple):
        spacing = Vec3(*spacing)

    # Pad the volume with zeros to avoid edge effects
    volume = np.pad(volume, 1, mode="constant")

    # Extract the mesh from the volume
    vertices, triangles = mcubes.marching_cubes(mcubes.smooth(volume), 0.0)

    vertices = [
        Vec3(
            vertex[0] * spacing[0],
            vertex[1] * spacing[1],
            vertex[2] * spacing[2],
        )
        - spacing
        + origin
        for vertex in vertices
    ]
    faces = [tuple(triangle) for triangle in triangles]

    # Convert the mesh to the correct format
    mesh = Mesh(
        Vec3.zero(),
        vertices,
        faces,
    )
    mesh.recalculate_origin(num_samples)

    return mesh
