import numpy as np
from evdplanner.rs import Camera, IntersectionSort, Mesh

class CPURenderer:
    camera: Camera
    mesh: Mesh
    def __init__(self, camera: Camera, mesh: Mesh) -> None: ...
    def render(self, intersection_mode: IntersectionSort, epsilon: float = 1e-8) -> np.ndarray: ...
