import numpy as np
from evdplanner.rs import Camera, IntersectionSort, Mesh

class CPURenderer:
    camera: Camera
    mesh: Mesh
    def __init__(
        self, camera: Camera, mesh: Mesh
    ) -> None: ...
    def render(self, intersection_mode: IntersectionSort) -> np.ndarray: ...
