import numpy as np

from evdplanner.rs import Camera, Mesh, Vec3

def find_target(
    mesh: Mesh,
    origin: Vec3,
    n_steps: int = 128,
    n_iter: int = 3,
    initial_fov: float = 45.0,
    check_radially: bool = False,
    radius: float = 4.0,
    radial_samples: int = 8,
    radial_rings: int = 2,
    objective_distance_weight: float = 0.5,
    thickness_threshold: float = 10.0,
    depth_threshold: float = 70.0,
    epsilon: float = 1e-8,
) -> tuple[Vec3, float]: ...
def objective_function(
    distance: float,
    thickness: float,
    distance_weight: float,
    thickness_threshold: float,
    depth_threshold: float,
) -> float: ...
def generate_objective_image(
    mesh: Mesh,
    camera: Camera,
    check_radially: bool = False,
    radius: float = 4.0,
    radial_samples: int = 8,
    radial_rings: int = 2,
    objective_distance_weight: float = 0.5,
    thickness_threshold: float = 10.0,
    depth_threshold: float = 70.0,
    epsilon: float = 1e-8,
) -> np.ndarray: ...
