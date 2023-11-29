use pyo3::prelude::*;

use crate::linalg::Vec3;
use crate::rendering::cameras::BaseCamera;
use crate::rendering::cameras::projector::Projector;
use crate::rendering::Ray;

#[pyclass(name = "ProjectionCamera", extends = BaseCamera)]
pub struct PerspectiveCamera {
    #[pyo3(get, set)]
    pub position: Vec3,
    #[pyo3(get, set)]
    pub forward: Vec3,
    #[pyo3(get, set)]
    pub up: Vec3,
    #[pyo3(get, set)]
    pub right: Vec3,
    #[pyo3(get, set)]
    pub fov: f64,
    #[pyo3(get, set)]
    pub aspect_ratio: f64
}

#[pymethods]
impl PerspectiveCamera {
    #[new]
    pub fn new(position: Vec3, forward: Vec3, up: Vec3, fov: f64, aspect_ratio: f64) -> Self {
        let right = forward.cross(&up).normalize();
        let up = right.cross(forward).normalize();

        Self {
            position,
            forward,
            up,
            right,
            fov,
            aspect_ratio
        }
    }

    pub fn ray_from_screenspace(&self, x: f64, y: f64) -> Ray {
        let x = (x - 0.5) * self.aspect_ratio * self.fov;
        let y = (y - 0.5) * self.fov;

        let direction = self.forward + self.right * x + self.up * y;
        Ray::new(self.position, direction.normalize())
    }

    fn position_to_screenspace(&self, position: Vec3) -> (f64, f64) {
        let direction = position - self.position;
        let x = direction.dot(&self.right) / self.aspect_ratio;
        let y = direction.dot(&self.up);
        (x, y)
    }
}

impl Projector for PerspectiveCamera {
    fn ray_from_screenspace(&self, x: f64, y: f64) -> Ray {
        PerspectiveCamera::ray_from_screenspace(self, x, y)
    }

    fn position_to_screenspace(&self, position: Vec3) -> (f64, f64) {
        PerspectiveCamera::position_to_screenspace(self, position)
    }
}
