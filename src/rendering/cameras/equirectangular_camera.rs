use pyo3::prelude::*;
use crate::linalg::Vec3;
use crate::rendering::cameras::BaseCamera;
use crate::rendering::cameras::projector::Projector;

#[pyclass(name = "EquirectangularCamera", extends = BaseCamera)]
pub struct EquirectangularCamera {
    #[pyo3(get, set)]
    pub position: Vec3,
    #[pyo3(get, set)]
    pub forward: Vec3,
    #[pyo3(get, set)]
    pub up: Vec3,
    #[pyo3(get, set)]
    pub right: Vec3,
}

#[pymethods]
impl EquirectangularCamera {
    #[new]
    pub fn new(position: Vec3, forward: Vec3, up: Vec3, aspect_ratio: f64) -> Self {
        let right = forward.cross(&up).normalize();
        let up = right.cross(forward).normalize();

        Self {
            position,
            forward,
            up,
            right,
        }
    }
}

impl Projector for EquirectangularCamera {
    fn ray_from_screenspace(&self, x: f64, y: f64) -> Vec3 {

    }

    fn position_to_screenspace(&self, position: Vec3) -> (f64, f64) {
    }
}