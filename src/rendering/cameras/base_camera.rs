use pyo3::prelude::*;
use crate::linalg::Vec3;

use crate::rendering::{Ray};
use crate::rendering::cameras::projector::Projector;

#[pyclass(name = "BaseCamera", subclass)]
pub struct BaseCamera {}

#[pymethods]
impl BaseCamera {
    #[new]
    pub fn new() -> Self {
        Self {}
    }

    fn ray_from_screenspace(&self, _x: f64, _y: f64) -> Ray {
        panic!("BaseCamera::ray_from_screenspace() called")
    }

    fn position_to_screenspace(&self, _position: Vec3) -> (f64, f64) {
        panic!("BaseCamera::position_to_screenspace() called")
    }
}

impl Projector for BaseCamera {
    fn ray_from_screenspace(&self, x: f64, y: f64) -> Ray {
        BaseCamera::ray_from_screenspace(self, x, y)
    }

    fn position_to_screenspace(&self, position: Vec3) -> (f64, f64) {
        BaseCamera::position_to_screenspace(self, position)
    }
}

impl Default for BaseCamera {
    fn default() -> Self {
        Self::new()
    }
}