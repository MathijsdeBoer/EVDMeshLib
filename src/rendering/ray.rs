use pyo3::prelude::*;

use crate::linalg::Vec3;

#[pyclass]
pub struct Ray {
    #[pyo3(get)]
    pub origin: Vec3,
    #[pyo3(get)]
    pub direction: Vec3,
}

#[pymethods]
impl Ray {
    #[new]
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    pub fn at(&self, t: f64) -> Vec3 {
        self.origin + t * self.direction
    }

    pub fn __repr__(&self) -> String {
        format!("Ray(origin={}, direction={})", self.origin, self.direction)
    }

    pub fn __str__(&self) -> String {
        format!("Ray(origin={}, direction={})", self.origin, self.direction)
    }
}
