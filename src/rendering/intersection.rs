use pyo3::prelude::*;

use crate::linalg::Vec3;

#[pyclass]
pub struct Intersection {
    #[pyo3(get)]
    pub distance: f64,
    #[pyo3(get)]
    pub position: Vec3,
    #[pyo3(get)]
    pub normal: Vec3,
}

#[pymethods]
impl Intersection {
    #[new]
    pub fn new(distance: f64, position: Vec3, normal: Vec3) -> Self {
        Self {
            distance,
            position,
            normal,
        }
    }
}
