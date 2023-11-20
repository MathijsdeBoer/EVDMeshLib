use std::f64::consts::PI;

use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Copy)]
struct Vec3 {
    #[pyo3(get, set)]
    x: f64,

    #[pyo3(get, set)]
    y: f64,

    #[pyo3(get, set)]
    z: f64,
}

#[pymethods]
impl Vec3 {
    #[new]
    #[pyo3(signature = (x=0.0, y=0.0, z=0.0))]
    pub fn init(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }


}