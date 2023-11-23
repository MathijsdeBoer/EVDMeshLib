use pyo3::prelude::*;

use crate::rendering::{Intersection, Ray};

#[pyclass(name = "Intersectable", subclass)]
pub struct IntersectableObject {}

#[pymethods]
impl IntersectableObject {
    #[new]
    pub fn new() -> Self {
        Self {}
    }

    fn intersect(&self, _ray: &Ray) -> Option<Intersection> {
        None
    }
}

impl Default for  IntersectableObject {
    fn default() -> Self {
        Self::new()
    }
}

pub trait Intersectable {
    fn intersect(&self, ray: &Ray) -> Option<Intersection>;
}

impl Intersectable for IntersectableObject {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        IntersectableObject::intersect(self, ray)
    }
}
