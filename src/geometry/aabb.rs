use crate::geometry::intersectable::{Intersectable, IntersectableObject};
use crate::linalg::Vec3;
use crate::rendering::{Intersection, Ray};
use pyo3::prelude::*;

#[pyclass(extends=IntersectableObject)]
#[derive(Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

#[pymethods]
impl Aabb {
    #[new]
    pub fn new(min: Vec3, max: Vec3) -> PyClassInitializer<Self> {
        PyClassInitializer::from(IntersectableObject::new()).add_subclass(Self { min, max })
    }
}

impl Intersectable for Aabb {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        todo!()
    }
}
