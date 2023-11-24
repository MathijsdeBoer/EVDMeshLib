use crate::linalg::Vec3;
use crate::rendering::Ray;

pub trait Projector {
    fn project_to_2d(&self, x: f64, y: f64) -> Ray;
    fn project_from_world(&self, position: Vec3) -> (f64, f64);
} 