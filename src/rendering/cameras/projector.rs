use crate::linalg::Vec3;
use crate::rendering::Ray;

pub trait Projector {
    fn ray_from_screenspace(&self, x: f64, y: f64) -> Ray;
    fn position_to_screenspace(&self, position: Vec3) -> (f64, f64);
} 