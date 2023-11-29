use pyo3::prelude::*;

use crate::linalg::Vec3;
use crate::rendering::cameras::BaseCamera;
use crate::rendering::cameras::projector::Projector;
use crate::rendering::Ray;

#[pyclass(name = "OrthographicCamera", extends = BaseCamera)]
pub struct OrthographicCamera {
    position: Vec3,
    forward: Vec3,
    up: Vec3,
    right: Vec3,
    width: f64,
    height: f64
}

#[pymethods]
impl OrthographicCamera {
    #[new]
    pub fn new(position: Vec3, forward: Vec3, up: Vec3, width: f64, height: f64) -> Self {
        let right = forward.cross(&up).normalize();
        let up = right.cross(forward).normalize();

        Self {
            position,
            forward,
            up,
            right,
            width,
            height
        }
    }

    fn ray_from_screenspace(&self, x: f64, y: f64) -> Ray {
        let x = (x - 0.5) * self.width;
        let y = (y - 0.5) * self.height;

        let position = self.position + self.right * x + self.up * y;
        Ray::new(position, self.forward)
    }

    fn position_to_screenspace(&self, position: Vec3) -> (f64, f64) {
        let back_to_camera = -self.forward;

        // Intersect with camera plane
        let t = (self.position - position).dot(&back_to_camera) / back_to_camera.dot(&back_to_camera);
        let intersection = position + back_to_camera * t;

        // Project to 2D
        let x = intersection.dot(&self.right) / self.width + 0.5;
        let y = intersection.dot(&self.up) / self.height + 0.5;
    }
}

impl Projector for OrthographicCamera {
    fn ray_from_screenspace(&self, x: f64, y: f64) -> Ray {
        OrthographicCamera::ray_from_screenspace(self, x, y)
    }

    fn position_to_screenspace(&self, position: Vec3) -> (f64, f64) {
        OrthographicCamera::position_to_screenspace(self, position)
    }
}