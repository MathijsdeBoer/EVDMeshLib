use crate::linalg::Vec3;
use crate::rendering::projector::Projector;
use crate::rendering::Ray;

pub struct OrthographicCamera {
    position: Vec3,
    forward: Vec3,
    up: Vec3,
    right: Vec3,
    width: f64,
    height: f64
}


impl OrthographicCamera {
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
}

impl Projector for OrthographicCamera {
    fn project_to_2d(&self, x: f64, y: f64) -> Ray {
        let x = (x - 0.5) * self.width;
        let y = (y - 0.5) * self.height;

        let position = self.position + self.right * x + self.up * y;
        Ray::new(position, self.forward)
    }

    fn project_from_world(&self, position: Vec3) -> (f64, f64) {

    }
}