use crate::linalg::Vec3;
use crate::rendering::projector::Projector;

struct ProjectionCamera {
    position: Vec3,
    forward: Vec3,
    up: Vec3,
    right: Vec3,
    fov: f64,
    aspect_ratio: f64
}

impl ProjectionCamera {
    pub fn new(position: Vec3, forward: Vec3, up: Vec3, fov: f64, aspect_ratio: f64) -> Self {
        let right = forward.cross(&up).normalize();
        let up = right.cross(forward).normalize();

        Self {
            position,
            forward,
            up,
            right,
            fov,
            aspect_ratio
        }
    }
}

impl Projector for ProjectionCamera {
    fn project_to_2d(&self, x: f64, y: f64) -> Ray {
        let x = (x - 0.5) * self.aspect_ratio * self.fov;
        let y = (y - 0.5) * self.fov;

        let direction = self.forward + self.right * x + self.up * y;
        Ray::new(self.position, direction.normalize())
    }

    fn project_from_world(&self, position: Vec3) -> (f64, f64) {
        let direction = position - self.position;
        let x = direction.dot(&self.right) / self.fov / self.aspect_ratio + 0.5;
        let y = direction.dot(&self.up) / self.fov + 0.5;
        (x, y)
    }
}
