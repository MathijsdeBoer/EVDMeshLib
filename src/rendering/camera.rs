use crate::linalg::Vec3;
use crate::rendering::Ray;
use pyo3::{pyclass, pymethods};
use std::f64::consts::PI;

const TWO_PI: f64 = 2.0 * PI;
const HALF_PI: f64 = 0.5 * PI;

#[pyclass]
#[derive(Clone, Copy)]
pub enum CameraType {
    Perspective,
    Orthographic,
    Equirectangular,
}

#[pyclass]
#[derive(Clone, Copy)]
pub struct Camera {
    #[pyo3(get, set)]
    pub origin: Vec3,
    #[pyo3(get, set)]
    pub forward: Vec3,
    #[pyo3(get, set)]
    pub right: Vec3,
    #[pyo3(get, set)]
    pub up: Vec3,

    #[pyo3(get, set)]
    pub x_resolution: usize,
    #[pyo3(get, set)]
    pub y_resolution: usize,

    #[pyo3(get, set)]
    pub camera_type: CameraType,

    #[pyo3(get, set)]
    pub fov: f64,
    #[pyo3(get, set)]
    pub aspect_ratio: f64,
    #[pyo3(get, set)]
    pub size: f64,
}

#[pymethods]
impl Camera {
    #[new]
    pub fn new(
        origin: Vec3,
        forward: Vec3,
        up: Vec3,
        x_resolution: usize,
        y_resolution: usize,
        camera_type: CameraType,
        fov: Option<f64>,
        aspect_ratio: Option<f64>,
        size: Option<f64>,
    ) -> Self {
        let right = up.cross(&forward).unit_vector();
        let up = right.cross(&forward).unit_vector();

        match camera_type {
            CameraType::Perspective => {
                let fov = fov.unwrap_or(90.0).to_radians();
                let aspect_ratio = aspect_ratio.unwrap_or(1.0);

                Self {
                    origin,
                    forward,
                    right,
                    up,
                    x_resolution,
                    y_resolution,
                    camera_type,
                    fov,
                    aspect_ratio,
                    size: 0.0,
                }
            }
            CameraType::Orthographic => {
                let size = size.unwrap_or(1.0);
                let aspect_ratio = aspect_ratio.unwrap_or(1.0);

                Self {
                    origin,
                    forward,
                    right,
                    up,
                    x_resolution,
                    y_resolution,
                    camera_type,
                    fov: 0.0,
                    aspect_ratio,
                    size,
                }
            }
            CameraType::Equirectangular => {
                Self {
                    origin,
                    forward,
                    right,
                    up,
                    x_resolution,
                    y_resolution,
                    camera_type,
                    fov: 0.0,
                    aspect_ratio: 0.0,
                    size: 0.0,
                }
            }
        }
    }

    pub fn cast_ray(&self, x: usize, y: usize) -> Ray {
        match self.camera_type {
            CameraType::Perspective => {
                todo!("Perspective camera not yet implemented")
            }
            CameraType::Orthographic => {
                todo!("Orthographic camera not yet implemented")
            }
            CameraType::Equirectangular => {
                // Spherical sampling adapted from
                // https://www.pbr-book.org/3ed-2018/Camera_Models/Environment_Camera

                // Phi is pretty straightforward
                let phi: f64 = PI * (y as f64 / self.y_resolution as f64);

                // We want to ensure that the middle of the image (0.5, 0.5) maps to the forward
                // direction, so we need to subtract 0.5 * PI from theta.
                let theta: f64 = TWO_PI * (x as f64 / self.x_resolution as f64) - HALF_PI;

                // Direction in spherical coordinates
                let spherical = Vec3::new(1.0, theta, phi);
                // Direction in camera space
                let camera_dir = Vec3::spherical_to_cartesian(&spherical).unit_vector();
                // Direction in world space, based on camera orientation
                let world_dir = self.right * camera_dir.x
                    + self.forward * camera_dir.y
                    + self.up * -camera_dir.z;

                Ray::new(self.origin, world_dir)
            }
        }
    }

    #[pyo3(signature = (point, normalized = false))]
    pub fn project_back(&self, point: &Vec3, normalized: bool) -> (f64, f64) {
        match self.camera_type {
            CameraType::Perspective => {
                todo!("Perspective camera not yet implemented")
            }
            CameraType::Orthographic => {
                todo!("Orthographic camera not yet implemented")
            }
            CameraType::Equirectangular => {
                let direction_world = (*point - self.origin).unit_vector();
                let direction_camera = Vec3 {
                    x: direction_world.dot(&self.right),
                    y: direction_world.dot(&self.forward),
                    z: direction_world.dot(&self.up),
                };
                let spherical = Vec3::cartesian_to_spherical(&direction_camera, true);

                let mut x = (spherical.y + HALF_PI) / TWO_PI;
                let y = spherical.z / PI;

                while !(0.0..1.0).contains(&x) {
                    if x < 0.0 {
                        x += 1.0;
                    } else {
                        x -= 1.0;
                    }
                }

                if normalized {
                    (x, y)
                } else {
                    (x * self.x_resolution as f64, y * self.y_resolution as f64)
                }
            }
        }
    }

    pub fn translate(&mut self, vec: &Vec3) {
        self.origin += self.right * vec.x + self.up * vec.y + self.forward * vec.z;
    }

    pub fn rotate(&mut self, axis: &Vec3, angle: f64) {
        let original_origin = self.origin;
        self.translate(&-original_origin);
        self.forward = self.forward.rotate_around(axis, angle);
        self.right = self.right.rotate_around(axis, angle);
        self.up = self.up.rotate_around(axis, angle);
        self.translate(&original_origin);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::linalg::Vec3;
    use crate::rendering::Ray;

    #[test]
    fn test_equirectangular_camera_cast_ray() {
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let forward = Vec3::new(0.0, -1.0, 0.0);
        let up = Vec3::new(0.0, 0.0, 1.0);

        let camera = Camera::new(
            origin,
            forward,
            up,
            100,
            100,
            CameraType::Equirectangular,
            None,
            None,
            None,
        );

        let directions = vec![
            (50, 50, forward),
            (0, 50, -forward),
            (0, 0, up),
            (99, 0, up),
            (0, 99, -up),
            (75, 50, camera.right),
            (25, 50, -camera.right),
        ];

        for (x, y, expected_direction) in directions {
            let ray = camera.cast_ray(x, y);
            assert_eq!(
                ray.direction, expected_direction,
                "Testing pixel ({}, {})", x, y
            );
        }
    }

    #[test]
    fn test_equirectangular_camera_project_back() {
        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            100,
            100,
            CameraType::Equirectangular,
            None,
            None,
            None,
        );

        let points = vec![
            (Vec3::new(0.0, 0.0, 1.0), (0.25, 1.0)),
            (Vec3::new(0.0, 0.0, -1.0), (0.25, 0.0)),

            (Vec3::new(0.0, 1.0, 0.0), (0.0, 0.5)),
            (Vec3::new(0.0, -1.0, 0.0), (0.5, 0.5)),

            (Vec3::new(1.0, 0.0, 0.0), (0.25, 0.5)),
            (Vec3::new(-1.0, 0.0, 0.0), (0.75, 0.5)),

            (Vec3::new(0.0, -1.0, 1.0), (0.5, 0.75)),
            (Vec3::new(0.0, 1.0, 1.0), (0.0, 0.75)),
            (Vec3::new(1.0, 0.0, 1.0), (0.25, 0.75)),
            (Vec3::new(-1.0, 0.0, 1.0), (0.75, 0.75)),

            (Vec3::new(0.0, -1.0, -1.0), (0.5, 0.25)),
            (Vec3::new(0.0, 1.0, -1.0), (0.0, 0.25)),
            (Vec3::new(1.0, 0.0, -1.0), (0.25, 0.25)),
            (Vec3::new(-1.0, 0.0, -1.0), (0.75, 0.25)),
        ];

        for (point, (x, y)) in points {
            assert_eq!(
                camera.project_back(&point, true), (x, y),
                "Testing point {:?}", point
            );
        }
    }
}
