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
    pub left: Vec3,
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
        let left = -forward.cross(&up).unit_vector();
        let up = forward.cross(&left).unit_vector();

        match camera_type {
            CameraType::Perspective => {
                let fov = fov.unwrap_or(90.0).to_radians();
                let aspect_ratio = aspect_ratio.unwrap_or(1.0);

                Self {
                    origin,
                    forward,
                    left,
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
                    left,
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
                    left,
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
                let origin = self.origin;
                let forward = self.forward;
                let left = self.left;
                let up = self.up;

                // Imaginary plane at distance 1 from the camera
                let plane_distance = 1.0;
                let plane_width = 2.0 * (self.fov / 2.0).tan();
                let plane_height = plane_width / self.aspect_ratio;

                // find the position of the pixel on the plane
                let x = (x as f64 / self.x_resolution as f64 - 0.5) * plane_width;
                let y = (y as f64 / self.y_resolution as f64 - 0.5) * plane_height;
                let plane_position = left * x + up * y + forward * plane_distance;

                Ray::new(
                    origin,
                    (plane_position - origin).unit_vector(),
                )
            }
            CameraType::Orthographic => {
                let forward = self.forward;
                let left = self.left;
                let up = self.up;

                let x = (x as f64 / self.x_resolution as f64 - 0.5) * self.size * self.aspect_ratio;
                let y = (y as f64 / self.y_resolution as f64 - 0.5) * self.size;

                Ray::new(
                    self.origin + left * x + up * y,
                    forward,
                )
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
                let world_dir = self.left * camera_dir.x
                    + self.forward * camera_dir.y
                    + self.up * camera_dir.z;

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
                let back_direction = -self.forward;

                // Ray-plane intersection
                let t = (*point - self.origin).dot(&self.forward) / back_direction.dot(&self.forward);
                let intersection = *point + back_direction * t;

                // Find the position of the pixel on the plane,
                // where the origin is (0.5, 0.5)
                let x = (intersection.dot(&self.left) / self.size + 0.5) * self.x_resolution as f64;
                let y = (intersection.dot(&self.up) / self.size + 0.5) * self.y_resolution as f64;

                if normalized {
                    (x / self.x_resolution as f64, y / self.y_resolution as f64)
                } else {
                    (x, y)
                }
            }
            CameraType::Equirectangular => {
                let direction_world = (*point - self.origin).unit_vector();
                let direction_camera = Vec3 {
                    x: direction_world.dot(&self.left),
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
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::linalg::Vec3;

    #[test]
    fn test_equirectangular_cast_ray() {
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let forward = Vec3::new(0.0, -1.0, 0.0);
        let up = Vec3::new(0.0, 0.0, 1.0);

        let resolution = 512;
        let half_res = resolution / 2;
        let one_quarter_res = resolution / 4;
        let three_quarters_res = one_quarter_res * 3;

        let camera = Camera::new(
            origin,
            forward,
            up,
            resolution,
            half_res,
            CameraType::Equirectangular,
            None,
            None,
            None,
        );

        let directions = vec![
            (half_res, one_quarter_res, camera.forward),
            (0, one_quarter_res, -camera.forward),
            (three_quarters_res, one_quarter_res, -camera.left),
            (one_quarter_res, one_quarter_res, camera.left),
            (0, 0, camera.up),
            (resolution, 0, camera.up),
            (0, half_res, -camera.up),
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
    fn test_equirectangular_cast_ray2() {
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let forward = Vec3::new(1.0, 0.0, 0.0);
        let up = Vec3::new(0.0, 0.0, 1.0);

        let resolution = 512;
        let half_res = resolution / 2;
        let one_quarter_res = resolution / 4;
        let three_quarters_res = one_quarter_res * 3;

        let camera = Camera::new(
            origin,
            forward,
            up,
            resolution,
            half_res,
            CameraType::Equirectangular,
            None,
            None,
            None,
        );

        let directions = vec![
            (half_res, one_quarter_res, camera.forward),
            (0, one_quarter_res, -camera.forward),
            (three_quarters_res, one_quarter_res, -camera.left),
            (one_quarter_res, one_quarter_res, camera.left),
            (0, 0, camera.up),
            (resolution, 0, camera.up),
            (0, half_res, -camera.up),
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
            (Vec3::new(0.0, 0.0, 1.0), (0.25, 0.0)),
            (Vec3::new(0.0, 0.0, -1.0), (0.25, 1.0)),

            (Vec3::new(0.0, 1.0, 0.0), (0.0, 0.5)),
            (Vec3::new(0.0, -1.0, 0.0), (0.5, 0.5)),

            (Vec3::new(1.0, 0.0, 0.0), (0.25, 0.5)),
            (Vec3::new(-1.0, 0.0, 0.0), (0.75, 0.5)),

            (Vec3::new(0.0, -1.0, 1.0), (0.5, 0.25)),
            (Vec3::new(0.0, 1.0, 1.0), (0.0, 0.25)),
            (Vec3::new(1.0, 0.0, 1.0), (0.25, 0.25)),
            (Vec3::new(-1.0, 0.0, 1.0), (0.75, 0.25)),

            (Vec3::new(0.0, -1.0, -1.0), (0.5, 0.75)),
            (Vec3::new(0.0, 1.0, -1.0), (0.0, 0.75)),
            (Vec3::new(1.0, 0.0, -1.0), (0.25, 0.75)),
            (Vec3::new(-1.0, 0.0, -1.0), (0.75, 0.75)),
        ];

        for (point, (x, y)) in points {
            assert_eq!(
                camera.project_back(&point, true), (x, y),
                "Testing point {:?}", point
            );
        }
    }
}
