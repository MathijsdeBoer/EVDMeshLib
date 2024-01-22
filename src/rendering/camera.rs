use crate::linalg::Vec3;
use crate::rendering::Ray;
use pyo3::{pyclass, pymethods};
use std::f64::consts::PI;

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
    #[pyo3(get, set)]
    pub theta_offset: f64,
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
        theta_offset: Option<f64>,
    ) -> Self {
        match camera_type {
            CameraType::Perspective => {
                let fov = fov.unwrap_or(90.0).to_radians();
                let aspect_ratio = aspect_ratio.unwrap_or(1.0);
                let right = forward.cross(&up).unit_vector();
                let up = right.cross(&forward).unit_vector();
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
                    theta_offset: 0.0,
                }
            }
            CameraType::Orthographic => {
                let size = size.unwrap_or(1.0);
                let aspect_ratio = aspect_ratio.unwrap_or(1.0);
                let right = forward.cross(&up).unit_vector();
                let up = right.cross(&forward).unit_vector();
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
                    theta_offset: 0.0,
                }
            }
            CameraType::Equirectangular => {
                let aspect_ratio = aspect_ratio.unwrap_or(1.0);
                let right = forward.cross(&up).unit_vector();
                let up = right.cross(&forward).unit_vector();
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
                    size: 0.0,
                    theta_offset: theta_offset.unwrap_or(0.0),
                }
            }
        }
    }

    pub fn cast_ray(&self, x: usize, y: usize) -> Ray {
        let x = x as f64 / self.x_resolution as f64;
        let y = y as f64 / self.y_resolution as f64;
        
        match self.camera_type {
            CameraType::Perspective => {
                let x = (x - 0.5) * self.aspect_ratio * self.fov.tan();
                let y = (y - 0.5) * self.fov.tan();
                let direction = self.forward + self.right * x + self.up * y;

                Ray::new(self.origin, direction.unit_vector())
            }
            CameraType::Orthographic => {
                let x = (x - 0.5) * self.size * self.aspect_ratio;
                let y = (y - 0.5) * self.size;
                let origin = self.origin + self.right * x + self.up * y;

                Ray::new(origin, self.forward.unit_vector())
            }
            CameraType::Equirectangular => {
                // Spherical sampling adapted from
                // https://www.pbr-book.org/3ed-2018/Camera_Models/Environment_Camera
                let phi: f64 = PI * y;
                let theta: f64 = 2.0 * PI * x + self.theta_offset;

                let direction = Vec3::spherical_to_cartesian(Vec3::new(1.0, theta, phi));
                Ray::new(self.origin, direction.unit_vector())
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
