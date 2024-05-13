use itertools::iproduct;
use ndarray::{Array3, Ix3};
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::geometry::Mesh;
use crate::rendering::{Camera, Intersection, IntersectionSort};

#[pyclass]
pub struct Renderer {
    #[pyo3(get, set)]
    pub camera: Camera,

    #[pyo3(get, set)]
    pub mesh: Mesh,
}

#[pymethods]
impl Renderer {
    #[new]
    pub fn new(camera: Camera, mesh: Mesh) -> Self {
        Self { camera, mesh }
    }

    #[pyo3(signature = (intersection_mode, epsilon=1e-8))]
    pub fn render(
        &self,
        py: Python<'_>,
        intersection_mode: IntersectionSort,
        epsilon: f64,
    ) -> Py<PyArray<f64, Ix3>> {
        let mut image =
            Array3::<f64>::zeros((self.camera.y_resolution, self.camera.x_resolution, 4));

        let values: Vec<((usize, usize), Option<Intersection>)> =
            self.render_internal(intersection_mode, epsilon);

        for ((y, x), intersection) in values {
            if let Some(intersection) = intersection {
                image[[y, x, 0]] = intersection.distance;
                image[[y, x, 1]] = intersection.normal.x;
                image[[y, x, 2]] = intersection.normal.y;
                image[[y, x, 3]] = intersection.normal.z;
            }
        }

        image.into_pyarray(py).to_owned()
    }

    pub fn generate_intersections(
        &self,
        intersection_mode: IntersectionSort,
        epsilon: f64,
    ) -> Vec<((usize, usize), Intersection)> {
        self.render_internal(intersection_mode, epsilon)
            .into_iter()
            .filter_map(|((y, x), intersection)| {
                intersection.map(|intersection| ((y, x), intersection))
            })
            .collect()
    }
}

impl Renderer {
    pub fn render_internal(
        &self,
        intersection_mode: IntersectionSort,
        epsilon: f64,
    ) -> Vec<((usize, usize), Option<Intersection>)> {
        let image_pixels: Vec<(usize, usize)> =
            iproduct!(0..self.camera.y_resolution, 0..self.camera.x_resolution).collect();

        image_pixels
            .par_iter()
            .map(|(y, x)| {
                (
                    (*y, *x),
                    self.mesh.intersect(
                        &self.camera.cast_ray(*x as f64, *y as f64),
                        intersection_mode,
                        epsilon,
                    ),
                )
            })
            .collect()
    }
}
