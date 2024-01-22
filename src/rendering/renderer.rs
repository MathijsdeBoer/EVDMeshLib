use std::iter::zip;

use indicatif::{ProgressBar, ProgressStyle};
use itertools::iproduct;
use ndarray::{Array3, Ix3};
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::geometry::Mesh;
use crate::rendering::{Camera, Intersection, IntersectionSort};

#[pyclass(name = "CPURenderer", subclass)]
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
        Self {
            camera,
            mesh,
        }
    }

    pub fn render(
        &self,
        py: Python<'_>,
        intersection_mode: IntersectionSort,
    ) -> Py<PyArray<f64, Ix3>> {
        let mut image = Array3::<f64>::zeros((self.camera.y_resolution, self.camera.x_resolution, 4));

        let image_pixels: Vec<(usize, usize)> =
            iproduct!(0..self.camera.y_resolution, 0..self.camera.x_resolution).collect();
        // let bar = ProgressBar::new(image_pixels.len() as u64);
        // bar.set_style(
        //     ProgressStyle::with_template(
        //         "[{elapsed_precise} / {eta_precise}] {spinner} {wide_bar} {msg} {pos:>7}/{len:7}",
        //     )
        //     .unwrap(),
        // );
        // bar.set_message("Rendering");

        let values: Vec<Option<Intersection>> = image_pixels
            .iter()
            .map(|(y, x)| {
                // bar.inc(1);
                let ray = self.camera.cast_ray(
                    *x, *y
                );
                self.mesh.intersect(&ray, intersection_mode)
            })
            .collect();
        // bar.finish();

        for ((y, x), intersection) in zip(image_pixels, values) {
            if let Some(intersection) = intersection {
                image[[y, x, 0]] = intersection.distance;
                image[[y, x, 1]] = intersection.normal.x;
                image[[y, x, 2]] = intersection.normal.y;
                image[[y, x, 3]] = intersection.normal.z;
            }
        }

        image.into_pyarray(py).to_owned()
    }
}
