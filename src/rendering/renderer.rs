use pyo3::prelude::*;

use crate::rendering::Intersection;
use crate::rendering::cameras::projector::Projector;
use crate::rendering::scene::Scene;

#[pyclass]
struct Renderer {
    projector: BaseCamera,
    scene: Scene,
    
    x_resolution: usize,
    y_resolution: usize,
    
    #[pyo3(get)]
    pub image: Vec<Vec<Option<Intersection>>>,
}

#[pymethods]
impl Renderer {
    #[new]
    pub fn new(
        projector: impl Projector + 'static,
        scene: Scene,
        x_resolution: usize,
        y_resolution: usize,
    ) -> Self {
        Self {
            projector: Box::new(projector),
            scene,
            x_resolution,
            y_resolution,
            image: vec![vec![None; x_resolution]; y_resolution],
        }
    }
    
    pub fn render(&self) {
        // ...
    }
}