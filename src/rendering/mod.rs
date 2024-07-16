mod camera;
mod intersection;
mod ray;
mod renderer;
#[cfg(target_feature = "avx")]
pub mod simd;
mod target;

pub use camera::{Camera, CameraType};
pub use intersection::{Intersection, IntersectionSort};
pub use ray::Ray;
pub use renderer::Renderer;
pub use target::{find_target, generate_objective_image, objective_function};

use pyo3::prelude::*;

#[pymodule]
fn rendering(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Camera>()?;
    m.add_class::<CameraType>()?;
    m.add_class::<Intersection>()?;
    m.add_class::<IntersectionSort>()?;
    m.add_class::<Ray>()?;
    m.add_class::<Renderer>()?;

    m.add_function(wrap_pyfunction!(find_target, m)?)?;
    m.add_function(wrap_pyfunction!(objective_function, m)?)?;
    m.add_function(wrap_pyfunction!(generate_objective_image, m)?)?;
    Ok(())
}

pub fn init_rendering(_py: Python, m: &PyModule) -> PyResult<()> {
    let rendering_child_module = PyModule::new(_py, "evdplanner.rs.rendering")?;
    rendering(_py, rendering_child_module)?;

    m.add("rendering", rendering_child_module)?;

    _py.import("sys")?
        .getattr("modules")?
        .set_item("evdplanner.rs.rendering", rendering_child_module)?;

    Ok(())
}
