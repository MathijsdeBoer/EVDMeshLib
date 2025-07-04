mod deformer;
mod mesh;

pub use deformer::Deformer;
pub use mesh::{Mesh, Triangle};

use pyo3::prelude::*;

#[pymodule]
fn geometry(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Deformer>()?;
    m.add_class::<Mesh>()?;
    m.add_class::<Triangle>()?;
    Ok(())
}

pub fn init_geometry(_py: Python, m: &PyModule) -> PyResult<()> {
    let geometry_child_module = PyModule::new(_py, "evdplanner.rs.geometry")?;
    geometry(_py, geometry_child_module)?;

    m.add("geometry", geometry_child_module)?;

    _py.import("sys")?
        .getattr("modules")?
        .set_item("evdplanner.rs.geometry", geometry_child_module)?;

    Ok(())
}
