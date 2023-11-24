mod vec3;

pub use vec3::Vec3;

use pyo3::prelude::*;

#[pymodule]
fn linalg(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Vec3>()?;
    Ok(())
}

pub fn init_linalg(_py: Python, m: &PyModule) -> PyResult<()> {
    let linalg_child_module = PyModule::new(_py, "evdmeshlib.rs.linalg")?;
    linalg(_py, linalg_child_module)?;

    m.add("linalg", linalg_child_module)?;

    _py.import("sys")?
        .getattr("modules")?
        .set_item("evdmeshlib.rs.linalg", linalg_child_module)?;

    Ok(())
}
