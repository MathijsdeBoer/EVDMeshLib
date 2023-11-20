mod vec3;

pub use vec3::Vec3;

use pyo3::prelude::*;

pub fn init_linalg(_py: Python, m: &PyModule) -> PyResult<()> {
    let linalg_submodule = PyModule::new(_py, "linalg")?;
    linalg_submodule.add_class::<Vec3>()?;
    m.add_submodule(linalg_submodule)?;

    Ok(())
}