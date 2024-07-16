mod mat4;
#[cfg(target_feature = "avx")]
pub mod simd;
mod vec3;

pub use mat4::Mat4;
pub use vec3::Vec3;

use pyo3::prelude::*;

#[pymodule]
fn linalg(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Vec3>()?;
    m.add_class::<Mat4>()?;
    Ok(())
}

pub fn init_linalg(_py: Python, m: &PyModule) -> PyResult<()> {
    let linalg_child_module = PyModule::new(_py, "evdplanner.rs.linalg")?;
    linalg(_py, linalg_child_module)?;

    m.add("linalg", linalg_child_module)?;

    _py.import("sys")?
        .getattr("modules")?
        .set_item("evdplanner.rs.linalg", linalg_child_module)?;

    Ok(())
}
