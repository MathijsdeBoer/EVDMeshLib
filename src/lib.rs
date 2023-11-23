use pyo3::prelude::*;

mod geometry;
mod linalg;
mod rendering;

#[pymodule]
fn _rs(_py: Python, m: &PyModule) -> PyResult<()> {
    linalg::init_linalg(_py, m)?;

    Ok(())
}
