use pyo3::prelude::*;

mod linalg;

#[pymodule]
fn _rs(_py: Python, m: &PyModule) -> PyResult<()> {
    linalg::init_linalg(_py, m)?;

    Ok(())
}
