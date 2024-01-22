use pyo3::prelude::*;

mod geometry;
mod linalg;
mod rendering;

#[pymodule]
fn rs(_py: Python, m: &PyModule) -> PyResult<()> {
    geometry::init_geometry(_py, m)?;
    linalg::init_linalg(_py, m)?;
    rendering::init_rendering(_py, m)?;

    m.add_class::<geometry::Mesh>()?;
    m.add_class::<linalg::Vec3>()?;
    m.add_class::<rendering::Camera>()?;
    m.add_class::<rendering::CameraType>()?;
    m.add_class::<rendering::Intersection>()?;
    m.add_class::<rendering::IntersectionSort>()?;
    m.add_class::<rendering::Ray>()?;
    m.add_class::<rendering::Renderer>()?;

    Ok(())
}
