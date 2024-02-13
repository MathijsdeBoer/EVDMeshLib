use crate::linalg::Vec3;
use pyo3::prelude::*;
use std::ops::{Add, AddAssign, Index, Mul, Sub, SubAssign};

#[pyclass]
#[derive(Clone, Copy)]
pub struct Mat4 {
    #[pyo3(get)]
    pub data: [f64; 16],
}

#[pymethods]
impl Mat4 {
    #[new]
    pub fn new(data: [f64; 16]) -> Self {
        Mat4 { data }
    }

    #[staticmethod]
    #[rustfmt::skip]
    pub fn identity() -> Self {
        Mat4 {
            data: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    #[staticmethod]
    #[rustfmt::skip]
    pub fn translation(x: f64, y: f64, z: f64) -> Self {
        Mat4 {
            data: [
                1.0, 0.0, 0.0, x,
                0.0, 1.0, 0.0, y,
                0.0, 0.0, 1.0, z,
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    #[staticmethod]
    #[rustfmt::skip]
    pub fn scale(x: f64, y: f64, z: f64) -> Self {
        Mat4 {
            data: [
                x, 0.0, 0.0, 0.0,
                0.0, y, 0.0, 0.0,
                0.0, 0.0, z, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    #[staticmethod]
    #[rustfmt::skip]
    pub fn rotation(axis: Vec3, angle: f64) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        let one_sub_theta = 1.0 - cos;
        let (x, y, z) = (axis.x, axis.y, axis.z);
        Mat4 {
            data: [
                one_sub_theta * x * x + cos,
                one_sub_theta * x * y - sin * z,
                one_sub_theta * x * z + sin * y,
                0.0,

                one_sub_theta * x * y + sin * z,
                one_sub_theta * y * y + cos,
                one_sub_theta * y * z - sin * x,
                0.0,

                one_sub_theta * x * z - sin * y,
                one_sub_theta * y * z + sin * x,
                one_sub_theta * z * z + cos,
                0.0,

                0.0,
                0.0,
                0.0,
                1.0,
            ],
        }
    }

    pub fn transpose(&self) -> Mat4 {
        let mut data = [0.0; 16];
        for i in 0..4 {
            for j in 0..4 {
                data[i * 4 + j] = self.data[j * 4 + i];
            }
        }
        Mat4 { data }
    }

    pub fn __len__(&self) -> usize {
        16
    }

    pub fn __getitem__(&self, index: (usize, usize)) -> f64 {
        let idx = index.0 * 4 + index.1;
        if idx < 16 {
            self.data[idx]
        } else {
            panic!("Index out of range")
        }
    }

    pub fn __setitem__(&mut self, index: (usize, usize), value: f64) {
        let idx = index.0 * 4 + index.1;
        if idx < 16 {
            self.data[idx] = value;
        } else {
            panic!("Index out of range")
        }
    }
}

impl Add<Mat4> for Mat4 {
    type Output = Mat4;

    fn add(self, other: Mat4) -> Mat4 {
        let mut data = [0.0; 16];
        for (i, (a, b)) in self.data.iter().zip(other.data.iter()).enumerate() {
            data[i] = a + b;
        }
        Mat4 { data }
    }
}

impl AddAssign<Mat4> for Mat4 {
    fn add_assign(&mut self, other: Mat4) {
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
    }
}

impl Sub<Mat4> for Mat4 {
    type Output = Mat4;

    fn sub(self, other: Mat4) -> Mat4 {
        let mut data = [0.0; 16];
        for (i, (a, b)) in self.data.iter().zip(other.data.iter()).enumerate() {
            data[i] = a - b;
        }
        Mat4 { data }
    }
}

impl SubAssign<Mat4> for Mat4 {
    fn sub_assign(&mut self, other: Mat4) {
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a -= b;
        }
    }
}

impl Mul<Mat4> for Mat4 {
    type Output = Mat4;

    fn mul(self, other: Mat4) -> Mat4 {
        let mut data = [0.0; 16];
        for i in 0..4 {
            for j in 0..4 {
                data[i * 4 + j] = self[(i, 0)] * other[(0, j)]
                    + self[(i, 1)] * other[(1, j)]
                    + self[(i, 2)] * other[(2, j)]
                    + self[(i, 3)] * other[(3, j)];
            }
        }
        Mat4 { data }
    }
}

impl Mul<f64> for Mat4 {
    type Output = Mat4;

    fn mul(self, scalar: f64) -> Mat4 {
        let mut data = [0.0; 16];
        for (i, a) in self.data.iter().enumerate() {
            data[i] = a * scalar;
        }
        Mat4 { data }
    }
}

impl Mul<Mat4> for f64 {
    type Output = Mat4;

    fn mul(self, mat: Mat4) -> Mat4 {
        mat * self
    }
}

impl Mul<Vec3> for Mat4 {
    type Output = Vec3;

    fn mul(self, vec: Vec3) -> Vec3 {
        let x = self[(0, 0)] * vec.x + self[(0, 1)] * vec.y + self[(0, 2)] * vec.z + self[(0, 3)];
        let y = self[(1, 0)] * vec.x + self[(1, 1)] * vec.y + self[(1, 2)] * vec.z + self[(1, 3)];
        let z = self[(2, 0)] * vec.x + self[(2, 1)] * vec.y + self[(2, 2)] * vec.z + self[(2, 3)];
        Vec3 { x, y, z }
    }
}

impl Index<(usize, usize)> for Mat4 {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &f64 {
        &self.data[row * 4 + col]
    }
}

impl Index<usize> for Mat4 {
    type Output = f64;

    fn index(&self, index: usize) -> &f64 {
        &self.data[index]
    }
}
