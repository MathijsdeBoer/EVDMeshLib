use std::f64::consts::PI;
use std::fmt::{Debug, Display, Formatter};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Copy)]
pub struct Vec3 {
    #[pyo3(get, set)]
    pub x: f64,

    #[pyo3(get, set)]
    pub y: f64,

    #[pyo3(get, set)]
    pub z: f64,
}

#[pymethods]
impl Vec3 {
    #[new]
    #[pyo3(signature = (x=0.0, y=0.0, z=0.0))]
    pub fn init(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    #[staticmethod]
    pub fn zero() -> Self {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    #[staticmethod]
    pub fn one() -> Self {
        Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (v, wrap_around = false))]
    pub fn cartesian_to_spherical(v: &Self, wrap_around: bool) -> Self {
        let rho = v.length();
        let mut theta = v.y.atan2(v.x);
        let phi = (v.z / rho).acos();

        if wrap_around && theta < 0.0 {
            theta += 2.0 * PI;
        }

        Self {
            x: rho,
            y: theta,
            z: phi,
        }
    }

    #[staticmethod]
    pub fn spherical_to_cartesian(vec: Self) -> Self {
        let rho = vec.rho();
        let phi = vec.phi();
        let theta = vec.theta();
        Vec3 {
            x: rho * phi.sin() * theta.cos(),
            y: rho * phi.sin() * theta.sin(),
            z: rho * phi.cos(),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (v, width = 1, height = 1))]
    pub fn spherical_to_image(v: &Self, width: usize, height: usize) -> (f64, f64) {
        let (theta, phi) = (v.y, v.z);
        let x = (theta / (2.0 * PI)) * width as f64;
        let y = (phi / PI) * height as f64;
        (x, y)
    }

    #[staticmethod]
    #[pyo3(signature = (x, y, width = 1, height = 1))]
    pub fn image_to_spherical(x: f64, y: f64, width: usize, height: usize) -> Self {
        let theta = (x / width as f64) * 2.0 * PI;
        let phi = (y / height as f64) * PI;
        Self {
            x: 1.0,
            y: theta,
            z: phi,
        }
    }

    #[staticmethod]
    pub fn cartesian_to_cylindrical(vec: Self) -> Self {
        let rho = vec.rho();
        let phi = vec.phi();
        let z = vec.z;
        Vec3 { x: rho, y: phi, z }
    }

    #[staticmethod]
    pub fn cylindrical_to_cartesian(vec: Self) -> Self {
        let rho = vec.rho();
        let phi = vec.phi();
        let z = vec.z;
        Vec3 {
            x: rho * phi.cos(),
            y: rho * phi.sin(),
            z,
        }
    }

    #[getter]
    pub fn squared_length(&self) -> f64 {
        self.dot(self)
    }

    #[getter]
    pub fn length(&self) -> f64 {
        self.squared_length().sqrt()
    }

    #[getter]
    pub fn unit_vector(&self) -> Self {
        let k = 1.0 / self.length();
        Vec3 {
            x: self.x * k,
            y: self.y * k,
            z: self.z * k,
        }
    }

    #[getter]
    pub fn phi(&self) -> f64 {
        self.y.atan2(self.x)
    }

    #[getter]
    pub fn theta(&self) -> f64 {
        (self.z / self.length()).acos()
    }

    #[getter]
    pub fn rho(&self) -> f64 {
        self.length()
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn rotate_around(&self, axis: &Self, theta: f64) -> Self {
        let norm_self = self.unit_vector();
        let norm_axis = axis.unit_vector();

        let direction_parallel = norm_self.dot(&norm_axis) * norm_axis;
        let direction_perpendicular = norm_self - direction_parallel;

        direction_parallel * theta.cos()
            + norm_axis.cross(&direction_perpendicular) * theta.sin()
            + direction_perpendicular * theta.cos()
    }

    pub fn as_float_list(&self) -> Vec<f64> {
        vec![self.x, self.y, self.z]
    }

    pub fn __add__(&self, other: &Self) -> Self {
        *self + *other
    }

    pub fn __iadd__(&mut self, other: &Self) {
        *self += *other;
    }

    pub fn __sub__(&self, other: &Self) -> Self {
        *self - *other
    }

    pub fn __isub__(&mut self, other: &Self) {
        *self -= *other;
    }

    pub fn __mul__(&self, factor: f64) -> Self {
        *self * factor
    }

    pub fn __imul__(&mut self, factor: f64) {
        *self *= factor;
    }

    pub fn __truediv__(&self, factor: f64) -> Self {
        *self / factor
    }

    pub fn __itruediv__(&mut self, factor: f64) {
        *self /= factor;
    }

    pub fn __neg__(&self) -> Self {
        -*self
    }

    pub fn __eq__(&self, other: &Self) -> bool {
        *self == *other
    }

    pub fn __ne__(&self, other: &Self) -> bool {
        *self != *other
    }

    pub fn __str__(&self) -> String {
        format!("Vec3({}, {}, {})", self.x, self.y, self.z)
    }

    pub fn __repr__(&self) -> String {
        format!("Vec3({}, {}, {})", self.x, self.y, self.z)
    }
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        };
    }
}

impl Sum for Vec3 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        };
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, factor: f64) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
            z: self.z * factor,
        }
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, vec: Vec3) -> Vec3 {
        Vec3 {
            x: vec.x * self,
            y: vec.y * self,
            z: vec.z * self,
        }
    }
}

impl MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, factor: f64) {
        *self = Self {
            x: self.x * factor,
            y: self.y * factor,
            z: self.z * factor,
        };
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;

    fn div(self, factor: f64) -> Self {
        let f = 1.0 / factor;
        Self {
            x: self.x * f,
            y: self.y * f,
            z: self.z * f,
        }
    }
}

impl DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, factor: f64) {
        *self = Self {
            x: self.x / factor,
            y: self.y / factor,
            z: self.z / factor,
        };
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl PartialEq for Vec3 {
    fn eq(&self, other: &Self) -> bool {
        (self.x - other.x).abs() < f64::EPSILON
            && (self.y - other.y).abs() < f64::EPSILON
            && (self.z - other.z).abs() < f64::EPSILON
    }
}

impl Eq for Vec3 {}

impl Debug for Vec3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vec3({}, {}, {})", self.x, self.y, self.z)
    }
}

impl Display for Vec3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vec3({:.5}, {:.5}, {:.5})", self.x, self.y, self.z)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_add() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.0, 6.0);
        let v3 = v1.__add__(&v2);
        assert_eq!(v3, Vec3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_sub() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.0, 6.0);
        let v3 = v1.__sub__(&v2);
        assert_eq!(v3, Vec3::new(-3.0, -3.0, -3.0));
    }

    #[test]
    fn test_mul() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = v1.__mul__(2.0);
        assert_eq!(v2, Vec3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_div() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = v1.__truediv__(2.0);
        assert_eq!(v2, Vec3::new(0.5, 1.0, 1.5));
    }

    #[test]
    fn test_neg() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = v1.__neg__();
        assert_eq!(v2, Vec3::new(-1.0, -2.0, -3.0));
    }

    #[test]
    fn test_eq() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(1.0, 2.0, 3.0);
        let v3 = Vec3::new(1.0, 2.0, 4.0);
        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_unit_vector() {
        let v1 = Vec3::new(2.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 2.0, 0.0);
        let v3 = Vec3::new(0.0, 0.0, 2.0);
        assert_eq!(v1.unit_vector(), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(v2.unit_vector(), Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(v3.unit_vector(), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_dot() {
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);
        let v3 = Vec3::new(0.0, 0.0, 1.0);
        let v4 = Vec3::new(-1.0, 0.0, 0.0);
        assert_eq!(v1.dot(&v2), 0.0);
        assert_eq!(v1.dot(&v3), 0.0);
        assert_eq!(v1.dot(&v1), 1.0);
        assert_eq!(v1.dot(&v4), -1.0);
    }

    #[test]
    fn test_cross() {
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(v1.cross(&v2), Vec3::new(0.0, 0.0, 1.0));
    }
}
