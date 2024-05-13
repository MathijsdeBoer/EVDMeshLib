use std::fmt::{Debug, Display};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use wide::f64x4;

use crate::linalg::Vec3;

#[derive(Clone, Copy)]
pub struct Vec3x4 {
    pub x: f64x4,
    pub y: f64x4,
    pub z: f64x4,
}

impl Vec3x4 {
    pub fn new(x: f64x4, y: f64x4, z: f64x4) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        let zero = unsafe { _mm256_setzero_pd() };
        Self {
            x: zero,
            y: zero,
            z: zero,
        }
    }

    pub fn one() -> Self {
        let one = f64x4::ONE;
        Self {
            x: one,
            y: one,
            z: one,
        }
    }

    pub fn broadcast(v: Vec3) -> Self {
        let x = f64x4::splat(v.x);
        let y = f64x4::splat(v.y);
        let z = f64x4::splat(v.z);
        Self { x, y, z }
    }

    pub fn unpack(&self) -> (Vec3, Vec3, Vec3, Vec3) {
        let mut x = self.x.to_array();
        let mut y = self.y.to_array();
        let mut z = self.z.to_array();
        (
            Vec3::new(x[0], y[0], z[0]),
            Vec3::new(x[1], y[1], z[1]),
            Vec3::new(x[2], y[2], z[2]),
            Vec3::new(x[3], y[3], z[3]),
        )
    }

    pub fn pack(v1: Vec3, v2: Vec3, v3: Vec3, v4: Vec3) -> Self {
        let x = f64x4::new([v1.x, v2.x, v3.x, v4.x]);
        let y = f64x4::new([v1.y, v2.y, v3.y, v4.y]);
        let z = f64x4::new([v1.z, v2.z, v3.z, v4.z]);
        Self { x, y, z }
    }

    pub fn dot(&self, other: &Self) -> F64x4 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Self) -> Self {
        let x = self.y * other.z - self.z * other.y;
        let y = self.z * other.x - self.x * other.z;
        let z = self.x * other.y - self.y * other.x;
        Self { x, y, z }
    }

    pub fn squared_length(&self) -> F64x4 {
        self.dot(self)
    }

    pub fn length(&self) -> F64x4 {
        self.length_squared().sqrt()
    }

    pub fn unit_vector(&self) -> Self {
        let length = 1.0 / self.length();
        let x = self.x * length;
        let y = self.y * length;
        let z = self.z * length;
        Self { x, y, z }
    }
}

impl Add for Vec3x4 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let x = self.x + other.x;
        let y = self.y + other.y;
        let z = self.z + other.z;
        Self { x, y, z }
    }
}

impl Add<Vec3> for Vec3x4 {
    type Output = Self;

    fn add(self, v: Vec3) -> Self {
        let x = self.x + v.x;
        let y = self.y + v.y;
        let z = self.z + v.z;
        Self { x, y, z }
    }
}

impl AddAssign for Vec3x4 {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl AddAssign<Vec3> for Vec3x4 {
    fn add_assign(&mut self, v: Vec3) {
        self.x += v.x;
        self.y += v.y;
        self.z += v.z;
    }
}

impl Sub for Vec3x4 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let x = self.x - other.x;
        let y = self.y - other.y;
        let z = self.z - other.z;
        Self { x, y, z }
    }
}

impl Sub<Vec3> for Vec3x4 {
    type Output = Self;

    fn sub(self, v: Vec3) -> Self {
        let x = self.x - v.x;
        let y = self.y - v.y;
        let z = self.z - v.z;
        Self { x, y, z }
    }
}

impl SubAssign for Vec3x4 {
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl SubAssign<Vec3> for Vec3x4 {
    fn sub_assign(&mut self, v: Vec3) {
        self.x -= v.x;
        self.y -= v.y;
        self.z -= v.z;
    }
}

impl Mul for Vec3x4 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let x = self.x * other.x;
        let y = self.y * other.y;
        let z = self.z * other.z;
        Self { x, y, z }
    }
}

impl Mul<f64> for Vec3x4 {
    type Output = Self;

    fn mul(self, t: f64) -> Self {
        let x = self.x * t;
        let y = self.y * t;
        let z = self.z * t;
        Self { x, y, z }
    }
}

impl Mul<Vec3> for Vec3x4 {
    type Output = Self;

    fn mul(self, v: Vec3) -> Self {
        let x = self.x * v.x;
        let y = self.y * v.y;
        let z = self.z * v.z;
        Self { x, y, z }
    }
}

impl Mul<Vec3x4> for Vec3 {
    type Output = Vec3x4;

    fn mul(self, v: Vec3x4) -> Vec3x4 {
        v * self
    }
}

impl Mul<Vec3x4> for f64 {
    type Output = Vec3x4;

    fn mul(self, v: Vec3x4) -> Vec3x4 {
        v * self
    }
}

impl MulAssign for Vec3x4 {
    fn mul_assign(&mut self, other: Self) {
        self.x *= other.x;
        self.y *= other.y;
        self.z *= other.z;
    }
}

impl MulAssign<f64> for Vec3x4 {
    fn mul_assign(&mut self, t: f64) {
        self.x *= t;
        self.y *= t;
        self.z *= t;
    }
}

impl MulAssign<Vec3> for Vec3x4 {
    fn mul_assign(&mut self, v: Vec3) {
        self.x *= v.x;
        self.y *= v.y;
        self.z *= v.z;
    }
}

impl Div for Vec3x4 {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let x = self.x / other.x;
        let y = self.y / other.y;
        let z = self.z / other.z;
        Self { x, y, z }
    }
}

impl Div<f64> for Vec3x4 {
    type Output = Self;

    fn div(self, t: f64) -> Self {
        let t = 1.0 / t;
        self * t
    }
}

impl Div<Vec3> for Vec3x4 {
    type Output = Self;

    fn div(self, v: Vec3) -> Self {
        let x = self.x / v.x;
        let y = self.y / v.y;
        let z = self.z / v.z;
        Self { x, y, z }
    }
}

impl DivAssign for Vec3x4 {
    fn div_assign(&mut self, other: Self) {
        self.x /= other.x;
        self.y /= other.y;
        self.z /= other.z;
    }
}

impl DivAssign<f64> for Vec3x4 {
    fn div_assign(&mut self, t: f64) {
        let t = 1.0 / t;
        self.x *= t;
        self.y *= t;
        self.z *= t;
    }
}

impl DivAssign<Vec3> for Vec3x4 {
    fn div_assign(&mut self, v: Vec3) {
        self.x /= v.x;
        self.y /= v.y;
        self.z /= v.z;
    }
}

impl Neg for Vec3x4 {
    type Output = Self;

    fn neg(self) -> Self {
        let x = -self.x;
        let y = -self.y;
        let z = -self.z;
        Self { x, y, z }
    }
}

impl Index<usize> for Vec3x4 {
    type Output = F64x4;

    fn index(&self, i: usize) -> &f64x4 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index out of bounds"),
        }
    }
}

impl IndexMut<usize> for Vec3x4 {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Index out of bounds"),
        }
    }
}

impl Debug for Vec3x4 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let (v1, v2, v3, v4) = self.unpack();
        write!(f, "Vec3x4({:?}, {:?}, {:?}, {:?})", v1, v2, v3, v4)
    }
}

impl Display for Vec3x4 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let (v1, v2, v3, v4) = self.unpack();
        write!(f, "Vec3x4({}, {}, {}, {})", v1, v2, v3, v4)
    }
}
