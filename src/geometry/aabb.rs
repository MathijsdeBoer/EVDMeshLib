use pyo3::prelude::*;

use crate::geometry::Triangle;
use crate::linalg::Vec3;
use crate::rendering::Ray;

#[pyclass]
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

#[pymethods]
impl Aabb {
    #[new]
    pub fn new() -> Self {
        Self {
            min: Vec3::new(f64::MAX, f64::MAX, f64::MAX),
            max: Vec3::new(f64::MIN, f64::MIN, f64::MIN),
        }
    }

    #[pyo3(signature = (ray, epsilon = 1e-8))]
    pub fn intersect(&self, ray: &Ray, epsilon: f64) -> bool {
        let mut tmin = f64::MAX;
        let mut tmax = f64::MIN;

        for i in 0..3 {
            if ray.direction[i] != 0.0 {
                let t1 = (self.min[i] - ray.origin[i]) / ray.direction[i];
                let t2 = (self.max[i] - ray.origin[i]) / ray.direction[i];

                tmin = tmin.min(t1.min(t2));
                tmax = tmax.max(t1.max(t2));
            } else if ray.origin[i] < self.min[i] || ray.origin[i] > self.max[i] {
                return false;
            }
        }

        tmax >= tmin && tmax >= epsilon
    }

    pub fn grow(&mut self, v: Vec3) {
        self.min.x = self.min.x.min(v.x);
        self.min.y = self.min.y.min(v.y);
        self.min.z = self.min.z.min(v.z);

        self.max.x = self.max.x.max(v.x);
        self.max.y = self.max.y.max(v.y);
        self.max.z = self.max.z.max(v.z);
    }

    pub fn is_inside(&self, v: Vec3) -> bool {
        v.x >= self.min.x
            && v.x <= self.max.x
            && v.y >= self.min.y
            && v.y <= self.max.y
            && v.z >= self.min.z
            && v.z <= self.max.z
    }

    #[getter]
    pub fn longest_axis(&self) -> usize {
        let x = self.max.x - self.min.x;
        let y = self.max.y - self.min.y;
        let z = self.max.z - self.min.z;

        if x > y && x > z {
            0
        } else if y > z {
            1
        } else {
            2
        }
    }
}

impl Aabb {
    pub fn grow_from_triangles(&mut self, vertices: &[Vec3], triangles: &[Triangle]) {
        triangles.iter().for_each(|t| {
            self.grow(vertices[t.a]);
            self.grow(vertices[t.b]);
            self.grow(vertices[t.c]);
        });
    }
}

impl Default for Aabb {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Triangle;
    use crate::linalg::Vec3;
    use crate::rendering::Ray;

    #[test]
    fn test_new() {
        let aabb = Aabb::new();
        assert_eq!(aabb.min, Vec3::new(f64::MAX, f64::MAX, f64::MAX));
        assert_eq!(aabb.max, Vec3::new(f64::MIN, f64::MIN, f64::MIN));
    }

    #[test]
    fn test_intersect_ray_inside() {
        let mut aabb = Aabb::new();
        aabb.grow(Vec3::new(1.0, 1.0, 1.0));
        aabb.grow(Vec3::new(-1.0, -1.0, -1.0));

        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, 1.0));
        assert!(aabb.intersect(&ray, 1e-8));
    }

    #[test]
    fn test_intersect() {
        let mut aabb = Aabb::new();
        aabb.grow(Vec3::new(1.0, 1.0, 1.0));
        aabb.grow(Vec3::new(-1.0, -1.0, -1.0));

        let ray = Ray::new(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        assert!(aabb.intersect(&ray, 1e-8));
    }

    #[test]
    fn test_intersect2() {
        let mut aabb = Aabb::new();
        aabb.grow(Vec3::new(1.0, 1.0, 1.0));
        aabb.grow(Vec3::new(-1.0, -1.0, -1.0));

        let ray = Ray::new(
            Vec3::new(2.0, 2.0, 2.0),
            Vec3::new(-1.0, -1.0, -1.0).unit_vector(),
        );
        assert!(aabb.intersect(&ray, 1e-8));
    }

    #[test]
    fn test_intersect3() {
        let mut aabb = Aabb::new();
        aabb.grow(Vec3::new(1.0, 1.0, 1.0));
        aabb.grow(Vec3::new(-1.0, -1.0, -1.0));

        let ray = Ray::new(
            Vec3::new(2.0, 2.0, 2.0),
            Vec3::new(1.0, 1.0, 1.0).unit_vector(),
        );
        assert!(!aabb.intersect(&ray, 1e-8));
    }

    #[test]
    fn test_intersect4() {
        let mut aabb = Aabb::new();
        aabb.grow(Vec3::new(1.0, 1.0, 1.0));
        aabb.grow(Vec3::new(-1.0, -1.0, -1.0));

        let ray = Ray::new(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        assert!(!aabb.intersect(&ray, 1e-8));
    }

    #[test]
    fn test_grow() {
        let mut aabb = Aabb::new();
        aabb.grow(Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(aabb.min, Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(aabb.max, Vec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn test_is_inside() {
        let mut aabb = Aabb::new();
        aabb.grow(Vec3::new(1.0, 1.0, 1.0));
        assert!(!aabb.is_inside(Vec3::new(0.5, 0.5, 0.5)));
        assert!(aabb.is_inside(Vec3::new(1.0, 1.0, 1.0)));
    }

    #[test]
    fn test_longest_axis() {
        let mut aabb = Aabb::new();
        aabb.grow(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(aabb.longest_axis(), 2);
    }

    #[test]
    fn test_grow_from_triangles() {
        let mut aabb = Aabb::new();
        let vertices = vec![
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(2.0, 2.0, 2.0),
            Vec3::new(3.0, 3.0, 3.0),
        ];
        let triangles = vec![Triangle {
            a: 0,
            b: 1,
            c: 2,
            index: 0,
            normal: Vec3::zero(),
            area: 0.0,
        }];
        aabb.grow_from_triangles(&vertices, &triangles);
        assert_eq!(aabb.min, Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(aabb.max, Vec3::new(3.0, 3.0, 3.0));
    }
}
