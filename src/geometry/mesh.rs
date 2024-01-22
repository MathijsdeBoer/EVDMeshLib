use std::fs::OpenOptions;

use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array3, Ix3};
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;
use stl_io::create_stl_reader;

use crate::linalg::Vec3;
use crate::rendering::{Intersection, IntersectionSort, Ray};

#[pyclass]
#[derive(Clone)]
pub struct Mesh {
    #[pyo3(get)]
    pub origin: Vec3,

    #[pyo3(get)]
    pub vertices: Vec<Vec3>,
    #[pyo3(get)]
    pub normals: Vec<Vec3>,
    #[pyo3(get)]
    pub areas: Vec<f64>,

    #[pyo3(get)]
    pub triangles: Vec<(usize, usize, usize)>,
}

#[pymethods]
impl Mesh {
    #[new]
    pub fn new(origin: Vec3, vertices: Vec<Vec3>, triangles: Vec<(usize, usize, usize)>) -> Self {
        let mut normals = Vec::with_capacity(triangles.len());
        let mut areas = Vec::with_capacity(triangles.len());

        for (i, j, k) in &triangles {
            let a = vertices[*i];
            let b = vertices[*j];
            let c = vertices[*k];

            let normal = (b - a).cross(&(c - a)).unit_vector();
            let area = (b - a).cross(&(c - a)).length() / 2.0;

            normals.push(normal);
            areas.push(area);
        }

        Self {
            origin,
            vertices,
            normals,
            areas,
            triangles,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (path, num_samples=10_000))]
    pub fn from_file(path: &str, num_samples: usize) -> Self {
        let mut input_file = OpenOptions::new()
            .read(true)
            .open(path)
            .expect("Failed to open file {path}");
        let mut stl_reader =
            create_stl_reader(&mut input_file).expect("Failed to create STL reader");

        let mesh = stl_reader
            .as_indexed_triangles()
            .expect("Failed to read raw mesh");
        let vertices: Vec<Vec3> = mesh
            .vertices
            .iter()
            .map(|v| Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64))
            .collect();

        // Create a vector of triangles based on the original mesh data
        let triangles: Vec<(usize, usize, usize)> = mesh
            .faces
            .iter()
            .map(|f| (f.vertices[0], f.vertices[1], f.vertices[2]))
            .collect();

        let mut result = Self::new(Vec3::zero(), vertices, triangles);

        // Calculate the geometric center of the mesh
        // Create a vector of weights based on the areas of the triangles in the mesh
        let weights: Vec<f64> = result.areas.to_vec();
        // Create a weighted distribution based on the weights
        let distribution = WeightedIndex::new(weights).unwrap();
        let mut rng = thread_rng();

        let mut samples = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            // Randomly select a triangle from the mesh based on the weighted distribution
            let t = result.triangles[distribution.sample(&mut rng)];

            // Get the vertices of the selected triangle
            let a = result.vertices[t.0];
            let b = result.vertices[t.1];
            let c = result.vertices[t.2];

            // Generate two random numbers u and v such that 0 <= u, v, u+v <= 1
            let u = rng.gen_range(0.0..1.0);
            let v = rng.gen_range(0.0..1.0 - u);
            // Calculate w such that u + v + w = 1
            let w = 1.0 - u - v;

            // Calculate a point inside the triangle using barycentric coordinates and add it to the samples
            samples.push(a * u + b * v + c * w);
        }

        // Calculate the geometric center of the mesh by averaging the sample points
        let origin = samples.iter().fold(Vec3::zero(), |acc, v| acc + *v) / num_samples as f64;
        // Set the origin of the mesh to the calculated geometric center
        result.origin = origin;

        result
    }

    pub fn merge_nearby_vertices(&mut self) {
        // Merge vertices that are close together
        // Initialize a vector to map the old vertex indices to the new ones
        let mut vertex_map = Vec::with_capacity(self.vertices.len());
        // Initialize a vector to store the merged vertices
        let mut merged_vertices = Vec::with_capacity(self.vertices.len());

        let bar = ProgressBar::new(self.vertices.len() as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
            )
            .unwrap(),
        );

        // Iterate over each vertex
        for v in self.vertices.iter() {
            let mut merged = false;
            // Iterate over each already processed vertex
            for (j, w) in merged_vertices.iter().enumerate() {
                // If the current vertex is close to an already processed vertex
                if (*v - *w).length() < f64::EPSILON {
                    // Map the current vertex to the close vertex
                    vertex_map.push(j);
                    merged = true;
                    break;
                }
            }

            // If the current vertex is not close to any already processed vertex
            if !merged {
                // Map the current vertex to a new index
                vertex_map.push(merged_vertices.len());
                // Add the current vertex to the list of processed vertices
                merged_vertices.push(*v);
            }

            bar.inc(1);
        }
        bar.finish();

        self.vertices = merged_vertices;

        // Remap vertices
        // Create a new vector of triangles where the vertices of each triangle
        // are remapped using the vertex_map
        self.triangles = self
            .triangles
            .iter()
            .map(|(i, j, k)| (vertex_map[*i], vertex_map[*j], vertex_map[*k]))
            .collect();
    }

    pub fn recalculate_normals(&mut self) {
        // Calculate the normal of each triangle
        self.normals = self
            .triangles
            .iter()
            .map(|(i, j, k)| {
                let a = self.vertices[*i];
                let b = self.vertices[*j];
                let c = self.vertices[*k];

                (b - a).cross(&(c - a)).unit_vector()
            })
            .collect();
    }

    pub fn recalculate_areas(&mut self) {
        // Calculate the area of each triangle
        self.areas = self
            .triangles
            .iter()
            .map(|(i, j, k)| {
                let a = self.vertices[*i];
                let b = self.vertices[*j];
                let c = self.vertices[*k];

                (b - a).cross(&(c - a)).length() / 2.0
            })
            .collect();
    }

    /// Intersects a ray with the mesh.
    ///
    /// This method intersects a ray with the mesh and returns the intersection point.
    /// The intersection point is the point on the mesh that is closest or farthest to the ray's origin,
    /// depending on the sorting parameter.
    ///
    /// # Arguments
    ///
    /// * `ray` - A reference to the ray to intersect with the mesh.
    /// * `sorting` - A value that determines whether to return the nearest or farthest intersection point.
    ///
    /// # Returns
    ///
    /// * `Option<Intersection>` - The intersection point, if it exists. Otherwise, None.
    pub fn intersect(&self, ray: &Ray, sorting: IntersectionSort) -> Option<Intersection> {
        // Parallel iterate over each triangle in the mesh
        let intersections = self.triangles.par_iter().filter_map(|(i, j, k)| {
            // Get the vertices and normal of the current triangle
            let a = self.vertices[*i];
            let b = self.vertices[*j];
            let c = self.vertices[*k];
            let normal = self.normals[*i];

            // If the ray is parallel to the triangle, there is no intersection
            if normal.dot(&ray.direction).abs() < f64::EPSILON {
                return None;
            }

            // Calculate the distance from the ray's origin to the plane of the triangle
            let d: f64 = -normal.dot(&a);
            let t: f64 = -(normal.dot(&ray.origin) + d) / normal.dot(&ray.direction);
            // If the triangle is behind the ray, there is no intersection
            if t < 0.0 {
                return None;
            }

            // Calculate the intersection point with the plane
            let p: Vec3 = ray.at(t);

            // Calculate the vectors from the vertices of the triangle to the intersection point
            let ab: Vec3 = b - a;
            let bc: Vec3 = c - b;
            let ca: Vec3 = a - c;

            let ap: Vec3 = p - a;
            let bp: Vec3 = p - b;
            let cp: Vec3 = p - c;

            // Check if the intersection point is inside the triangle
            if normal.dot(&ab.cross(&ap)) >= 0.0
                && normal.dot(&bc.cross(&bp)) >= 0.0
                && normal.dot(&ca.cross(&cp)) >= 0.0
            {
                // If the intersection point is inside the triangle, return it
                Some(Intersection {
                    distance: t,
                    position: p,
                    normal,
                })
            } else {
                // If the intersection point is outside the triangle, there is no intersection
                None
            }
        });

        println!("{} intersections", intersections.clone().count());

        // Return the intersection point that is closest or farthest to the ray's origin,
        // depending on the sorting parameter
        match sorting {
            IntersectionSort::Nearest => {
                intersections.min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap())
            }
            IntersectionSort::Farthest => {
                intersections.max_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap())
            }
        }
    }

    pub fn triangles_as_vertex_array(&self, py: Python<'_>) -> Py<PyArray<f64, Ix3>> {
        let mut triangles = Array3::zeros((self.triangles.len(), 3, 3));
        for (i, triangle) in self.triangles.iter().enumerate() {
            triangles[[i, 0, 0]] = self.vertices[triangle.0].x;
            triangles[[i, 0, 1]] = self.vertices[triangle.0].y;
            triangles[[i, 0, 2]] = self.vertices[triangle.0].z;

            triangles[[i, 1, 0]] = self.vertices[triangle.1].x;
            triangles[[i, 1, 1]] = self.vertices[triangle.1].y;
            triangles[[i, 1, 2]] = self.vertices[triangle.1].z;

            triangles[[i, 2, 0]] = self.vertices[triangle.2].x;
            triangles[[i, 2, 1]] = self.vertices[triangle.2].y;
            triangles[[i, 2, 2]] = self.vertices[triangle.2].z;
        }
        triangles.into_pyarray(py).to_owned()
    }

    #[getter]
    pub fn num_triangles(&self) -> usize {
        self.triangles.len()
    }

    #[getter]
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    #[getter]
    pub fn surface_area(&self) -> f64 {
        self.areas.iter().sum()
    }

    #[getter]
    pub fn volume(&self) -> f64 {
        self.triangles
            .par_iter()
            .map(|(i, j, k)| self.signed_volume_of_triangle(*i, *j, *k))
            .sum::<f64>()
            .abs()
    }

    #[getter]
    pub fn bounding_box(&self) -> (Vec3, Vec3) {
        let mut min = Vec3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max = Vec3::new(f64::MIN, f64::MIN, f64::MIN);

        for v in &self.vertices {
            if v.x < min.x {
                min.x = v.x;
            }
            if v.y < min.y {
                min.y = v.y;
            }
            if v.z < min.z {
                min.z = v.z;
            }

            if v.x > max.x {
                max.x = v.x;
            }
            if v.y > max.y {
                max.y = v.y;
            }
            if v.z > max.z {
                max.z = v.z;
            }
        }

        (min, max)
    }
}

impl Mesh {
    fn signed_volume_of_triangle(&self, i: usize, j: usize, k: usize) -> f64 {
        // Per http://chenlab.ece.cornell.edu/Publication/Cha/icip01_Cha.pdf
        let v321 = self.vertices[k].x * self.vertices[j].y * self.vertices[i].z;
        let v231 = self.vertices[j].x * self.vertices[k].y * self.vertices[i].z;
        let v312 = self.vertices[k].x * self.vertices[i].y * self.vertices[j].z;
        let v132 = self.vertices[i].x * self.vertices[k].y * self.vertices[j].z;
        let v213 = self.vertices[j].x * self.vertices[i].y * self.vertices[k].z;
        let v123 = self.vertices[i].x * self.vertices[j].y * self.vertices[k].z;
        (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123)
    }
}
