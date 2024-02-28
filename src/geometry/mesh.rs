use std::fmt::Debug;
use std::fs::OpenOptions;
use std::ops::Index;

use crate::geometry::deformer::Deformer;
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::Bvh;
use nalgebra::{Point3, Vector3};
use ndarray::{Array3, Ix3};
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;
use stl_io::{create_stl_reader, write_stl};

use crate::linalg::{Mat4, Vec3};
use crate::rendering::{Intersection, IntersectionSort, Ray};

/// Intersects a ray with a triangle.
///
/// This function attempts to intersect a ray with a triangle.
/// If the ray intersects the triangle, the function returns the intersection point.
/// Otherwise, it returns None.
///
/// # Arguments
///
/// * `ray` - A reference to the ray to intersect with the triangle.
/// * `triangle` - A reference to the triangle to intersect with the ray.
/// * `vertices` - A reference to the vertices of the mesh that contains the triangle.
/// * `epsilon` - A small value used to determine if the ray is parallel to the triangle.
///
/// # Returns
///
/// * `Option<Intersection>` - The intersection point, if it exists. Otherwise, None.
fn ray_triangle_intersect(ray: &Ray, triangle: &Triangle, epsilon: f64) -> Option<Intersection> {
    // Get the vertices and normal of the current triangle
    let a = triangle.a;
    let b = triangle.b;
    let c = triangle.c;
    let normal = triangle.normal;

    // If the ray is parallel to the triangle, there is no intersection
    // due to floating point precision, we use an epsilon value to determine if the ray is parallel
    if normal.dot(&ray.direction).abs() < epsilon {
        return None;
    }

    // Calculate the distance from the ray's origin to the plane of the triangle
    let d: f64 = -normal.dot(&a);
    let t: f64 = -(normal.dot(&ray.origin) + d) / normal.dot(&ray.direction);
    // Check if triangle is behind ray
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
}

/// A triangle in a mesh.
///
/// This struct represents a triangle in a mesh.
/// It contains the indices of the vertices that form the triangle,
/// as well as the normal and area of the triangle.
///
/// # Fields
///
/// * `index` - The index of the triangle in the mesh.
/// * `a` - The index of the first vertex of the triangle.
/// * `b` - The index of the second vertex of the triangle.
/// * `c` - The index of the third vertex of the triangle.
/// * `normal` - The normal of the triangle.
/// * `area` - The area of the triangle.
///
/// # Methods
///
/// * `vertices` - Returns the indices of the vertices that form the triangle as a vector.
#[pyclass]
#[derive(Clone, Copy)]
pub struct Triangle {
    pub index: usize,

    #[pyo3(get)]
    pub a: Vec3,
    #[pyo3(get)]
    pub b: Vec3,
    #[pyo3(get)]
    pub c: Vec3,

    pub idx_a: usize,
    pub idx_b: usize,
    pub idx_c: usize,

    #[pyo3(get)]
    pub normal: Vec3,
    #[pyo3(get)]
    pub area: f64,

    node_index: usize,
}

impl Triangle {
    pub fn new(
        index: usize,
        a: Vec3,
        b: Vec3,
        c: Vec3,
        idx_a: usize,
        idx_b: usize,
        idx_c: usize,
        normal: Vec3,
        area: f64,
    ) -> Self {
        Self {
            index,
            a,
            b,
            c,
            idx_a,
            idx_b,
            idx_c,
            normal,
            area,
            node_index: 0,
        }
    }

    /// Returns the indices of the vertices that form the triangle as a vector.
    ///
    /// # Returns
    ///
    /// * `Vec<usize>` - The indices of the vertices that form the triangle.
    pub fn vertices(&self) -> Vec<Vec3> {
        vec![self.a, self.b, self.c]
    }
}

impl Bounded<f64, 3> for Triangle {
    fn aabb(&self) -> Aabb<f64, 3> {
        // Include a small epsilon value to avoid floating point precision issues
        // For example, triangles that are parallel to the x, y, or z axis
        // may not be intersected by rays due to floating point precision
        let min = Vec3::new(
            self.a.x.min(self.b.x).min(self.c.x) - 1e-8,
            self.a.y.min(self.b.y).min(self.c.y) - 1e-8,
            self.a.z.min(self.b.z).min(self.c.z) - 1e-8,
        );
        let max = Vec3::new(
            self.a.x.max(self.b.x).max(self.c.x) + 1e-8,
            self.a.y.max(self.b.y).max(self.c.y) + 1e-8,
            self.a.z.max(self.b.z).max(self.c.z) + 1e-8,
        );
        Aabb::with_bounds(
            Point3::new(min.x, min.y, min.z),
            Point3::new(max.x, max.y, max.z),
        )
    }
}

impl BHShape<f64, 3> for Triangle {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

impl Index<usize> for Triangle {
    type Output = Vec3;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.a,
            1 => &self.b,
            2 => &self.c,
            _ => panic!("Index out of bounds"),
        }
    }
}

impl PartialEq for Triangle {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Debug for Triangle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Triangle")
            .field("index", &self.index)
            .field("a", &self.a)
            .field("b", &self.b)
            .field("c", &self.c)
            .field("normal", &self.normal)
            .field("area", &self.area)
            .finish()
    }
}

/// A mesh.
///
/// This struct represents a mesh.
/// It contains the vertices and triangles that form the mesh,
/// as well as the origin of the mesh.
///
/// # Fields
///
/// * `origin` - The origin of the mesh.
/// * `vertices` - The vertices of the mesh.
/// * `triangles` - The triangles that form the mesh.
///
/// # Methods
///
/// * `new` - Creates a new mesh from the origin, vertices, and triangles.
/// * `load` - Loads a mesh from an STL file.
/// * `save` - Saves the mesh to an STL file.
/// * `uniform_sample` - Samples the mesh surface uniformly.
/// * `recalculate_normals` - Recalculates the normals of the triangles in the mesh.
/// * `recalculate_areas` - Recalculates the areas of the triangles in the mesh.
/// * `recalculate_origin` - Recalculates the origin of the mesh.
/// * `intersect` - Intersects a ray with the mesh.
/// * `triangles_as_vertex_array` - Returns the triangles of the mesh as a 3D array of vertices.
/// * `laplacian_smooth` - Smooths the mesh using the Laplacian smoothing algorithm.
/// * `num_triangles` - Returns the number of triangles in the mesh.
/// * `num_vertices` - Returns the number of vertices in the mesh.
/// * `surface_area` - Returns the surface area of the mesh.
/// * `volume` - Returns the volume of the mesh.
/// * `bounding_box` - Returns the bounding box of the mesh.
/// * `get_vertices` - Returns the vertices of the mesh.
/// * `set_vertices` - Sets the vertices of the mesh.
/// * `get_triangles` - Returns the triangles of the mesh.
/// * `set_triangles` - Sets the triangles of the mesh.
#[pyclass]
#[derive(Clone)]
pub struct Mesh {
    #[pyo3(get)]
    pub origin: Vec3,

    vertices: Vec<Vec3>,
    triangles: Vec<Triangle>,
    bvh: Bvh<f64, 3>,
}

#[pymethods]
impl Mesh {
    /// Creates a new mesh from the origin, vertices, and triangles.
    ///
    /// # Arguments
    ///
    /// * `origin` - The origin of the mesh.
    /// * `vertices` - The vertices of the mesh.
    /// * `triangles` - The triangles that form the mesh.
    ///
    /// # Returns
    ///
    /// * `Mesh` - The new mesh.
    #[new]
    pub fn new(origin: Vec3, vertices: Vec<Vec3>, triangles: Vec<(usize, usize, usize)>) -> Self {
        let mut normals = Vec::with_capacity(triangles.len());
        let mut areas = Vec::with_capacity(triangles.len());
        let mut in_triangles = Vec::with_capacity(triangles.len());

        for (index, (i, j, k)) in triangles.iter().enumerate() {
            let a = vertices[*i];
            let b = vertices[*j];
            let c = vertices[*k];

            let normal = (b - a).cross(&(c - a)).unit_vector();
            let area = (b - a).cross(&(c - a)).length() / 2.0;

            normals.push(normal);
            areas.push(area);
            in_triangles.push(Triangle::new(index, a, b, c, *i, *j, *k, normal, area));
        }

        Self {
            origin,
            vertices,
            triangles: in_triangles.clone(),
            bvh: Bvh::build(&mut in_triangles),
        }
    }

    /// Loads a mesh from an STL file.
    ///
    /// This method loads a mesh from an STL file and returns it.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the STL file.
    /// * `num_samples` - The number of samples to use for the origin calculation.
    ///
    /// # Returns
    ///
    /// * `Mesh` - The loaded mesh.
    #[staticmethod]
    #[pyo3(signature = (path, num_samples=10_000))]
    pub fn load(path: &str, num_samples: usize) -> Self {
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
        let mut triangles: Vec<Triangle> = mesh
            .faces
            .iter()
            .enumerate()
            .map(|(index, f)| {
                let a = vertices[f.vertices[0]];
                let b = vertices[f.vertices[1]];
                let c = vertices[f.vertices[2]];
                let area = (b - a).cross(&(c - a)).length() / 2.0;

                Triangle::new(
                    index,
                    a,
                    b,
                    c,
                    f.vertices[0],
                    f.vertices[1],
                    f.vertices[2],
                    Vec3::new(f.normal[0] as f64, f.normal[1] as f64, f.normal[2] as f64),
                    area,
                )
            })
            .collect();

        let mut mesh = Self {
            origin: Vec3::new(0.0, 0.0, 0.0),
            vertices,
            triangles: triangles.clone(),
            bvh: Bvh::build(&mut triangles),
        };

        let origin = mesh
            .uniform_sample(num_samples)
            .iter()
            .fold(Vec3::new(0.0, 0.0, 0.0), |acc, v| acc + *v)
            / num_samples as f64;
        mesh.origin = origin;

        mesh
    }

    /// Saves the mesh to an STL file.
    ///
    /// This method saves the mesh to an STL file.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the STL file.
    ///
    /// # Returns
    ///
    /// * `None`
    pub fn save(&self, path: &str) {
        let triangles: Vec<stl_io::Triangle> = self
            .triangles
            .iter()
            .map(|t| {
                let a = t.a;
                let b = t.b;
                let c = t.c;
                let normal = stl_io::Normal::new(t.normal.as_f32_array());

                stl_io::Triangle {
                    normal,
                    vertices: [
                        stl_io::Vertex::new(a.as_f32_array()),
                        stl_io::Vertex::new(b.as_f32_array()),
                        stl_io::Vertex::new(c.as_f32_array()),
                    ],
                }
            })
            .collect();

        let mut output_file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(path)
            .expect("Failed to open file {path}");

        write_stl(&mut output_file, triangles.iter()).expect("Failed to write STL file");
    }

    /// Samples the mesh surface uniformly.
    ///
    /// This method samples the mesh surface uniformly and returns the samples.
    ///
    /// # Arguments
    ///
    /// * `num_samples` - The number of samples to take.
    ///
    /// # Returns
    ///
    /// * `Vec<Vec3>` - The samples.
    #[pyo3(signature = (num_samples=1_000_000))]
    pub fn uniform_sample(&self, num_samples: usize) -> Vec<Vec3> {
        let mut rng = thread_rng();
        let mut samples = Vec::with_capacity(num_samples);

        let weights = &self.triangles.iter().map(|t| t.area).collect::<Vec<f64>>();
        let dist = WeightedIndex::new(weights).unwrap();

        for _ in 0..num_samples {
            let triangle = &self.triangles[dist.sample(&mut rng)];
            let a = triangle.a;
            let b = triangle.b;
            let c = triangle.c;

            let u: f64 = rng.gen_range(0.0..1.0);
            let v: f64 = rng.gen_range(0.0..1.0);

            let p = (1.0 - u.sqrt()) * a + (u.sqrt() * (1.0 - v)) * b + (u.sqrt() * v) * c;

            samples.push(p);
        }

        samples
    }

    pub fn transform(&mut self, transform: Mat4) {
        self.vertices.par_iter_mut().for_each(|vertex| {
            *vertex = transform * *vertex;
        });

        self.update_triangles();
    }

    pub fn deform(&mut self, deformer: Deformer) {
        self.vertices.par_iter_mut().for_each(|vertex| {
            *vertex = deformer.deform_vertex(vertex);
        });

        self.update_triangles();
    }

    #[pyo3(signature = (num_samples=1_000_000))]
    pub fn recalculate(&mut self, num_samples: usize) {
        self.rebuild_bvh();
        self.recalculate_normals();
        self.recalculate_areas();
        self.recalculate_origin(num_samples);
        self.update_triangles();
    }

    pub fn rebuild_bvh(&mut self) {
        self.bvh = Bvh::build(&mut self.triangles);
    }

    pub fn flatten_bvh(&mut self) {
        self.bvh.flatten();
    }

    pub fn print_bvh(&self) {
        self.bvh.pretty_print();
    }

    pub fn recalculate_normals(&mut self) {
        // Calculate the normal of each triangle
        self.triangles.par_iter_mut().for_each(|triangle| {
            let a = triangle.a;
            let b = triangle.b;
            let c = triangle.c;

            triangle.normal = (b - a).cross(&(c - a)).unit_vector();
        });
    }

    pub fn recalculate_areas(&mut self) {
        // Calculate the area of each triangle
        self.triangles.par_iter_mut().for_each(|triangle| {
            let a = triangle.a;
            let b = triangle.b;
            let c = triangle.c;

            triangle.area = (b - a).cross(&(c - a)).length() / 2.0;
        });
    }

    #[pyo3(signature = (num_samples=1_000_000))]
    pub fn recalculate_origin(&mut self, num_samples: usize) {
        let origin = self
            .uniform_sample(num_samples)
            .iter()
            .fold(Vec3::new(0.0, 0.0, 0.0), |acc, v| acc + *v)
            / num_samples as f64;
        self.origin = origin;
    }

    pub fn update_triangles(&mut self) {
        self.triangles = self
            .triangles
            .par_iter()
            .enumerate()
            .map(|(index, triangle)| {
                let a = self.vertices[triangle.idx_a];
                let b = self.vertices[triangle.idx_b];
                let c = self.vertices[triangle.idx_c];

                let normal = (b - a).cross(&(c - a)).unit_vector();
                let area = (b - a).cross(&(c - a)).length() / 2.0;

                Triangle::new(
                    index,
                    a,
                    b,
                    c,
                    triangle.idx_a,
                    triangle.idx_b,
                    triangle.idx_c,
                    normal,
                    area,
                )
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
    #[pyo3(signature = (ray, sorting, epsilon=1e-8))]
    pub fn intersect(
        &self,
        ray: &Ray,
        sorting: IntersectionSort,
        epsilon: f64,
    ) -> Option<Intersection> {
        let bvh_ray = bvh::ray::Ray::new(
            Point3::new(ray.origin.x, ray.origin.y, ray.origin.z),
            Vector3::new(ray.direction.x, ray.direction.y, ray.direction.z),
        );
        let triangles_to_intersect = self.bvh.traverse(&bvh_ray, &self.triangles);

        let intersections = triangles_to_intersect
            .par_iter()
            .filter_map(|triangle| ray_triangle_intersect(ray, triangle, epsilon))
            .collect::<Vec<Intersection>>();

        // Return the intersection point that is closest or farthest to the ray's origin,
        // depending on the sorting parameter
        match sorting {
            IntersectionSort::Nearest => intersections
                .into_par_iter()
                .min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap()),
            IntersectionSort::Farthest => intersections
                .into_par_iter()
                .max_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap()),
        }
    }

    pub fn triangles_as_vertex_array(&self, py: Python<'_>) -> Py<PyArray<f64, Ix3>> {
        let mut triangles = Array3::zeros((self.triangles.len(), 3, 3));
        for (i, triangle) in self.triangles.iter().enumerate() {
            triangles[[i, 0, 0]] = triangle.a.x;
            triangles[[i, 0, 1]] = triangle.a.y;
            triangles[[i, 0, 2]] = triangle.a.z;

            triangles[[i, 1, 0]] = triangle.b.x;
            triangles[[i, 1, 1]] = triangle.b.y;
            triangles[[i, 1, 2]] = triangle.b.z;

            triangles[[i, 2, 0]] = triangle.c.x;
            triangles[[i, 2, 1]] = triangle.c.y;
            triangles[[i, 2, 2]] = triangle.c.z;
        }
        triangles.into_pyarray(py).to_owned()
    }

    #[pyo3(signature = (iterations=10, smoothing_factor=0.5))]
    pub fn laplacian_smooth(&mut self, iterations: usize, smoothing_factor: f64) {
        let mut vertex_neighbours: Vec<Vec<usize>> = vec![Vec::new(); self.vertices.len()];

        for triangle in self.triangles.iter() {
            vertex_neighbours[triangle.idx_a].push(triangle.idx_b);
            vertex_neighbours[triangle.idx_a].push(triangle.idx_c);

            vertex_neighbours[triangle.idx_b].push(triangle.idx_a);
            vertex_neighbours[triangle.idx_b].push(triangle.idx_c);

            vertex_neighbours[triangle.idx_c].push(triangle.idx_a);
            vertex_neighbours[triangle.idx_c].push(triangle.idx_b);
        }

        for _ in 0..iterations {
            let neighbour_means: Vec<Vec3> = vertex_neighbours
                .par_iter()
                .map(|neighbours| {
                    let mut mean = Vec3::new(0.0, 0.0, 0.0);
                    for neighbour in neighbours {
                        mean += self.vertices[*neighbour];
                    }
                    mean / neighbours.len() as f64
                })
                .collect();

            self.vertices
                .par_iter_mut()
                .zip(neighbour_means.par_iter())
                .for_each(|(vertex, mean)| {
                    *vertex = *vertex + (*mean - *vertex) * smoothing_factor;
                });
        }

        self.update_triangles();
    }

    #[getter]
    pub fn num_triangles(&self) -> usize {
        self.triangles.len()
    }

    #[getter]
    pub fn surface_area(&self) -> f64 {
        self.triangles.par_iter().map(|t| t.area).sum()
    }

    #[getter]
    pub fn volume(&self) -> f64 {
        self.triangles
            .par_iter()
            .map(|t| self.signed_volume_of_triangle(t.a, t.b, t.c))
            .sum::<f64>()
            .abs()
    }

    #[getter]
    pub fn get_triangles(&self) -> Vec<Triangle> {
        self.triangles.clone()
    }

    #[getter]
    pub fn get_vertices(&self) -> Vec<Vec3> {
        self.vertices.clone()
    }

    #[setter]
    pub fn set_vertices(&mut self, vertices: Vec<Vec3>) {
        self.vertices = vertices;
        self.update_triangles();
    }
}

impl Mesh {
    fn signed_volume_of_triangle(&self, i: Vec3, j: Vec3, k: Vec3) -> f64 {
        // Per http://chenlab.ece.cornell.edu/Publication/Cha/icip01_Cha.pdf
        let v321 = k.x * j.y * i.z;
        let v231 = j.x * k.y * i.z;
        let v312 = k.x * i.y * j.z;
        let v132 = i.x * k.y * j.z;
        let v213 = j.x * i.y * k.z;
        let v123 = i.x * j.y * k.z;
        (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123)
    }
}
