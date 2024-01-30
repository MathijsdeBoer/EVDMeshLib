use pyo3::prelude::*;
use rayon::prelude::*;

use crate::geometry::aabb::Aabb;
use crate::geometry::Triangle;
use crate::linalg::Vec3;
use crate::rendering::Ray;

#[pyclass]
#[derive(Clone)]
pub struct BvhNode {
    #[pyo3(get)]
    aabb: Aabb,
    #[pyo3(get)]
    left: Option<usize>,
    #[pyo3(get)]
    right: Option<usize>,
    #[pyo3(get)]
    triangles: Vec<usize>,
}

#[pymethods]
impl BvhNode {
    pub fn intersect(&self, ray: &Ray, epsilon: f64) -> bool {
        self.aabb.intersect(ray, epsilon)
    }

    #[getter]
    pub fn num_triangles(&self) -> usize {
        self.triangles.len()
    }

    pub fn __str__(&self) -> String {
        format!(
            "Aabb: {:?}, Left: {:?}, Right: {:?}, Triangles: {:?}",
            self.aabb, self.left, self.right, self.triangles
        )
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Aabb: {:?}, Left: {:?}, Right: {:?}, Triangles: {:?}",
            self.aabb, self.left, self.right, self.triangles
        )
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Bvh {
    #[pyo3(get)]
    pub nodes: Vec<BvhNode>,

    is_trimmed: bool,
}

#[pymethods]
impl Bvh {
    #[pyo3(signature = (ray, epsilon = 1e-8))]
    pub fn intersect(&self, ray: &Ray, epsilon: f64) -> Vec<usize> {
        if self.is_trimmed {
            self.trimmed_intersect(ray, epsilon)
        } else {
            self.recursive_intersect(ray, epsilon, 0usize)
        }
    }

    pub fn trim(&mut self) {
        if self.is_trimmed {
            println!("BVH is already trimmed");
            return;
        }

        let new_nodes = self
            .nodes
            .iter()
            .filter_map(|node| {
                if node.triangles.is_empty() {
                    None
                } else {
                    Some(node.clone())
                }
            })
            .collect::<Vec<BvhNode>>();

        println!(
            "Trimmed BVH from {} nodes to {} nodes",
            self.nodes.len(),
            new_nodes.len()
        );

        self.nodes.clear();
        self.nodes = new_nodes;
        self.is_trimmed = true;
    }

    #[getter]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    #[getter]
    pub fn num_levels(&self) -> usize {
        if self.nodes.is_empty() {
            0usize
        } else {
            self.find_depth(0usize)
        }
    }
}

impl Bvh {
    pub fn new(
        vertices: &Vec<Vec3>,
        triangles: &Vec<Triangle>,
        max_depth: usize,
        min_leaf_size: usize,
    ) -> Self {
        // Initialise Root Node
        // Build Tree
        let mut result = Bvh {
            nodes: vec![],
            is_trimmed: false,
        };
        result.build_tree(vertices, triangles, 0, max_depth, min_leaf_size);

        result
    }

    fn recursive_intersect(&self, ray: &Ray, epsilon: f64, node_index: usize) -> Vec<usize> {
        let mut candidates: Vec<usize> = Vec::new();

        let current_node = &self.nodes[node_index];

        if current_node.aabb.intersect(ray, epsilon) {
            if current_node.triangles.is_empty() {
                if let Some(left) = current_node.left {
                    candidates.extend(self.recursive_intersect(ray, epsilon, left));
                }
                if let Some(right) = current_node.right {
                    candidates.extend(self.recursive_intersect(ray, epsilon, right));
                }
            } else {
                candidates.extend(&current_node.triangles);
            }
        }

        candidates
    }

    fn trimmed_intersect(&self, ray: &Ray, epsilon: f64) -> Vec<usize> {
        self.nodes
            .par_iter()
            .filter_map(|node| {
                if node.aabb.intersect(ray, epsilon) {
                    Some(node.triangles.clone())
                } else {
                    None
                }
            })
            .flatten()
            .collect()
    }

    fn build_tree(
        &mut self,
        vertices: &Vec<Vec3>,
        triangles: &Vec<Triangle>,
        current_depth: usize,
        max_depth: usize,
        min_leaf_size: usize,
    ) -> usize {
        if current_depth >= max_depth || triangles.len() <= min_leaf_size {
            let mut node = BvhNode {
                aabb: Aabb::default(),
                left: None,
                right: None,
                triangles: triangles.iter().map(|t| t.index).collect(),
            };
            node.aabb.grow_from_triangles(vertices, triangles);

            self.nodes.push(node);
            return self.nodes.len() - 1;
        }

        let mut aabb = Aabb::default();
        aabb.grow_from_triangles(vertices, triangles);
        self.nodes.push(BvhNode {
            aabb,
            left: None,
            right: None,
            triangles: Vec::new(),
        });
        let current_node_index = self.nodes.len() - 1;

        let longest_axis = aabb.longest_axis();
        let midpoint = (aabb.min[longest_axis] + aabb.max[longest_axis]) / 2.0;

        let mut left_triangles = Vec::new();
        let mut right_triangles = Vec::new();

        triangles.iter().for_each(|t| {
            let mut is_in_left = false;
            let mut is_in_right = false;

            for v in t.vertices() {
                if vertices[v][longest_axis] < midpoint {
                    is_in_left = true;
                } else {
                    is_in_right = true;
                }
            }

            if is_in_left {
                left_triangles.push(*t);
            }
            if is_in_right {
                right_triangles.push(*t);
            }
        });

        let left_node = self.build_tree(
            vertices,
            &left_triangles,
            current_depth + 1,
            max_depth,
            min_leaf_size,
        );
        let right_node = self.build_tree(
            vertices,
            &right_triangles,
            current_depth + 1,
            max_depth,
            min_leaf_size,
        );

        self.nodes[current_node_index].left = Some(left_node);
        self.nodes[current_node_index].right = Some(right_node);

        current_node_index
    }

    fn find_depth(&self, node_index: usize) -> usize {
        let node = &self.nodes[node_index];

        let left_depth = if let Some(left) = node.left {
            self.find_depth(left)
        } else {
            0usize
        };

        let right_depth = if let Some(right) = node.right {
            self.find_depth(right)
        } else {
            0usize
        };

        1usize + left_depth.max(right_depth)
    }
}
