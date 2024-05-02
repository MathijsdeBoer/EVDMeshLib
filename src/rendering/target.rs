use ndarray::{Array3, Ix3};
use numpy::{IntoPyArray, PyArray};
use pyo3::{pyfunction, Py, Python};
use rayon::prelude::*;

use crate::geometry::Mesh;
use crate::linalg::{Mat4, Vec3};
use crate::rendering::{Camera, CameraType, Intersection, IntersectionSort, Ray, Renderer};

#[pyfunction]
#[pyo3(signature = (distance, thickness, distance_weight=0.5, thickness_threshold=10., depth_threshold=70.0))]
pub fn objective_function(
    distance: f64,
    thickness: f64,
    distance_weight: f64,
    thickness_threshold: f64,
    depth_threshold: f64,
) -> f64 {
    let total_depth = distance + thickness * 0.5;
    let depth = 1.0f64.max(total_depth - depth_threshold).powf(2.0);

    let distance_part = distance * distance_weight;
    let thickness_part = (1.0 - distance_weight) * 0.0f64.max(thickness - thickness_threshold);

    (distance_part - thickness_part) * depth
}

fn sample_ray(mesh: &Mesh, ray: &Ray, epsilon: f64) -> (Vec3, f64, f64, bool) {
    let intersection = mesh.intersect(ray, IntersectionSort::Nearest, epsilon);

    if let Some(intersection) = intersection {
        let distance = intersection.distance;

        let new_ray = Ray::new(intersection.position + ray.direction * 1e-8, ray.direction);
        let new_intersection = mesh.intersect(&new_ray, IntersectionSort::Nearest, epsilon);
        if let Some(new_intersection) = new_intersection {
            let thickness = (intersection.position - new_intersection.position).length();
            let midpoint = (intersection.position + new_intersection.position) * 0.5;

            (midpoint, distance, thickness, true)
        } else {
            (Vec3::zero(), distance, 0.0, false)
        }
    } else {
        (Vec3::zero(), f64::INFINITY, 0.0, false)
    }
}

fn process_intersection(
    mesh: &Mesh,
    origin: &Vec3,
    intersection: &Option<Intersection>,
    check_radially: bool,
    radius: f64,
    radial_samples: usize,
    radial_rings: usize,
    epsilon: f64,
) -> (f64, f64) {
    let mut distance = f64::INFINITY;
    let mut thickness = 0.0;

    if let Some(intersection) = intersection {
        distance = intersection.distance;
        let direction = (intersection.position - *origin).unit_vector();
        let ray = Ray::new(intersection.position + direction * epsilon, direction);

        let next_intersection = mesh.intersect(&ray, IntersectionSort::Nearest, epsilon);
        if let Some(next_intersection) = next_intersection {
            thickness = (intersection.position - next_intersection.position).length();
        }

        if check_radially {
            let radial_rays: Vec<Ray> = (1..=radial_rings)
                .flat_map(|ring| {
                    (0..radial_samples).map(move |sample| {
                        let offset = (ring as f64 / radial_rings as f64) * radius;
                        let angle =
                            2.0 * std::f64::consts::PI * sample as f64 / radial_samples as f64;

                        let matrix = Mat4::rotation(direction, angle);
                        let perpendicular =
                            direction.cross(&Vec3::new(0.0, 0.0, 1.0)).unit_vector() * matrix;
                        let offset_origin = *origin + perpendicular * offset;

                        Ray::new(offset_origin, direction)
                    })
                })
                .collect();

            let radial_intersections: Vec<(f64, f64)> = radial_rays
                .par_iter()
                .map(|ray| {
                    let radial_intersection =
                        mesh.intersect(ray, IntersectionSort::Nearest, epsilon);
                    process_intersection(
                        mesh,
                        origin,
                        &radial_intersection,
                        false,
                        radius / radial_rings as f64,
                        radial_samples,
                        radial_rings,
                        epsilon,
                    )
                })
                .collect();

            let max_distance = radial_intersections
                .iter()
                .map(|(distance, _)| distance)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let min_thickness = radial_intersections
                .iter()
                .map(|(_, thickness)| thickness)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            distance = distance.max(*max_distance);
            thickness = thickness.min(*min_thickness);
        }
    }

    (distance, thickness)
}

#[pyfunction]
#[pyo3(signature = (mesh, origin, n_steps=128, n_iter=3, initial_fov=45.0, check_radially=false, radius=4.0, radial_samples=8, radial_rings=2, objective_distance_weight=0.5, thickness_threshold=10., depth_threshold=70.0, epsilon=1e-8))]
pub fn find_target(
    mesh: &Mesh,
    origin: &Vec3,
    n_steps: usize,
    n_iter: usize,
    initial_fov: f64,
    check_radially: bool,
    radius: f64,
    radial_samples: usize,
    radial_rings: usize,
    objective_distance_weight: f64,
    thickness_threshold: f64,
    depth_threshold: f64,
    epsilon: f64,
) -> (Vec3, f64) {
    let mut min_loss = f64::INFINITY;
    let mut best_target = Vec3::zero();
    let mut best_forward = (mesh.origin - *origin).unit_vector();

    let mut current_fov = initial_fov;

    for iter in 0..n_iter {
        let camera = Camera::new(
            *origin,
            best_forward,
            Vec3::new(0.0, 0.0, 1.0),
            n_steps,
            n_steps,
            CameraType::Perspective,
            Some(current_fov),
            None,
            None,
        );
        let renderer = Renderer::new(camera, mesh.clone());

        let render = renderer.render_internal(IntersectionSort::Nearest, epsilon);

        let iter_radius = radius * (iter as f64 / (n_iter - 1) as f64);
        let best_intersection: Option<&((usize, usize), Option<Intersection>)> =
            render.par_iter().min_by(|(_, intersection), (_, other)| {
                let (distance, thickness) = process_intersection(
                    mesh,
                    origin,
                    intersection,
                    check_radially && iter > 0,
                    iter_radius,
                    radial_samples,
                    radial_rings,
                    epsilon,
                );
                let (other_distance, other_thickness) = process_intersection(
                    mesh,
                    origin,
                    other,
                    check_radially && iter > 0,
                    iter_radius,
                    radial_samples,
                    radial_rings,
                    epsilon,
                );

                let loss = objective_function(
                    distance,
                    thickness,
                    objective_distance_weight,
                    thickness_threshold,
                    depth_threshold,
                );

                let other_loss = objective_function(
                    other_distance,
                    other_thickness,
                    objective_distance_weight,
                    thickness_threshold,
                    depth_threshold,
                );

                loss.partial_cmp(&other_loss).unwrap()
            });

        if let Some(((y, x), _)) = best_intersection {
            let best_ray = camera.cast_ray(*x as f64, *y as f64);

            let (midpoint, distance, thickness, valid) = sample_ray(mesh, &best_ray, epsilon);

            if valid {
                let loss = objective_function(
                    distance,
                    thickness,
                    objective_distance_weight,
                    thickness_threshold,
                    depth_threshold,
                );

                if loss < min_loss {
                    min_loss = loss;
                    best_target = midpoint;
                    best_forward = (midpoint - *origin).unit_vector();
                }
            }
        } else {
            panic!("No valid intersections found at iteration {}!", iter);
        }

        current_fov *= 0.25;
    }

    (best_target, min_loss)
}

#[pyfunction]
#[pyo3(signature = (mesh, camera, check_radially=false, radius=4.0, radial_samples=8, radial_rings=2, objective_distance_weight=0.5, thickness_threshold=10., depth_threshold=70.0, epsilon=1e-8))]
pub fn generate_objective_image(
    py: Python<'_>,
    mesh: &Mesh,
    camera: &Camera,
    check_radially: bool,
    radius: f64,
    radial_samples: usize,
    radial_rings: usize,
    objective_distance_weight: f64,
    thickness_threshold: f64,
    depth_threshold: f64,
    epsilon: f64,
) -> Py<PyArray<f64, Ix3>> {
    let renderer = Renderer::new(*camera, mesh.clone());
    let values = renderer.render_internal(IntersectionSort::Nearest, epsilon);

    let mut image = Array3::<f64>::zeros((camera.y_resolution, camera.x_resolution, 3));

    values.into_iter().for_each(|((y, x), intersection)| {
        let (distance, thickness) = process_intersection(
            mesh,
            &camera.origin,
            &intersection,
            check_radially,
            radius,
            radial_samples,
            radial_rings,
            epsilon,
        );

        let loss = objective_function(
            distance,
            thickness,
            objective_distance_weight,
            thickness_threshold,
            depth_threshold,
        );

        image[[y, x, 0]] = loss;
        image[[y, x, 1]] = distance;
        image[[y, x, 2]] = thickness;
    });

    image.into_pyarray(py).to_owned()
}
