use crate::geometry::Intersectable;
use crate::rendering::{Intersection, Ray};

pub struct Scene {
    pub objects: Vec<dyn Intersectable>,
}

impl Scene {
    pub fn new() -> Self {
        Self { objects: Vec::new() }
    }
    
    pub fn add_object(&mut self, object: impl Intersectable + 'static) {
        self.objects.push(Box::new(object));
    }
}

impl Intersectable for Scene {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        let mut closest_intersection = None;
        for object in &self.objects {
            if let Some(intersection) = object.intersect(ray) {
                if let Some(closest) = closest_intersection {
                    if intersection.distance < closest.distance {
                        closest_intersection = Some(intersection);
                    }
                } else {
                    closest_intersection = Some(intersection);
                }
            }
        }
        closest_intersection
    }
}