use noise::{NoiseFn, Seedable, Simplex};
use pyo3::prelude::*;

use crate::linalg::Vec3;

#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct Deformer {
    scale: f64,
    amplitude: f64,
    frequency: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,

    generator: Simplex,
}

#[pymethods]
impl Deformer {
    #[new]
    pub fn new(
        scale: f64,
        amplitude: f64,
        frequency: f64,
        octaves: usize,
        persistence: f64,
        lacunarity: f64,
        seed: u32,
    ) -> Self {
        Deformer {
            scale,
            amplitude,
            frequency,
            octaves,
            persistence,
            lacunarity,
            generator: Simplex::new(seed),
        }
    }

    pub fn deform_vertex(&self, v: &Vec3) -> Vec3 {
        *v + self.generate_noise(v) * self.scale
    }
}

impl Deformer {
    fn generate_noise(&self, v: &Vec3) -> Vec3 {
        let mut delta = Vec3::zero();
        let mut current_amplitude = self.amplitude;
        let mut current_frequency = self.frequency;

        let original_seed = self.generator.seed();
        for i in 0..self.octaves {
            delta.x += current_amplitude
                * self.generator.get([
                    v.x * current_frequency,
                    v.y * current_frequency,
                    v.z * current_frequency,
                ]);
            self.generator.set_seed(original_seed + 1u32);
            delta.y += current_amplitude
                * self.generator.get([
                    v.x * current_frequency,
                    v.y * current_frequency,
                    v.z * current_frequency,
                ]);
            self.generator.set_seed(original_seed + 2u32);
            delta.z += current_amplitude
                * self.generator.get([
                    v.x * current_frequency,
                    v.y * current_frequency,
                    v.z * current_frequency,
                ]);

            self.generator.set_seed(original_seed + i as u32 + 3u32);

            current_amplitude *= self.persistence;
            current_frequency *= self.lacunarity;
        }

        self.generator.set_seed(original_seed);
        delta
    }
}
