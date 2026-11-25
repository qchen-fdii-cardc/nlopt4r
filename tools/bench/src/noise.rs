//! Independent, reproducible, additive observation noise for benchmark callbacks.
use crate::instance::problem_identity;
use nlopt4r::{
    Error,
    utils::{ChaCha8Rng, Rng, SeedableRng},
};
use serde::{Deserialize, Serialize};

/// All enabled distributions have mean zero and unit variance before scaling.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Distribution {
    #[default]
    None,
    Gaussian,
    Uniform,
    Laplace,
}

/// Noise strength is an absolute standard deviation in objective-value units.
/// Version 1 uses ChaCha8 from rand_chacha 0.9 and the sampling formulas below.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoiseOptions {
    pub version: u32,
    pub distribution: Distribution,
    pub standard_deviation: f64,
    /// Experiment seed, independent of instance and optimizer RNGs.
    pub seed: u64,
}

impl Default for NoiseOptions {
    fn default() -> Self {
        Self {
            version: 1,
            distribution: Distribution::None,
            standard_deviation: 0.0,
            seed: 0,
        }
    }
}

impl NoiseOptions {
    pub fn validate(&self) -> Result<(), String> {
        if self.version != 1
            || !self.standard_deviation.is_finite()
            || self.standard_deviation < 0.0
            || (self.distribution == Distribution::None && self.standard_deviation != 0.0)
            || (self.distribution != Distribution::None && self.standard_deviation == 0.0)
        {
            return Err(
                "noise requires version 1, a positive finite level when enabled, and zero level for none"
                    .into(),
            );
        }
        Ok(())
    }

    pub fn enabled(&self) -> bool {
        self.distribution != Distribution::None
    }

    /// Reset per problem/dimension/run, but not per algorithm, level or transform.
    /// Matching evaluation indices share draws, not necessarily matching points.
    pub fn stream_seed(&self, problem: &str, dimension: usize, run_seed: u64) -> u64 {
        let mut mixer = ChaCha8Rng::seed_from_u64(
            self.seed ^ problem_identity(problem, dimension) ^ 0xe7037ed1a0b428db,
        );
        mixer.random::<u64>() ^ run_seed
    }
}

/// Mutable stream owned by one run; never shared with solver/instance RNGs.
pub struct NoiseSampler {
    options: NoiseOptions,
    rng: ChaCha8Rng,
    pub draws: usize,
}

impl NoiseSampler {
    pub fn new(options: &NoiseOptions, stream_seed: u64) -> Result<Self, String> {
        options.validate()?;
        if !options.enabled() {
            return Err("cannot construct a sampler for disabled noise".into());
        }
        Ok(Self {
            options: options.clone(),
            rng: ChaCha8Rng::seed_from_u64(stream_seed),
            draws: 0,
        })
    }

    /// One fresh draw per valid base-objective call, including finite-difference probes.
    /// No clipping, retries, averaging or outlier removal.
    pub fn perturbation(&mut self) -> Result<f64, Error> {
        self.draws += 1;
        let u = 1.0 - self.rng.random::<f64>();
        let standardized = match self.options.distribution {
            Distribution::Gaussian => {
                let angle = std::f64::consts::TAU * self.rng.random::<f64>();
                (-2.0 * u.ln()).sqrt() * angle.cos()
            }
            Distribution::Uniform => 3.0_f64.sqrt() * (2.0 * u - 1.0),
            Distribution::Laplace => {
                let v = 1.0 - self.rng.random::<f64>();
                (u.ln() - v.ln()) / 2.0_f64.sqrt()
            }
            Distribution::None => unreachable!("constructor rejects disabled noise"),
        };
        let perturbation = self.options.standard_deviation * standardized;
        if !perturbation.is_finite() {
            return Err(Error::NonFinite("benchmark noise"));
        }
        Ok(perturbation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_distributions_have_expected_scale_and_replay() {
        for distribution in [
            Distribution::Gaussian,
            Distribution::Uniform,
            Distribution::Laplace,
        ] {
            let options = NoiseOptions {
                distribution,
                standard_deviation: 2.0,
                ..Default::default()
            };
            let seed = options.stream_seed("sphere", 3, 7);
            let mut a = NoiseSampler::new(&options, seed).unwrap();
            let mut b = NoiseSampler::new(&options, seed).unwrap();
            let mut scaled = options.clone();
            scaled.standard_deviation = 4.0;
            let mut c = NoiseSampler::new(&scaled, seed).unwrap();
            let mut sum = 0.0;
            let mut squares = 0.0;
            for _ in 0..50_000 {
                let x = a.perturbation().unwrap();
                assert_eq!(x, b.perturbation().unwrap());
                assert_eq!(2.0 * x, c.perturbation().unwrap());
                if distribution == Distribution::Uniform {
                    assert!(x.abs() <= 2.0 * 3.0_f64.sqrt());
                }
                sum += x;
                squares += x * x;
            }
            assert!(sum.abs() / 50_000.0 < 0.04, "{distribution:?}");
            assert!((squares / 50_000.0 - 4.0).abs() < 0.15, "{distribution:?}");
            let other = options.stream_seed("sphere", 3, 8);
            assert_ne!(seed, other);
            assert_ne!(
                NoiseSampler::new(&options, seed)
                    .unwrap()
                    .perturbation()
                    .unwrap(),
                NoiseSampler::new(&options, other)
                    .unwrap()
                    .perturbation()
                    .unwrap()
            );
        }
    }

    #[test]
    fn invalid_noise_and_overflow_are_explicit() {
        for level in [-1.0, f64::NAN, f64::INFINITY, 0.0] {
            assert!(
                NoiseOptions {
                    distribution: Distribution::Gaussian,
                    standard_deviation: level,
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            NoiseOptions {
                standard_deviation: 1.0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        let mut sampler = NoiseSampler::new(
            &NoiseOptions {
                distribution: Distribution::Uniform,
                standard_deviation: f64::MAX,
                ..Default::default()
            },
            42,
        )
        .unwrap();
        assert!((0..100).any(|_| sampler.perturbation().is_err()));
    }
}
