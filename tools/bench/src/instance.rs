//! Recorded, seeded affine benchmark instances (not COCO/BBOB instances).
use crate::catalog::Problem;
use nlopt4r::{
    Matrix,
    utils::{ChaCha8Rng, Rng, SeedableRng},
};
use serde::{Deserialize, Serialize};

/// Transformation of solver coordinates: `z = reference + Q*(x-reference-shift)`.
///
/// The outer catalog box stays fixed. The recorded reference point plus `shift`
/// is a feasible minimizer with the original reference value. Rotations use a
/// proper orthogonal matrix; the base formula can be evaluated outside its
/// original nominal box.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transformation {
    /// Affine generator/representation version, currently 1.
    pub version: u32,
    /// Independent of optimizer seeds and initial-point RNG state.
    pub seed: u64,
    /// Whether the feasible reference was randomly translated.
    pub shift_enabled: bool,
    /// Whether `rotation` supplies a proper orthogonal coordinate change.
    pub rotation_enabled: bool,
    /// Translation relative to the base problem's certified reference point.
    pub shift: Vec<f64>,
    /// Row-major matrix; empty when rotation is disabled.
    pub rotation: Vec<Vec<f64>>,
}

impl Transformation {
    /// Generate one problem/dimension instance, independent of solver RNG state.
    ///
    /// Returns `None` for both switches disabled. Shifted optima are sampled
    /// inside the central 60% of the original box; rotations are Gaussian-QR
    /// orthogonal bases corrected to determinant +1. The dimension must match
    /// the catalog entry. Version and exact coefficients are serializable.
    pub fn new(
        problem: &Problem,
        dimension: usize,
        shift_enabled: bool,
        rotation_enabled: bool,
        seed: u64,
    ) -> Result<Option<Self>, String> {
        if problem.dimension(dimension)? != dimension {
            return Err("instance dimension does not match the problem".into());
        }
        if !shift_enabled && !rotation_enabled {
            return Ok(None);
        }
        // Stable FNV-1a problem identity; do not use process-randomized hashers.
        let mut identity = 0xcbf29ce484222325u64;
        for byte in problem.id.bytes().chain((dimension as u64).to_le_bytes()) {
            identity = (identity ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
        // Independent streams keep the shift identical when toggling rotation.
        let mut shift_rng = ChaCha8Rng::seed_from_u64(seed ^ identity);
        let mut rotation_rng = ChaCha8Rng::seed_from_u64(seed ^ identity ^ 0xa0761d6478bd642f);
        let (reference, _) = problem.reference(dimension);
        let shift = reference
            .iter()
            .map(|&r| {
                if shift_enabled {
                    let fraction = 0.2 + 0.6 * shift_rng.random::<f64>();
                    problem.bounds.0 + fraction * (problem.bounds.1 - problem.bounds.0) - r
                } else {
                    0.0
                }
            })
            .collect();
        let rotation = if rotation_enabled {
            // Gaussian QR with diagonal sign correction gives an isotropic
            // orthogonal basis; fix the final sign to obtain determinant +1.
            let gaussian = Matrix::from_fn(dimension, dimension, |_, _| {
                let u = 1.0 - rotation_rng.random::<f64>();
                let angle = std::f64::consts::TAU * rotation_rng.random::<f64>();
                (-2.0 * u.ln()).sqrt() * angle.cos()
            });
            let (mut q, r) = gaussian.qr().unpack();
            for j in 0..dimension {
                if r[(j, j)] < 0.0 {
                    q.column_mut(j).scale_mut(-1.0);
                }
            }
            if q.determinant() < 0.0 {
                q.column_mut(dimension - 1).scale_mut(-1.0);
            }
            (0..dimension)
                .map(|i| (0..dimension).map(|j| q[(i, j)]).collect())
                .collect()
        } else {
            Vec::new()
        };
        let result = Self {
            version: 1,
            seed,
            shift_enabled,
            rotation_enabled,
            shift,
            rotation,
        };
        result.validate(problem, dimension)?;
        Ok(Some(result))
    }

    /// Feasible transformed reference `base_reference + shift`.
    /// Use the same problem/dimension this validated transformation describes.
    pub fn reference_point(&self, problem: &Problem) -> Vec<f64> {
        problem
            .reference(self.shift.len())
            .0
            .iter()
            .zip(&self.shift)
            .map(|(r, s)| r + s)
            .collect()
    }

    /// Map solver coordinates to base-formula coordinates without clamping.
    /// Input must have the validated instance dimension. This may leave the
    /// original nominal box; the outer box is a bound on solver coordinates only.
    pub fn to_base(&self, problem: &Problem, x: &[f64]) -> Vec<f64> {
        if !self.rotation_enabled {
            return x.iter().zip(&self.shift).map(|(x, s)| x - s).collect();
        }
        let reference = problem.reference(x.len()).0;
        let displacement: Vec<_> = x
            .iter()
            .zip(&reference)
            .zip(&self.shift)
            .map(|((x, r), s)| x - (r + s))
            .collect();
        self.rotation
            .iter()
            .zip(reference)
            .map(|(row, r)| {
                r + row
                    .iter()
                    .zip(&displacement)
                    .map(|(q, d)| q * d)
                    .sum::<f64>()
            })
            .collect()
    }

    /// Chain rule `gradient_x = Q' * gradient_z`.
    pub fn gradient(&self, base_gradient: &[f64]) -> Vec<f64> {
        if !self.rotation_enabled {
            return base_gradient.to_vec();
        }
        (0..base_gradient.len())
            .map(|j| {
                self.rotation
                    .iter()
                    .zip(base_gradient)
                    .map(|(row, g)| row[j] * g)
                    .sum()
            })
            .collect()
    }

    /// Check version, dimensions, finite coefficients, reference feasibility and
    /// orthogonality/determinant (tolerance 1e-10). Recorded matrices are
    /// authoritative for replay; validation does not regenerate them from the seed.
    pub fn validate(&self, problem: &Problem, n: usize) -> Result<(), String> {
        if problem.dimension(n)? != n {
            return Err("instance dimension does not match the problem".into());
        }
        if self.version != 1
            || (!self.shift_enabled && !self.rotation_enabled)
            || self.shift.len() != n
            || self.shift.iter().any(|v| !v.is_finite())
            || (!self.shift_enabled && self.shift.iter().any(|s| *s != 0.0))
            || self
                .reference_point(problem)
                .iter()
                .any(|x| !x.is_finite() || *x < problem.bounds.0 || *x > problem.bounds.1)
        {
            return Err("invalid instance version, shift or reference feasibility".into());
        }
        if !self.rotation_enabled {
            return if self.rotation.is_empty() {
                Ok(())
            } else {
                Err("disabled rotation must have an empty matrix".into())
            };
        }
        if self.rotation.len() != n
            || self
                .rotation
                .iter()
                .any(|row| row.len() != n || row.iter().any(|v| !v.is_finite()))
        {
            return Err("invalid instance rotation dimensions or entries".into());
        }
        let q = Matrix::from_fn(n, n, |i, j| self.rotation[i][j]);
        if (q.transpose() * &q - Matrix::identity(n, n)).norm() > 1e-10
            || (q.determinant() - 1.0).abs() > 1e-10
        {
            return Err("instance rotation must be orthogonal with determinant +1".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::CATALOG;

    #[test]
    fn transformations_preserve_references_and_have_independent_repeatable_streams() {
        for p in CATALOG {
            let n = p.dimension(5).unwrap();
            for (shift, rotate) in [(true, false), (false, true), (true, true)] {
                let t = Transformation::new(p, n, shift, rotate, 17)
                    .unwrap()
                    .unwrap();
                assert_eq!(
                    Some(t.clone()),
                    Transformation::new(p, n, shift, rotate, 17).unwrap()
                );
                let x = t.reference_point(p);
                assert!(x.iter().all(|x| *x >= p.bounds.0 && *x <= p.bounds.1));
                let z = t.to_base(p, &x);
                let reference = p.reference(n);
                for (actual, expected) in z.iter().zip(&reference.0) {
                    assert!((actual - expected).abs() < 1e-10, "{}", p.id);
                }
                assert!((p.value(&z) - reference.1).abs() < 1e-7, "{}", p.id);
                assert!(p.violation(&z) < 1e-10, "{}", p.id);
            }
            let shifted = Transformation::new(p, n, true, false, 17).unwrap().unwrap();
            let both = Transformation::new(p, n, true, true, 17).unwrap().unwrap();
            assert_eq!(shifted.shift, both.shift);
            let other = Transformation::new(p, n, true, true, 18).unwrap().unwrap();
            assert_ne!(both.shift, other.shift);
            if n > 1 {
                assert_ne!(both.rotation, other.rotation);
            }
        }
    }

    #[test]
    fn rotations_preserve_lengths_including_one_and_large_dimensions() {
        let p = &CATALOG[0];
        for n in [1, 20, 128] {
            let t = Transformation::new(p, n, false, true, 17).unwrap().unwrap();
            let x: Vec<_> = (0..n).map(|i| (i as f64 + 0.3).sin()).collect();
            let z = t.to_base(p, &x);
            let length = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
            assert!((length(&x) - length(&z)).abs() < 1e-10);
            if n == 1 {
                assert_eq!(t.rotation, vec![vec![1.0]]);
            }
        }
    }

    #[test]
    fn rotation_changes_orientation_and_chain_rule_matches_independent_differences() {
        let p = CATALOG.iter().find(|p| p.id == "ellipsoid").unwrap();
        let t = Transformation::new(p, 4, true, true, 2).unwrap().unwrap();
        let x = [0.23, -0.17, 0.35, 0.41];
        let g = t.gradient(&p.gradient(&t.to_base(p, &x)));
        for i in 0..4 {
            let mut plus = x;
            let mut minus = x;
            plus[i] += 1e-5;
            minus[i] -= 1e-5;
            let numeric = (p.value(&t.to_base(p, &plus)) - p.value(&t.to_base(p, &minus))) / 2e-5;
            assert!((g[i] - numeric).abs() < 1e-6 * (1.0 + g[i].abs()));
        }
        // For diagonal H, a nonzero mixed difference proves a non-axis-aligned landscape.
        let origin = t.reference_point(p);
        let mut a = origin.clone();
        let mut b = origin.clone();
        let mut ab = origin.clone();
        a[0] += 0.1;
        b[1] += 0.1;
        ab[0] += 0.1;
        ab[1] += 0.1;
        let f = |x: &[f64]| p.value(&t.to_base(p, x));
        assert!((f(&ab) - f(&a) - f(&b) + f(&origin)).abs() > 1.0);
        let mut bad = t;
        bad.rotation[0][0] += 0.1;
        assert!(bad.validate(p, 4).is_err());
    }
}
