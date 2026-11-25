//! Validated problem domains and compatibility exports.
pub use crate::constrained::{ConstrainedProblem, Constraints};
pub use crate::least_squares::LeastSquares;
use crate::{Error, Vector};
pub use crate::{Evaluator, Function, Objective};
use rand::Rng;

/// Finite coordinate bounds. Equal endpoints represent fixed coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Bounds {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Bounds {
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Result<Self, Error> {
        if lower.is_empty() || lower.len() != upper.len() {
            return Err(Error::InvalidInput(
                "bounds must have equal nonzero dimensions",
            ));
        }
        for (&lo, &hi) in lower.iter().zip(&upper) {
            if !lo.is_finite() || !hi.is_finite() || lo > hi || !(hi - lo).is_finite() {
                return Err(Error::InvalidInput(
                    "bounds must be finite, ordered and have finite widths",
                ));
            }
        }
        Ok(Self { lower, upper })
    }
    pub fn uniform(n: usize, lower: f64, upper: f64) -> Result<Self, Error> {
        Self::new(vec![lower; n], vec![upper; n])
    }
    pub fn dimension(&self) -> usize {
        self.lower.len()
    }
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }
    pub fn upper(&self) -> &[f64] {
        &self.upper
    }
    pub fn width(&self, i: usize) -> f64 {
        self.upper[i] - self.lower[i]
    }
    pub fn validate_point(&self, x: &[f64]) -> Result<(), Error> {
        crate::utils::validate_slice(x, self.dimension(), "bounded point")?;
        if !self.contains(x) {
            return Err(Error::InvalidInput("point lies outside bounds"));
        }
        Ok(())
    }
    pub fn contains(&self, x: &[f64]) -> bool {
        x.len() == self.dimension()
            && x.iter()
                .enumerate()
                .all(|(i, v)| v.is_finite() && *v >= self.lower[i] && *v <= self.upper[i])
    }
    pub fn project(&self, x: &mut [f64]) -> Result<(), Error> {
        crate::utils::validate_slice(x, self.dimension(), "projection")?;
        for (i, v) in x.iter_mut().enumerate() {
            *v = v.clamp(self.lower[i], self.upper[i]);
        }
        Ok(())
    }
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R, x: &mut [f64]) -> Result<(), Error> {
        if x.len() != self.dimension() {
            return Err(Error::DimensionMismatch {
                context: "sample buffer",
                expected: (self.dimension(), 1),
                actual: (x.len(), 1),
            });
        }
        for (i, v) in x.iter_mut().enumerate() {
            *v = self.lower[i] + rng.random::<f64>() * self.width(i);
        }
        self.project(x)
    }
    pub fn midpoint(&self) -> Vector {
        Vector::from_iterator(
            self.dimension(),
            (0..self.dimension()).map(|i| self.lower[i] + 0.5 * self.width(i)),
        )
    }
    /// Map normalized coordinates in [0,1] to the domain.
    pub fn from_unit(&self, unit: &[f64]) -> Result<Vector, Error> {
        crate::utils::validate_slice(unit, self.dimension(), "normalized coordinates")?;
        if unit.iter().any(|v| !(0.0..=1.0).contains(v)) {
            return Err(Error::InvalidInput(
                "normalized coordinates must lie in [0,1]",
            ));
        }
        Ok(Vector::from_iterator(
            self.dimension(),
            unit.iter()
                .enumerate()
                .map(|(i, v)| self.lower[i] + v * self.width(i)),
        ))
    }
    pub fn to_unit(&self, x: &[f64]) -> Result<Vector, Error> {
        self.validate_point(x)?;
        Ok(Vector::from_iterator(
            self.dimension(),
            x.iter().enumerate().map(|(i, v)| {
                if self.width(i) == 0.0 {
                    0.0
                } else {
                    (v - self.lower[i]) / self.width(i)
                }
            }),
        ))
    }
}
