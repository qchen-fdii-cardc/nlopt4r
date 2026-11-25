//! Validated problem domains and compatibility exports.
pub use crate::constrained::{ConstrainedProblem, Constraints};
pub use crate::least_squares::LeastSquares;
use crate::{Error, Vector};
pub use crate::{Evaluator, Function, Objective};
use rand::Rng;

/// Finite coordinate bounds. Equal endpoints represent fixed coordinates.
///
/// Bounds are enforced only by solvers accepting them explicitly. They do not
/// change an unconstrained objective or make it safe outside its domain.
///
/// ```
/// use nlopt4r::Bounds;
/// let bounds = Bounds::new(vec![-2.0, 3.0], vec![2.0, 3.0])?;
/// let mut x = [5.0, -1.0];
/// bounds.project(&mut x)?;
/// assert_eq!(x, [2.0, 3.0]);
/// assert_eq!(bounds.to_unit(&x)?[1], 0.0); // Fixed coordinate.
/// # Ok::<(), nlopt4r::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Bounds {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Bounds {
    /// Validate equally sized nonempty endpoints: finite, ordered, with finite widths.
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
    /// Repeat one validated closed interval across `n > 0` coordinates.
    pub fn uniform(n: usize, lower: f64, upper: f64) -> Result<Self, Error> {
        Self::new(vec![lower; n], vec![upper; n])
    }
    /// Number of coordinates, always positive.
    pub fn dimension(&self) -> usize {
        self.lower.len()
    }
    /// Inclusive lower endpoints.
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }
    /// Inclusive upper endpoints.
    pub fn upper(&self) -> &[f64] {
        &self.upper
    }
    /// Coordinate width, zero for a fixed coordinate.
    ///
    /// # Panics
    /// Panics if `i >= self.dimension()`.
    pub fn width(&self, i: usize) -> f64 {
        self.upper[i] - self.lower[i]
    }
    /// Reject wrong dimensions, non-finite entries or points outside the closed box.
    pub fn validate_point(&self, x: &[f64]) -> Result<(), Error> {
        crate::utils::validate_slice(x, self.dimension(), "bounded point")?;
        if !self.contains(x) {
            return Err(Error::InvalidInput("point lies outside bounds"));
        }
        Ok(())
    }
    /// Whether all entries are finite, dimension-correct and within inclusive endpoints.
    /// Invalid input returns `false`; use [`Self::validate_point`] for an error.
    pub fn contains(&self, x: &[f64]) -> bool {
        x.len() == self.dimension()
            && x.iter()
                .enumerate()
                .all(|(i, v)| v.is_finite() && *v >= self.lower[i] && *v <= self.upper[i])
    }
    /// Clamp a finite dimension-correct point in place. NaN/infinity are errors,
    /// not silently replaced by bounds.
    pub fn project(&self, x: &mut [f64]) -> Result<(), Error> {
        crate::utils::validate_slice(x, self.dimension(), "projection")?;
        for (i, v) in x.iter_mut().enumerate() {
            *v = v.clamp(self.lower[i], self.upper[i]);
        }
        Ok(())
    }
    /// Fill a dimension-correct buffer with uniform samples; fixed variables stay fixed.
    /// Pass a seeded RNG (for example [`crate::utils::ChaCha8Rng`]) for repeatable runs.
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
    /// Center of each interval, computed without summing potentially large endpoints.
    pub fn midpoint(&self) -> Vector {
        Vector::from_iterator(
            self.dimension(),
            (0..self.dimension()).map(|i| self.lower[i] + 0.5 * self.width(i)),
        )
    }
    /// Map normalized coordinates in `[0,1]` to the domain.
    /// Reject wrong dimensions or out-of-range/non-finite inputs.
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
    /// Normalize an in-bounds point to `[0,1]`; fixed coordinates map to zero.
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
