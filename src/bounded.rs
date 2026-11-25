//! Feasible projected-gradient descent for finite coordinate bounds.
//!
//! The unit-step mapping `G(x)=x-project(x-gradient)` vanishes at first-order
//! box stationarity even when the unconstrained gradient does not. Every trial
//! remains feasible; the initial finite point is projected before evaluation.
//! This is a first-order local method, not full L-BFGS-B.
//!
//! ```
//! use nlopt4r::{Bounds, Options, Vector, bounded::projected_gradient};
//! let problem = (
//!     |x: &Vector| (x[0]-3.0).powi(2),
//!     |x: &Vector| Vector::from_vec(vec![2.0*(x[0]-3.0)]),
//! );
//! let result = projected_gradient(
//!     &problem, &Vector::zeros(1), &Bounds::uniform(1, -1.0, 1.0)?, &Options::default(),
//! )?;
//! assert!(result.converged());
//! assert!((result.x[0]-1.0).abs() < 1e-6); // Active-bound optimum.
//! # Ok::<(), nlopt4r::Error>(())
//! ```
use crate::{
    Bounds, Error, Evaluator, Options, Solution, TerminationReason, Vector, initial, line_search,
    stopping::{Run, snapshot},
};

/// Euclidean norm of the unit-step gradient mapping `x - P(x-gradient)`.
/// It is zero exactly at first-order box stationarity, including fixed variables.
/// Requires a finite dimension-correct gradient and an already in-bounds point.
pub fn projected_gradient_norm(
    x: &Vector,
    gradient: &Vector,
    bounds: &Bounds,
) -> Result<f64, Error> {
    bounds.validate_point(x.as_slice())?;
    crate::vector(gradient, x.len(), "bounded gradient")?;
    let projected = projected_point(x, gradient, bounds)?;
    crate::finite((x - projected).norm(), "projected gradient norm")
}

fn projected_point(x: &Vector, gradient: &Vector, bounds: &Bounds) -> Result<Vector, Error> {
    // Avoid overflow in x-g before clamping, including very large finite gradients.
    let mut point = x.clone();
    for i in 0..point.len() {
        point[i] = (x[i] - gradient[i]).clamp(bounds.lower()[i], bounds.upper()[i]);
    }
    bounds.validate_point(point.as_slice())?;
    Ok(point)
}

/// Project the initial point, then backtrack along a feasible projected direction.
///
/// Every callback receives a point in the box. `Solution::gradient_norm` is
/// the unit-step gradient mapping norm, not the unconstrained gradient norm.
/// This is projected gradient, not L-BFGS-B.
///
/// Uses the gradient tolerance from [`Options::stopping`] and Armijo parameters
/// from [`Options::line_search`]. Accepted value/gradient pairs are reused.
/// Budget/stagnation reports are not certified minima. Shape, callback and
/// numerical errors retain the last fully evaluated bounded checkpoint.
pub fn projected_gradient<P: Evaluator + ?Sized>(
    problem: &P,
    x0: &Vector,
    bounds: &Bounds,
    options: &Options,
) -> Result<Solution, Error> {
    options.validate()?;
    initial(x0, &options.stopping)?;
    let mut x = x0.clone();
    bounds.project(x.as_mut_slice())?;
    let mut run = Run::new(&options.stopping)?;
    let result = (|| {
        let mut cached = None;
        for k in 0..=options.stopping.max_iterations {
            let (value, gradient) = match cached.take() {
                Some(pair) => pair,
                None => run.evaluate(problem, &x)?,
            };
            let projected = projected_point(&x, &gradient, bounds)?;
            let d = projected - &x;
            let norm = crate::finite(d.norm(), "projected gradient norm")?;
            let reason = run.reason(
                &x,
                value,
                k,
                norm <= options.stopping.gradient_tolerance,
                TerminationReason::GradientTolerance,
                k > 0,
            );
            let current = run.record(snapshot(x.clone(), value, norm, k, reason));
            if reason.is_some() {
                return Ok(current);
            }
            let mut accepted = None;
            line_search::backtrack(value, gradient.dot(&d), &options.line_search, |alpha| {
                let mut trial = &x + alpha * &d;
                bounds.project(trial.as_mut_slice())?;
                let pair = run.evaluate(problem, &trial)?;
                let value = pair.0;
                accepted = Some((trial, pair));
                Ok(value)
            })?;
            let (trial, pair) =
                accepted.ok_or(Error::InvalidInput("empty projected line search"))?;
            x = trial;
            cached = Some(pair);
        }
        unreachable!()
    })();
    run.finish(result, "projected gradient")
}
