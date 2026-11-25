//! Feasible projected-gradient descent for finite coordinate bounds.
use crate::{
    Bounds, Error, Evaluator, Options, Solution, TerminationReason, Vector, initial, line_search,
    stopping::{Run, snapshot},
};

/// Euclidean norm of the unit-step gradient mapping `x - P(x-gradient)`.
/// It is zero exactly at first-order box stationarity, including fixed variables.
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
