//! Bounded coordinate pattern search and Nelder-Mead simplex search.
//!
//! Both methods minimize a finite value-only objective over [`Bounds`], starting
//! from an already feasible `x0`. Trials are projected; fixed coordinates do not
//! participate in polling/simplex construction. Neither computes derivatives or
//! turns mesh collapse into an optimality claim.
//!
//! ```
//! use nlopt4r::{Bounds, Vector, value_only, derivative_free::{nelder_mead, Options}};
//! let problem = value_only(|x: &[f64]| Ok((x[0] - 0.25).powi(2)));
//! let bounds = Bounds::uniform(1, -1.0, 1.0)?;
//! let mut options = Options::default();
//! options.budget.target_value = Some(1e-8);
//! let result = nelder_mead(&problem, &Vector::zeros(1), &bounds, &options)?;
//! assert!(result.value <= 1e-8);
//! # Ok::<(), nlopt4r::Error>(())
//! ```
use crate::core::{BudgetOptions, EvaluationContext, SearchTermination};
use crate::{Bounds, Error, EvaluationCounts, Evaluator, Vector};

/// Shared local derivative-free settings; distinct from gradient-based [`crate::Options`].
#[derive(Debug, Clone)]
pub struct Options {
    /// Positive objective allowance and optional target.
    pub budget: BudgetOptions,
    /// Poll/simplex iteration limit, default 2000.
    pub max_iterations: usize,
    /// Initial step as a fraction of coordinate bound width.
    /// Must be finite in `(0,1]`; default 0.1.
    pub initial_step: f64,
    /// Normalized mesh/simplex diameter threshold; not an optimality certificate.
    /// Positive and finite, default 1e-8.
    pub step_tolerance: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            budget: BudgetOptions::default(),
            max_iterations: 2000,
            initial_step: 0.1,
            step_tolerance: 1e-8,
        }
    }
}
impl Options {
    /// Validate the budget and normalized step/tolerance before evaluation.
    pub fn validate(&self) -> Result<(), Error> {
        self.budget.validate()?;
        if !self.initial_step.is_finite()
            || !(0.0..=1.0).contains(&self.initial_step)
            || self.initial_step == 0.0
        {
            return Err(Error::InvalidInput(
                "initial normalized step must be in (0,1]",
            ));
        }
        crate::positive(self.step_tolerance, "mesh tolerance")
    }
}
/// Best checked value-only point and the actual search stop.
#[derive(Debug, Clone)]
pub struct Solution {
    /// Best evaluated position, including interrupted initialization/shrink steps.
    pub x: Vector,
    /// Objective at `x`.
    pub value: f64,
    /// Completed outer progress index when the search stopped.
    pub iterations: usize,
    /// Attempted calls; these methods never request gradients.
    pub evaluations: EvaluationCounts,
    /// Target, limit or stagnation; there is no gradient-based convergence flag.
    pub termination_reason: SearchTermination,
}
fn report<P: Evaluator + ?Sized>(
    ctx: &EvaluationContext<'_, P>,
    iteration: usize,
    reason: SearchTermination,
) -> Solution {
    let best = ctx.best().expect("evaluated initial point");
    Solution {
        x: best.x.clone(),
        value: best.value,
        iterations: iteration,
        evaluations: ctx.counts(),
        termination_reason: reason,
    }
}
fn finish<P: Evaluator + ?Sized>(
    ctx: &EvaluationContext<'_, P>,
    iteration: usize,
    result: Result<Solution, Error>,
) -> Result<Solution, Error> {
    match result {
        Err(Error::SearchStopped(reason))
            if ctx.stop_reason() == Some(reason) && ctx.best().is_some() =>
        {
            Ok(report(ctx, iteration, reason))
        }
        Err(e) => Err(ctx.failure(e)),
        Ok(r) => Ok(r),
    }
}
/// Poll positive/negative coordinate directions around the current best point.
///
/// An unsuccessful poll halves the normalized mesh. Zero-width coordinates are
/// skipped; unchanged projected trials are not evaluated. O(n) working vectors.
/// `x0` must be finite, dimension-correct and in bounds. Small mesh or an entirely
/// fixed domain returns [`SearchTermination::Stagnated`], unless a target/limit
/// already applies. Errors retain a best-point checkpoint when available.
pub fn pattern_search<P: Evaluator + ?Sized>(
    p: &P,
    x0: &Vector,
    bounds: &Bounds,
    options: &Options,
) -> Result<Solution, Error> {
    options.validate()?;
    bounds.validate_point(x0.as_slice())?;
    let mut ctx = EvaluationContext::new(p, bounds.dimension(), &options.budget)?;
    let mut iteration = 0;
    let result = (|| {
        ctx.evaluate(x0.as_slice())?;
        let mut x = x0.clone();
        let mut step = options.initial_step;
        for k in 0..=options.max_iterations {
            iteration = k;
            if let Some(reason) = ctx.stop_reason() {
                return Ok(report(&ctx, k, reason));
            }
            if step <= options.step_tolerance
                || (0..bounds.dimension()).all(|i| bounds.width(i) == 0.0)
            {
                return Ok(report(&ctx, k, SearchTermination::Stagnated));
            }
            if k == options.max_iterations {
                return Ok(report(&ctx, k, SearchTermination::MaxIterations));
            }
            let before = ctx.best().expect("initial point").value;
            for i in 0..x.len() {
                if bounds.width(i) == 0.0 {
                    continue;
                }
                for sign in [-1.0, 1.0] {
                    let mut trial = x.clone();
                    trial[i] += sign * step * bounds.width(i);
                    bounds.project(trial.as_mut_slice())?;
                    if trial == x {
                        continue;
                    }
                    ctx.evaluate(trial.as_slice())?;
                }
            }
            let best = ctx.best().expect("evaluated point");
            x.copy_from(&best.x);
            if best.value >= before {
                step *= 0.5;
            }
        }
        unreachable!()
    })();
    finish(&ctx, iteration, result)
}
fn trial<P: Evaluator + ?Sized>(
    ctx: &mut EvaluationContext<'_, P>,
    bounds: &Bounds,
    mut x: Vector,
) -> Result<(Vector, f64), Error> {
    bounds.project(x.as_mut_slice())?;
    let value = ctx.evaluate(x.as_slice())?;
    Ok((x, value))
}
/// Projected Nelder-Mead with reflection, expansion, contraction and shrink.
///
/// Coefficients are 1, 2, 1/2 and 1/2. The simplex has one vertex per free
/// coordinate plus the starting vertex; storage is O(n^2). Budget exhaustion
/// during initialization or shrink still returns the best valid point.
/// Boundary projection can collapse the simplex; small normalized diameter
/// reports stagnation, not a stationary-point certificate. An initial simplex
/// that cannot be represented at machine precision is an explicit input error.
pub fn nelder_mead<P: Evaluator + ?Sized>(
    p: &P,
    x0: &Vector,
    bounds: &Bounds,
    options: &Options,
) -> Result<Solution, Error> {
    options.validate()?;
    bounds.validate_point(x0.as_slice())?;
    let mut ctx = EvaluationContext::new(p, bounds.dimension(), &options.budget)?;
    let mut iteration = 0;
    let result = (|| {
        let initial = ctx.evaluate(x0.as_slice())?;
        let free: Vec<_> = (0..bounds.dimension())
            .filter(|i| bounds.width(*i) > 0.0)
            .collect();
        if free.is_empty() {
            return Ok(report(
                &ctx,
                0,
                ctx.stop_reason().unwrap_or(SearchTermination::Stagnated),
            ));
        }
        let d = free.len();
        let mut simplex = vec![(x0.clone(), initial)];
        for i in &free {
            let mut x = x0.clone();
            let distance = options.initial_step * bounds.width(*i);
            x[*i] = if bounds.upper()[*i] - x[*i] >= distance {
                x[*i] + distance
            } else {
                x[*i] - distance.min(x[*i] - bounds.lower()[*i])
            };
            if x == *x0 {
                return Err(Error::InvalidInput(
                    "initial simplex collapses at machine precision",
                ));
            }
            simplex.push(trial(&mut ctx, bounds, x)?);
        }
        for k in 0..=options.max_iterations {
            iteration = k;
            if let Some(reason) = ctx.stop_reason() {
                return Ok(report(&ctx, k, reason));
            }
            simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
            let diameter = simplex
                .iter()
                .skip(1)
                .flat_map(|(x, _)| {
                    free.iter()
                        .map(|i| (x[*i] - simplex[0].0[*i]).abs() / bounds.width(*i))
                })
                .fold(0.0, f64::max);
            if diameter <= options.step_tolerance {
                return Ok(report(&ctx, k, SearchTermination::Stagnated));
            }
            if k == options.max_iterations {
                return Ok(report(&ctx, k, SearchTermination::MaxIterations));
            }
            let mut center = Vector::zeros(x0.len());
            for (x, _) in simplex.iter().take(d) {
                center += x / d as f64;
            }
            let reflected = trial(&mut ctx, bounds, &center + (&center - &simplex[d].0))?;
            if reflected.1 < simplex[0].1 {
                let expanded = trial(&mut ctx, bounds, &center + 2.0 * (&reflected.0 - &center))?;
                simplex[d] = if expanded.1 < reflected.1 {
                    expanded
                } else {
                    reflected
                };
            } else if reflected.1 < simplex[d - 1].1 {
                simplex[d] = reflected;
            } else {
                let outside = reflected.1 < simplex[d].1;
                let contracted = trial(
                    &mut ctx,
                    bounds,
                    if outside {
                        &center + 0.5 * (&reflected.0 - &center)
                    } else {
                        &center + 0.5 * (&simplex[d].0 - &center)
                    },
                )?;
                if contracted.1 < if outside { reflected.1 } else { simplex[d].1 } {
                    simplex[d] = contracted;
                } else {
                    for i in 1..=d {
                        simplex[i] = trial(
                            &mut ctx,
                            bounds,
                            &simplex[0].0 + 0.5 * (&simplex[i].0 - &simplex[0].0),
                        )?;
                    }
                }
            }
        }
        unreachable!()
    })();
    finish(&ctx, iteration, result)
}
