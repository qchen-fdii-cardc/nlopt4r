//! Armijo backtracking and bounded scalar minimization.
//!
//! [`armijo`] searches along a supplied descent direction and needs an analytic
//! gradient at the origin. Scalar golden-section/Brent methods need only values
//! but assume a continuous unimodal objective on a finite closed interval.
//!
//! ```
//! use nlopt4r::line_search::{minimize_scalar_with_options, ScalarMethod, ScalarOptions};
//! let result = minimize_scalar_with_options(
//!     |x| (x-0.4).powi(2), (-1.0, 2.0), ScalarMethod::Parabolic, &ScalarOptions::default(),
//! )?;
//! assert!((result.x-0.4).abs() < 1e-6);
//! assert_eq!(result.status, nlopt4r::Status::Converged);
//! # Ok::<(), nlopt4r::Error>(())
//! ```
use crate::{
    Error, EvaluationCounts, Evaluator, StopOptions, TerminationReason, Tolerance, Vector,
    evaluate, finite, positive,
    stopping::{Run, snapshot},
    vector,
};

/// Armijo sufficient-decrease backtracking controls.
#[derive(Debug, Clone)]
pub struct ArmijoOptions {
    /// Positive finite initial trial length, default 1.
    pub initial_step: f64,
    /// Multiply rejected step lengths by this factor in `(0,1)`, default 0.5.
    pub contraction: f64,
    /// c1 in `f(x+a*d) <= f(x)+c1*a*g'd`, in `(0,1)`, default 0.2.
    pub sufficient_decrease: f64,
    /// Positive trial allowance, default 60; no unchecked fallback after exhaustion.
    pub max_iterations: usize,
}

impl Default for ArmijoOptions {
    fn default() -> Self {
        Self {
            initial_step: 1.0,
            contraction: 0.5,
            sufficient_decrease: 0.2,
            max_iterations: 60,
        }
    }
}

impl ArmijoOptions {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        positive(self.initial_step, "initial step")?;
        if !(0.0..1.0).contains(&self.contraction) || self.contraction == 0.0 {
            return Err(Error::InvalidInput("contraction must be in (0, 1)"));
        }
        if !(0.0..1.0).contains(&self.sufficient_decrease) || self.sufficient_decrease == 0.0 {
            return Err(Error::InvalidInput("sufficient decrease must be in (0, 1)"));
        }
        if self.max_iterations == 0 {
            return Err(Error::InvalidInput(
                "line search iteration limit must be positive",
            ));
        }
        Ok(())
    }
}

/// Accepted Armijo trial; does not contain a gradient.
#[derive(Debug, Clone)]
pub struct Step {
    /// Accepted step length multiplying the supplied direction.
    pub alpha: f64,
    /// Objective at the accepted point.
    pub value: f64,
    /// Number of trial evaluations.
    pub evaluations: usize,
}

/// Find an Armijo step along a finite dimension-correct descent direction.
///
/// Evaluates value/gradient at `x` once, then requests only values at trials.
/// [`Step::evaluations`] excludes the origin call. This standalone helper does
/// not enforce an outer objective budget; full solvers charge trials themselves.
/// Invalid directions/options, non-finite trials, callback failures and
/// exhausted backtracking are explicit errors.
pub fn armijo<P: Evaluator + ?Sized>(
    problem: &P,
    x: &Vector<f64>,
    direction: &Vector<f64>,
    options: &ArmijoOptions,
) -> Result<Step, Error> {
    options.validate()?;
    if x.is_empty() {
        return Err(Error::InvalidInput("empty point"));
    }
    vector(direction, x.len(), "direction")?;
    let (value, gradient) = evaluate(problem, x)?;
    backtrack(value, gradient.dot(direction), options, |alpha| {
        let trial = x + alpha * direction;
        vector(&trial, x.len(), "line search point")?;
        finite(
            problem.evaluate(trial.as_slice(), None)?,
            "line search objective",
        )
    })
}

pub(crate) fn backtrack<F>(
    value: f64,
    slope: f64,
    options: &ArmijoOptions,
    mut trial: F,
) -> Result<Step, Error>
where
    F: FnMut(f64) -> Result<f64, Error>,
{
    finite(slope, "directional derivative")?;
    if slope >= 0.0 {
        return Err(Error::InvalidInput(
            "line search requires a descent direction",
        ));
    }
    let mut alpha = options.initial_step;
    let mut last_step = alpha;
    for k in 0..options.max_iterations {
        let next = trial(alpha)?;
        last_step = alpha;
        if next <= value + options.sufficient_decrease * alpha * slope {
            return Ok(Step {
                alpha,
                value: next,
                evaluations: k + 1,
            });
        }
        alpha *= options.contraction;
    }
    Err(Error::LineSearchFailed {
        evaluations: options.max_iterations,
        last_step,
    })
}

/// Derivative-free method for a continuous unimodal scalar bracket.
#[derive(Debug, Clone, Copy)]
pub enum ScalarMethod {
    /// Golden-section interval reduction.
    GoldenSection,
    /// Safeguarded parabolic interpolation (Brent's method).
    Parabolic,
}

/// Best evaluated scalar point, remaining bracket and actual termination.
#[derive(Debug, Clone)]
pub struct ScalarSolution {
    /// Returned scalar point.
    pub x: f64,
    /// Objective at `x`.
    pub value: f64,
    /// Remaining search interval.
    pub bracket: (f64, f64),
    /// Completed scalar iterations.
    pub iterations: usize,
    /// Interval convergence or a non-converged budget/iteration stop.
    pub status: crate::Status,
    /// Detailed stop criterion.
    pub termination_reason: TerminationReason,
    /// Attempted objective calls, including endpoint evaluations; no derivatives.
    pub evaluations: EvaluationCounts,
}

/// Scalar interval tolerance and shared iteration/objective allowances.
#[derive(Debug, Clone)]
pub struct ScalarOptions {
    /// Shared controls; gradient tolerance is validated but not used for scalar optimality.
    pub stopping: StopOptions,
    /// Mixed interval tolerance; defaults to absolute 1e-8 and relative sqrt(machine epsilon).
    pub interval_tolerance: Tolerance,
}
impl Default for ScalarOptions {
    fn default() -> Self {
        Self {
            stopping: StopOptions::default(),
            interval_tolerance: Tolerance {
                absolute: 1e-8,
                relative: f64::EPSILON.sqrt(),
            },
        }
    }
}

/// Minimize a continuous unimodal function on a finite closed interval.
///
/// Convenience wrapper using absolute `tolerance`, relative sqrt(machine epsilon),
/// and the supplied iteration cap without an objective cap. Use
/// [`minimize_scalar_with_options`] for strict evaluation budgets.
pub fn minimize_scalar<F: Fn(f64) -> f64>(
    f: F,
    bounds: (f64, f64),
    tolerance: f64,
    max_iterations: usize,
    method: ScalarMethod,
) -> Result<ScalarSolution, Error> {
    positive(tolerance, "scalar tolerance")?;
    let options = ScalarOptions {
        stopping: StopOptions {
            max_iterations,
            ..StopOptions::default()
        },
        interval_tolerance: Tolerance {
            absolute: tolerance,
            relative: f64::EPSILON.sqrt(),
        },
    };
    minimize_scalar_with_options(f, bounds, method, &options)
}

/// Search a finite strictly increasing `(lower, upper)` interval, including
/// its endpoints, without derivatives.
///
/// All callback values must be finite. Interval convergence is meaningful under
/// the continuity/unimodality assumption, not a general global guarantee.
/// A budget interrupted before enough evaluations returns a non-converged
/// report if a valid point exists. Non-finite/input failures are explicit;
/// scalar error checkpoints use a length-one vector and NaN gradient norm.
pub fn minimize_scalar_with_options<F: Fn(f64) -> f64>(
    f: F,
    bounds: (f64, f64),
    method: ScalarMethod,
    options: &ScalarOptions,
) -> Result<ScalarSolution, Error> {
    options.interval_tolerance.validate()?;
    let mut run = Run::new(&options.stopping)?;
    let result = scalar_impl(f, bounds, method, options, &mut run);
    match result {
        Ok(result) => Ok(result),
        Err(error) => {
            let last = run.finish(Err(error), "scalar minimization")?;
            Ok(ScalarSolution {
                x: last.x[0],
                value: last.value,
                bracket: bounds,
                iterations: last.iterations,
                status: last.status,
                termination_reason: last.termination_reason,
                evaluations: last.evaluations,
            })
        }
    }
}

fn scalar_impl<F: Fn(f64) -> f64>(
    f: F,
    bounds: (f64, f64),
    method: ScalarMethod,
    options: &ScalarOptions,
    run: &mut Run,
) -> Result<ScalarSolution, Error> {
    let (mut a, mut b) = bounds;
    if !a.is_finite() || !b.is_finite() || a >= b || !(b - a).is_finite() {
        return Err(Error::InvalidInput("scalar bounds"));
    }
    let eval = |x: f64, run: &mut Run| {
        run.charge()?;
        let value = finite(f(x), "scalar objective")?;
        if run.last.as_ref().is_none_or(|last| value < last.value) {
            run.checkpoint(snapshot(
                Vector::from_element(1, x),
                value,
                f64::NAN,
                0,
                None,
            ));
        }
        Ok::<_, Error>(value)
    };
    let endpoints = [(a, eval(a, run)?), (b, eval(b, run)?)];
    let golden = (3.0 - 5.0_f64.sqrt()) / 2.0;
    let mut x = a + golden * (b - a);
    let mut w = x;
    let mut v = x;
    let mut fx = eval(x, run)?;
    let mut fw = fx;
    let mut fv = fx;
    let mut d: f64 = 0.0;
    let mut e: f64 = 0.0;
    let mut k = 0;
    let reason = loop {
        let mid = 0.5 * (a + b);
        let tol =
            options.interval_tolerance.absolute + options.interval_tolerance.relative * x.abs();
        let mut best = (x, fx);
        for &endpoint in &endpoints {
            if endpoint.1 < best.1 {
                best = endpoint;
            }
        }
        let reason = run.reason(
            &Vector::from_element(1, best.0),
            best.1,
            k,
            (x - mid).abs() <= 2.0 * tol - 0.5 * (b - a),
            TerminationReason::IntervalTolerance,
            k > 0,
        );
        run.record(snapshot(
            Vector::from_element(1, best.0),
            best.1,
            f64::NAN,
            k,
            reason,
        ));
        if let Some(reason) = reason {
            break reason;
        }
        let mut parabolic = false;
        if matches!(method, ScalarMethod::Parabolic) && e.abs() > tol {
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            let mut q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            let previous = e;
            e = d;
            if p.abs() < (0.5 * q * previous).abs() && p > q * (a - x) && p < q * (b - x) {
                d = p / q;
                let u = x + d;
                if u - a < 2.0 * tol || b - u < 2.0 * tol {
                    d = if mid >= x { tol } else { -tol };
                }
                parabolic = true;
            }
        }
        if !parabolic {
            e = if x < mid { b - x } else { a - x };
            d = golden * e;
        }
        let u = x + if d.abs() >= tol {
            d
        } else if d >= 0.0 {
            tol
        } else {
            -tol
        };
        let fu = eval(u, run)?;
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
        k += 1;
    };
    for (point, value) in endpoints {
        if value < fx {
            x = point;
            fx = value;
        }
    }
    a = a.min(x);
    b = b.max(x);
    Ok(ScalarSolution {
        x,
        value: fx,
        bracket: (a, b),
        iterations: k,
        status: snapshot(Vector::from_element(1, x), fx, 0.0, k, Some(reason)).status,
        termination_reason: reason,
        evaluations: run.counts,
    })
}
