//! Bracketing and safeguarded bisection for the strong Wolfe conditions.
//!
//! Armijo decrease is `f(x+a*d) <= f(x)+c1*a*g'd`; strong curvature requires
//! `|g(x+a*d)'d| <= c2*|g'd|`. These conditions support safe quasi-Newton updates.
//! [`strong_wolfe`] accepts a callback so external solvers can own accounting.
use crate::{Error, Vector, finite, positive, vector};

/// Strong-Wolfe parameters shared by bracketing and zoom.
#[derive(Debug, Clone)]
pub struct StrongWolfeOptions {
    /// Armijo c1, default 1e-4; must satisfy `0 < c1 < c2 < 1`.
    pub sufficient_decrease: f64,
    /// Strong-curvature c2, default 0.9.
    pub curvature: f64,
    /// Positive finite starting trial length, default 1.
    pub initial_step: f64,
    /// Positive finite expansion cap, at least `initial_step`; default 1e20.
    pub maximum_step: f64,
    /// Total trial callbacks, shared by bracketing and zoom.
    /// Positive, default 80; excludes the supplied origin evaluation.
    pub max_evaluations: usize,
}

impl Default for StrongWolfeOptions {
    fn default() -> Self {
        Self {
            sufficient_decrease: 1e-4,
            curvature: 0.9,
            initial_step: 1.0,
            maximum_step: 1e20,
            max_evaluations: 80,
        }
    }
}

impl StrongWolfeOptions {
    /// Validate coefficient order, step bounds and positive trial allowance.
    pub fn validate(&self) -> Result<(), Error> {
        if !(0.0 < self.sufficient_decrease
            && self.sufficient_decrease < self.curvature
            && self.curvature < 1.0)
            || self.max_evaluations == 0
        {
            return Err(Error::InvalidInput("strong Wolfe parameters"));
        }
        positive(self.initial_step, "initial Wolfe step")?;
        positive(self.maximum_step, "maximum Wolfe step")?;
        if self.initial_step > self.maximum_step {
            return Err(Error::InvalidInput("initial Wolfe step exceeds maximum"));
        }
        Ok(())
    }
}

/// Accepted line-search state, available for reuse without re-evaluation.
#[derive(Debug, Clone)]
pub struct StrongWolfeStep {
    /// Accepted positive step length.
    pub alpha: f64,
    /// Objective at the accepted trial.
    pub value: f64,
    /// Analytic gradient at that same trial.
    pub gradient: Vector,
    /// Number of combined value/gradient callbacks, excluding the supplied origin.
    pub evaluations: usize,
}

/// Find a step satisfying Armijo decrease and `|g(alpha)'d| <= c2 |g(0)'d|`.
///
/// `evaluate(alpha)` must return the value and gradient at `x + alpha * direction`.
/// The accepted gradient is returned for reuse, without another objective call.
/// Callback failures propagate; neither budgets nor invalid trials are silently retried.
///
/// The supplied origin value, gradient and direction must be finite and
/// dimension-consistent, with `gradient.dot(direction) < 0`. The callback owns
/// objective budgets; this routine separately caps total trials. Search exhaustion
/// or a bracket that cannot be refined at machine precision returns
/// [`Error::LineSearchFailed`], not an unchecked candidate.
pub fn strong_wolfe<F>(
    value: f64,
    gradient: &Vector,
    direction: &Vector,
    options: &StrongWolfeOptions,
    mut evaluate: F,
) -> Result<StrongWolfeStep, Error>
where
    F: FnMut(f64) -> Result<(f64, Vector), Error>,
{
    options.validate()?;
    finite(value, "Wolfe origin value")?;
    if gradient.is_empty() {
        return Err(Error::InvalidInput("empty Wolfe gradient"));
    }
    vector(gradient, gradient.len(), "Wolfe origin gradient")?;
    vector(direction, gradient.len(), "Wolfe direction")?;
    let slope = finite(gradient.dot(direction), "Wolfe origin derivative")?;
    if slope >= 0.0 {
        return Err(Error::InvalidInput("Wolfe direction must be descent"));
    }
    let mut previous_alpha = 0.0;
    let mut previous_value = value;
    let mut alpha = options.initial_step;
    let mut bracket: Option<(f64, f64, f64)> = None;
    for evaluations in 1..=options.max_evaluations {
        if let Some((lo, hi, _)) = bracket {
            // Midpoints are always interior: unlike unchecked interpolation they
            // cannot jump out of the bracket or approach one endpoint indefinitely.
            alpha = lo + 0.5 * (hi - lo);
            if alpha == lo || alpha == hi || alpha <= 0.0 {
                return Err(Error::LineSearchFailed {
                    evaluations: evaluations - 1,
                    last_step: alpha,
                });
            }
        }
        let (f, g) = evaluate(alpha)?;
        finite(f, "Wolfe trial value")?;
        vector(&g, gradient.len(), "Wolfe trial gradient")?;
        let derivative = finite(g.dot(direction), "Wolfe trial derivative")?;
        let armijo = f <= value + options.sufficient_decrease * alpha * slope;
        if let Some((lo, hi, flo)) = bracket {
            if !armijo || f >= flo {
                bracket = Some((lo, alpha, flo));
            } else {
                if derivative.abs() <= -options.curvature * slope {
                    return Ok(StrongWolfeStep {
                        alpha,
                        value: f,
                        gradient: g,
                        evaluations,
                    });
                }
                let hi = if derivative * (hi - lo) >= 0.0 {
                    lo
                } else {
                    hi
                };
                bracket = Some((alpha, hi, f));
            }
        } else if !armijo || (evaluations > 1 && f >= previous_value) {
            bracket = Some((previous_alpha, alpha, previous_value));
        } else {
            if derivative.abs() <= -options.curvature * slope {
                return Ok(StrongWolfeStep {
                    alpha,
                    value: f,
                    gradient: g,
                    evaluations,
                });
            }
            if derivative >= 0.0 {
                bracket = Some((alpha, previous_alpha, f));
            } else {
                previous_alpha = alpha;
                previous_value = f;
                let next = (2.0 * alpha).min(options.maximum_step);
                if next <= alpha {
                    return Err(Error::LineSearchFailed {
                        evaluations,
                        last_step: alpha,
                    });
                }
                alpha = next;
            }
        }
    }
    Err(Error::LineSearchFailed {
        evaluations: options.max_evaluations,
        last_step: alpha,
    })
}
