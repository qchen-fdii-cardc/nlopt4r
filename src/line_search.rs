//! Armijo backtracking and bounded scalar minimization.
use crate::{Error, Objective, Vector, evaluate, finite, positive, vector};

#[derive(Debug, Clone)]
pub struct ArmijoOptions {
    pub initial_step: f64,
    pub contraction: f64,
    pub sufficient_decrease: f64,
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

#[derive(Debug, Clone)]
pub struct Step {
    pub alpha: f64,
    pub value: f64,
    /// Number of trial evaluations.
    pub evaluations: usize,
}

pub fn armijo<P: Objective + ?Sized>(
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
        finite(problem.value(&trial), "line search objective")
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
    for k in 0..options.max_iterations {
        let next = trial(alpha)?;
        if next <= value + options.sufficient_decrease * alpha * slope {
            return Ok(Step {
                alpha,
                value: next,
                evaluations: k + 1,
            });
        }
        alpha *= options.contraction;
    }
    Err(Error::LineSearchFailed)
}

#[derive(Debug, Clone, Copy)]
pub enum ScalarMethod {
    GoldenSection,
    /// Safeguarded parabolic interpolation (Brent's method).
    Parabolic,
}

#[derive(Debug, Clone)]
pub struct ScalarSolution {
    pub x: f64,
    pub value: f64,
    pub bracket: (f64, f64),
    pub iterations: usize,
    pub status: crate::Status,
}

/// Minimize a continuous unimodal function on a finite closed interval.
pub fn minimize_scalar<F: Fn(f64) -> f64>(
    f: F,
    bounds: (f64, f64),
    tolerance: f64,
    max_iterations: usize,
    method: ScalarMethod,
) -> Result<ScalarSolution, Error> {
    positive(tolerance, "scalar tolerance")?;
    let (mut a, mut b) = bounds;
    if !a.is_finite() || !b.is_finite() || a >= b || !(b - a).is_finite() {
        return Err(Error::InvalidInput("scalar bounds"));
    }
    let eval = |x| finite(f(x), "scalar objective");
    let endpoints = [(a, eval(a)?), (b, eval(b)?)];
    let golden = (3.0 - 5.0_f64.sqrt()) / 2.0;
    let mut x = a + golden * (b - a);
    let mut w = x;
    let mut v = x;
    let mut fx = eval(x)?;
    let mut fw = fx;
    let mut fv = fx;
    let mut d: f64 = 0.0;
    let mut e: f64 = 0.0;
    let mut k = 0;
    while k < max_iterations {
        let mid = 0.5 * (a + b);
        let tol = tolerance + f64::EPSILON.sqrt() * x.abs();
        if (x - mid).abs() <= 2.0 * tol - 0.5 * (b - a) {
            break;
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
        let fu = eval(u)?;
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
    }
    let tol = tolerance + f64::EPSILON.sqrt() * x.abs();
    let converged = (x - 0.5 * (a + b)).abs() <= 2.0 * tol - 0.5 * (b - a);
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
        status: if converged {
            crate::Status::Converged
        } else {
            crate::Status::MaxIterations
        },
    })
}
