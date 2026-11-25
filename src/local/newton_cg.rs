//! Truncated Newton-CG using application-supplied Hessian-vector products.
use crate::{
    Error, Evaluator, Solution, StopOptions, TerminationReason, Vector, finite, initial, positive,
    stopping::{Run, snapshot},
    vector,
};

use super::{StrongWolfeOptions, strong_wolfe};

/// Matrix-free second derivatives. Fill every entry of `product` with H(x) v.
///
/// Implementations must represent a symmetric linear operator at each fixed x.
/// No dense Hessian is requested. Each product, including a failed callback, is
/// reported in `Solution.evaluations.hessian`; objective budgets exclude products.
pub trait HessianVector: Evaluator {
    fn hessian_vector(&self, x: &[f64], v: &[f64], product: &mut [f64]) -> Result<(), Error>;
}

pub(crate) fn product<P: HessianVector + ?Sized>(
    problem: &P,
    x: &Vector,
    v: &Vector,
    run: &mut Run,
) -> Result<Vector, Error> {
    vector(v, x.len(), "Hessian-vector direction")?;
    let mut result = Vector::from_element(x.len(), f64::NAN);
    run.counts.hessian += 1;
    problem.hessian_vector(x.as_slice(), v.as_slice(), result.as_mut_slice())?;
    vector(&result, x.len(), "Hessian-vector product")?;
    Ok(result)
}

#[derive(Debug, Clone)]
pub struct NewtonCgOptions {
    pub stopping: StopOptions,
    pub line_search: StrongWolfeOptions,
    pub max_cg_iterations: usize,
    /// Relative residual threshold for the inner Newton equation.
    pub cg_tolerance: f64,
}

impl Default for NewtonCgOptions {
    fn default() -> Self {
        Self {
            stopping: StopOptions::default(),
            line_search: StrongWolfeOptions::default(),
            max_cg_iterations: 100,
            cg_tolerance: 0.1,
        }
    }
}

impl NewtonCgOptions {
    pub fn validate(&self) -> Result<(), Error> {
        self.stopping.validate()?;
        self.line_search.validate()?;
        positive(self.cg_tolerance, "Newton-CG tolerance")?;
        if self.cg_tolerance >= 1.0 || self.max_cg_iterations == 0 {
            return Err(Error::InvalidInput("Newton-CG options"));
        }
        Ok(())
    }
}

fn direction<F>(g: &Vector, options: &NewtonCgOptions, mut multiply: F) -> Result<Vector, Error>
where
    F: FnMut(&Vector) -> Result<Vector, Error>,
{
    let mut p = Vector::zeros(g.len());
    let mut r = g.clone();
    let mut d = -&r;
    let target = options.cg_tolerance * g.norm();
    let mut rr = finite(r.norm_squared(), "Newton-CG squared residual")?;
    for i in 0..options.max_cg_iterations.min(g.len()) {
        let hd = multiply(&d)?;
        let curvature = finite(d.dot(&hd), "Newton-CG curvature")?;
        if curvature <= 0.0 {
            return Ok(if i == 0 { -g } else { p });
        }
        let alpha = finite(rr / curvature, "Newton-CG step")?;
        p += alpha * &d;
        r += alpha * hd;
        vector(&p, g.len(), "Newton-CG direction")?;
        let norm = finite(r.norm(), "Newton-CG residual")?;
        if norm <= target {
            return Ok(p);
        }
        let next_rr = finite(norm * norm, "Newton-CG squared residual")?;
        d = -&r + (next_rr / rr) * d;
        rr = next_rr;
    }
    Ok(p)
}

pub fn newton_cg<P: HessianVector + ?Sized>(
    problem: &P,
    x0: &Vector,
    options: &NewtonCgOptions,
) -> Result<Solution, Error> {
    options.validate()?;
    initial(x0, &options.stopping)?;
    let mut run = Run::new(&options.stopping)?;
    let result = minimize_impl(problem, x0, options, &mut run);
    run.finish(result, "Newton-CG")
}

fn minimize_impl<P: HessianVector + ?Sized>(
    problem: &P,
    x0: &Vector,
    options: &NewtonCgOptions,
    run: &mut Run,
) -> Result<Solution, Error> {
    let mut x = x0.clone();
    let (mut f, mut g) = run.evaluate(problem, &x)?;
    for k in 0..=options.stopping.max_iterations {
        let norm = finite(g.norm(), "Newton-CG gradient norm")?;
        let reason = run.reason(
            &x,
            f,
            k,
            norm <= options.stopping.gradient_tolerance,
            TerminationReason::GradientTolerance,
            k > 0,
        );
        let report = run.record(snapshot(x.clone(), f, norm, k, reason));
        if reason.is_some() {
            return Ok(report);
        }
        run.counts.subproblem += 1;
        let mut d = direction(&g, options, |v| product(problem, &x, v, run))?;
        if !g.dot(&d).is_finite() || g.dot(&d) >= -1e-12 * norm * d.norm() {
            d = -&g;
        }
        let step = strong_wolfe(f, &g, &d, &options.line_search, |alpha| {
            run.evaluate(problem, &(&x + alpha * &d))
        })?;
        x += step.alpha * d;
        f = step.value;
        g = step.gradient;
    }
    unreachable!()
}
