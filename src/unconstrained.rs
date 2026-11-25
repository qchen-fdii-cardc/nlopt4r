//! Legacy smooth local methods, reexported at the crate root and in [`crate::local`].
use crate::{
    Error, Evaluator, Matrix, Options, Solution, TerminationReason, Vector, finite, initial,
    line_search, solve,
    stopping::{Run, snapshot},
};

/// Smooth unconstrained method used by [`minimize`].
///
/// Gradient/CG methods use vector-sized workspace; quasi-Newton methods store
/// a dense inverse approximation. Newton methods additionally require an analytic
/// Hessian. For scalable alternatives see [`crate::local`].
#[derive(Debug, Clone, Copy)]
pub enum Method {
    /// Steepest descent `d=-gradient`.
    SteepestDescent,
    /// Unit-length negative-gradient direction.
    NormalizedGradient,
    /// Nonlinear conjugate gradients with periodic restart and descent safeguards.
    FletcherReeves,
    /// Dense Newton equation plus Armijo damping and a non-descent safeguard.
    DampedNewton,
    /// Dense Newton with a positive-definite shifted Hessian.
    ModifiedNewton,
    /// Dense inverse BFGS with guarded curvature updates.
    Bfgs,
    /// Dense Davidon-Fletcher-Powell inverse update.
    Dfp,
    /// Dense symmetric-rank-one inverse update with denominator safeguards.
    Sr1,
    /// Convex Broyden inverse-Hessian family: 0 is DFP, 1 is BFGS.
    Broyden {
        /// Finite interpolation weight in `[0,1]`.
        phi: f64,
    },
}

/// Minimize a smooth unconstrained objective from a finite nonempty initial point.
///
/// Every method requires an analytic gradient; dense Newton variants also need a
/// symmetric Hessian. All use [`crate::line_search::armijo`] settings. Domains and
/// bounds are not enforced: the callback must be defined at every trial.
///
/// Normal budget/stagnation stops return a [`Solution`] that may not converge.
/// Invalid options, callbacks and failed searches return explicit errors, with
/// [`Error::last_solution`] after a fully evaluated checkpoint.
pub fn minimize<P: Evaluator + ?Sized>(
    problem: &P,
    x0: &Vector<f64>,
    method: Method,
    options: &Options,
) -> Result<Solution, Error> {
    options.validate()?;
    initial(x0, &options.stopping)?;
    let mut run = Run::new(&options.stopping)?;
    let result = minimize_impl(problem, x0, method, options, &mut run);
    run.finish(result, "unconstrained minimization")
}

fn minimize_impl<P: Evaluator + ?Sized>(
    problem: &P,
    x0: &Vector,
    method: Method,
    options: &Options,
    run: &mut Run,
) -> Result<Solution, Error> {
    if let Method::Broyden { phi } = method
        && (!phi.is_finite() || !(0.0..=1.0).contains(&phi))
    {
        return Err(Error::InvalidInput("Broyden phi must be in [0, 1]"));
    }
    let n = x0.len();
    let mut x = x0.clone();
    let mut inverse = matches!(
        method,
        Method::Bfgs | Method::Dfp | Method::Sr1 | Method::Broyden { .. }
    )
    .then(|| Matrix::identity(n, n));
    let mut previous_g = Vector::zeros(n);
    let mut previous_d = Vector::zeros(n);
    let (mut f, mut g) = run.evaluate(problem, &x)?;
    for k in 0..=options.stopping.max_iterations {
        let norm = finite(g.norm(), "gradient norm")?;
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
        let mut d = match method {
            Method::SteepestDescent => -&g,
            Method::NormalizedGradient => -&g / norm,
            Method::FletcherReeves => {
                if k % (n + 1) == 0 {
                    -&g
                } else {
                    -&g + g.norm_squared() / previous_g.norm_squared() * &previous_d
                }
            }
            Method::DampedNewton => solve(&run.hessian(problem, &x)?, &(-&g))?,
            Method::ModifiedNewton => {
                let h = run.hessian(problem, &x)?;
                let mut shift = norm;
                loop {
                    let shifted = &h + shift * Matrix::identity(n, n);
                    if let Some(chol) = shifted.cholesky() {
                        break chol.solve(&(-&g));
                    }
                    shift *= 2.0;
                    if !shift.is_finite() {
                        return Err(Error::NonFinite("Newton regularization"));
                    }
                }
            }
            _ => -inverse.as_ref().expect("quasi-Newton workspace") * &g,
        };
        // FR and indefinite Newton/SR1 models need a descent safeguard.
        if g.dot(&d) >= -1e-12 * g.norm() * d.norm() {
            d = -&g;
            if let Some(inverse) = &mut inverse {
                *inverse = Matrix::identity(n, n);
            }
        }
        let step = line_search::backtrack(f, g.dot(&d), &options.line_search, |alpha| {
            run.value(problem, &(&x + alpha * &d))
        })?;
        let next = &x + step.alpha * &d;
        let (new_f, new_g) = run.evaluate(problem, &next)?;
        let s = &next - &x;
        let y = &new_g - &g;
        if let Some(inverse) = &mut inverse {
            update_inverse(inverse, &s, &y, method);
        }
        previous_g = g;
        previous_d = d;
        x = next;
        f = new_f;
        g = new_g;
    }
    unreachable!()
}

pub(crate) fn update_inverse(
    h: &mut Matrix<f64>,
    s: &Vector<f64>,
    y: &Vector<f64>,
    method: Method,
) {
    let hy = &*h * y;
    if matches!(method, Method::Sr1) {
        let r = s - &hy;
        let denominator = r.dot(y);
        if denominator.abs() > 1e-8 * r.norm() * y.norm() {
            *h += &r * r.transpose() / denominator;
        }
        return;
    }
    let phi = match method {
        Method::Bfgs => 1.0,
        Method::Dfp => 0.0,
        Method::Broyden { phi } => phi,
        _ => return,
    };
    let sy = s.dot(y);
    let yhy = y.dot(&hy);
    if sy > 1e-12 * s.norm() * y.norm() && yhy > 0.0 {
        let v = s / sy - &hy / yhy;
        *h += s * s.transpose() / sy - &hy * hy.transpose() / yhy + phi * yhy * &v * v.transpose();
        *h = 0.5 * (&*h + h.transpose());
    }
}
