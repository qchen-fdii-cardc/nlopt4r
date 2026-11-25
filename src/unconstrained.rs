use crate::{
    Error, Evaluator, Matrix, Options, Solution, TerminationReason, Vector, finite, initial,
    line_search, solve,
    stopping::{Run, snapshot},
};

#[derive(Debug, Clone, Copy)]
pub enum Method {
    SteepestDescent,
    NormalizedGradient,
    FletcherReeves,
    DampedNewton,
    ModifiedNewton,
    Bfgs,
    Dfp,
    Sr1,
    /// Convex Broyden inverse-Hessian family: 0 is DFP, 1 is BFGS.
    Broyden {
        phi: f64,
    },
}

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
    let mut inverse = Matrix::identity(n, n);
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
            _ => -&inverse * &g,
        };
        // FR and indefinite Newton/SR1 models need a descent safeguard.
        if g.dot(&d) >= -1e-12 * g.norm() * d.norm() {
            d = -&g;
            inverse = Matrix::identity(n, n);
        }
        let step = line_search::backtrack(f, g.dot(&d), &options.line_search, |alpha| {
            run.value(problem, &(&x + alpha * &d))
        })?;
        let next = &x + step.alpha * &d;
        let (new_f, new_g) = run.evaluate(problem, &next)?;
        let s = &next - &x;
        let y = &new_g - &g;
        update_inverse(&mut inverse, &s, &y, method);
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
