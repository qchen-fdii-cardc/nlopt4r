use crate::{
    Error, Matrix, Objective, Options, Solution, Vector, evaluate, hessian, initial, line_search,
    solution, solve,
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

pub fn minimize<P: Objective + ?Sized>(
    problem: &P,
    x0: &Vector<f64>,
    method: Method,
    options: &Options,
) -> Result<Solution, Error> {
    initial(x0, options)?;
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
    for k in 0..=options.max_iterations {
        let (f, g) = evaluate(problem, &x)?;
        let norm = g.norm();
        let converged = norm <= options.gradient_tolerance;
        if converged || k == options.max_iterations {
            return Ok(solution(x, f, norm, k, converged));
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
            Method::DampedNewton => solve(&hessian(problem, &x)?, &(-&g))?,
            Method::ModifiedNewton => {
                let h = hessian(problem, &x)?;
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
        let step = line_search::armijo(problem, &x, &d, &options.line_search)?;
        let next = &x + step.alpha * &d;
        let (_, new_g) = evaluate(problem, &next)?;
        let s = &next - &x;
        let y = &new_g - &g;
        update_inverse(&mut inverse, &s, &y, method);
        previous_g = g;
        previous_d = d;
        x = next;
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
