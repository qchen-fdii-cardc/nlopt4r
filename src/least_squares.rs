//! Levenberg-Marquardt for square and rectangular residual systems.
use crate::{
    Error, Matrix, Options, Solution, Vector, initial, line_search,
    matrix, solution, solve, vector,
};

pub trait LeastSquares {
    fn residuals(&self, x: &Vector) -> Vector;
    /// One row per residual, one column per variable.
    fn jacobian(&self, x: &Vector) -> Matrix;
}

impl<F, J> LeastSquares for (F, J)
where F: Fn(&Vector) -> Vector, J: Fn(&Vector) -> Matrix {
    fn residuals(&self, x: &Vector) -> Vector { self.0(x) }
    fn jacobian(&self, x: &Vector) -> Matrix { self.1(x) }
}

/// Minimize half the squared residual norm. Convergence means J'r is small,
/// not necessarily that the residual is zero.
pub fn levenberg_marquardt<P: LeastSquares + ?Sized>(
    problem: &P, x0: &Vector, options: &Options,
) -> Result<Solution, Error> {
    initial(x0, options)?;
    let mut x = x0.clone();
    let mut residual_count = None;
    for k in 0..=options.max_iterations {
        let r = problem.residuals(&x);
        let m = *residual_count.get_or_insert(r.len());
        if m == 0 { return Err(Error::InvalidInput("empty residuals")); }
        vector(&r, m, "residuals")?;
        let j = problem.jacobian(&x);
        matrix(&j, m, x.len(), "Jacobian")?;
        let g = j.transpose() * &r;
        let value = crate::finite(0.5 * r.norm_squared(), "least-squares objective")?;
        let converged = g.norm() <= options.gradient_tolerance;
        if converged || k == options.max_iterations {
            return Ok(solution(x, value, g.norm(), k, converged));
        }
        let mu = r.norm().max(1e-12);
        let a = j.transpose() * j + mu * Matrix::identity(x.len(), x.len());
        let d = solve(&a, &(-&g))?;
        // Validate the residual shape on every trial before computing its merit.
        let step = line_search::backtrack(value, g.dot(&d),
            &options.line_search, |alpha| {
                let trial = &x + alpha * &d;
                vector(&trial, x.len(), "least-squares iterate")?;
                let r = problem.residuals(&trial);
                vector(&r, m, "residuals")?;
                crate::finite(0.5 * r.norm_squared(), "least-squares objective")
            })?;
        x += step.alpha * d;
    }
    unreachable!()
}
