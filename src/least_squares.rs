//! Levenberg-Marquardt for square and rectangular residual systems.
use crate::{
    Error, EvaluationError, Matrix, Options, Solution, TerminationReason, Vector, initial,
    line_search, matrix, solve,
    stopping::{Run, snapshot},
    vector,
};

pub trait LeastSquares {
    fn residuals(&self, x: &Vector) -> Vector;
    /// One row per residual, one column per variable.
    fn jacobian(&self, x: &Vector) -> Matrix;
    fn try_residuals(&self, x: &Vector) -> Result<Vector, EvaluationError> {
        Ok(self.residuals(x))
    }
    fn try_jacobian(&self, x: &Vector) -> Result<Matrix, EvaluationError> {
        Ok(self.jacobian(x))
    }
}

impl<F, J> LeastSquares for (F, J)
where
    F: Fn(&Vector) -> Vector,
    J: Fn(&Vector) -> Matrix,
{
    fn residuals(&self, x: &Vector) -> Vector {
        self.0(x)
    }
    fn jacobian(&self, x: &Vector) -> Matrix {
        self.1(x)
    }
}

/// Minimize half the squared residual norm. Convergence means J'r is small,
/// not necessarily that the residual is zero.
pub fn levenberg_marquardt<P: LeastSquares + ?Sized>(
    problem: &P,
    x0: &Vector,
    options: &Options,
) -> Result<Solution, Error> {
    options.validate()?;
    initial(x0, &options.stopping)?;
    let mut run = Run::new(&options.stopping)?;
    let result = minimize_impl(problem, x0, options, &mut run);
    run.finish(result, "Levenberg-Marquardt")
}

fn residuals<P: LeastSquares + ?Sized>(p: &P, x: &Vector, run: &mut Run) -> Result<Vector, Error> {
    vector(x, x.len(), "least-squares iterate")?;
    run.charge()?;
    run.counts.residuals += 1;
    p.try_residuals(x).map_err(|source| Error::Evaluation {
        operation: "residuals",
        source,
    })
}

fn minimize_impl<P: LeastSquares + ?Sized>(
    problem: &P,
    x0: &Vector,
    options: &Options,
    run: &mut Run,
) -> Result<Solution, Error> {
    let mut x = x0.clone();
    let mut residual_count = None;
    for k in 0..=options.stopping.max_iterations {
        let r = residuals(problem, &x, run)?;
        let m = *residual_count.get_or_insert(r.len());
        if m == 0 {
            return Err(Error::InvalidInput("empty residuals"));
        }
        vector(&r, m, "residuals")?;
        run.counts.jacobian += 1;
        run.counts.gradient += 1;
        let j = problem
            .try_jacobian(&x)
            .map_err(|source| Error::Evaluation {
                operation: "Jacobian",
                source,
            })?;
        matrix(&j, m, x.len(), "Jacobian")?;
        let g = j.transpose() * &r;
        let value = crate::finite(0.5 * r.norm_squared(), "least-squares objective")?;
        let norm = crate::finite(g.norm(), "least-squares stationarity")?;
        let reason = run.reason(
            &x,
            value,
            k,
            norm <= options.stopping.gradient_tolerance,
            TerminationReason::GradientTolerance,
            k > 0,
        );
        let report = run.record(snapshot(x.clone(), value, norm, k, reason));
        if reason.is_some() {
            return Ok(report);
        }
        let mu = r.norm().max(1e-12);
        let a = j.transpose() * j + mu * Matrix::identity(x.len(), x.len());
        let d = solve(&a, &(-&g))?;
        // Validate the residual shape on every trial before computing its merit.
        let step = line_search::backtrack(value, g.dot(&d), &options.line_search, |alpha| {
            let trial = &x + alpha * &d;
            vector(&trial, x.len(), "least-squares iterate")?;
            let r = residuals(problem, &trial, run)?;
            vector(&r, m, "residuals")?;
            crate::finite(0.5 * r.norm_squared(), "least-squares objective")
        })?;
        x += step.alpha * d;
    }
    unreachable!()
}
