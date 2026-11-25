//! Levenberg-Marquardt for square and rectangular residual systems.
//!
//! Minimize `f(x)=||r(x)||^2/2` with analytic Jacobian shape `m x n`.
//! Residual count `m` must be positive and constant, and all outputs finite.
//! A tuple `(residuals, jacobian)` implements [`LeastSquares`].
//! Prefer [`levenberg_marquardt_with_options`] for adaptive column-scaled QR
//! steps; [`levenberg_marquardt`] preserves the legacy normal-equation method.
//!
//! ```
//! use nlopt4r::{Matrix, Vector, least_squares::{LmOptions, levenberg_marquardt_with_options}};
//! let problem = (
//!     |x: &Vector| Vector::from_vec(vec![x[0]-1.0, 2.0*x[0]-2.0]),
//!     |_: &Vector| Matrix::from_column_slice(2, 1, &[1.0, 2.0]),
//! );
//! let result = levenberg_marquardt_with_options(
//!     &problem, &Vector::zeros(1), &LmOptions::default(),
//! )?;
//! assert!(result.converged());
//! assert!((result.x[0]-1.0).abs() < 1e-6);
//! # Ok::<(), nlopt4r::Error>(())
//! ```
//!
//! Residual evaluations (including rejected/failed trials) consume objective
//! budget. Jacobian calls count separately as Jacobian and gradient work.
//! Small stationarity does not imply zero residual or a globally best fit.
//! References: Levenberg (1944), Marquardt (1963).
use crate::{
    Error, EvaluationError, Matrix, Options, Solution, StopOptions, TerminationReason, Vector,
    initial, line_search, matrix, solve,
    stopping::{Run, snapshot},
    vector,
};

/// Residual model with a row-oriented analytic Jacobian.
/// Override `try_*` methods for fallible application evaluation.
pub trait LeastSquares {
    /// Nonempty finite residual vector, with constant length throughout a solve.
    fn residuals(&self, x: &Vector) -> Vector;
    /// One row per residual, one column per variable.
    fn jacobian(&self, x: &Vector) -> Matrix;
    /// Fallible residual hook; defaults to [`Self::residuals`].
    fn try_residuals(&self, x: &Vector) -> Result<Vector, EvaluationError> {
        Ok(self.residuals(x))
    }
    /// Fallible Jacobian hook; defaults to [`Self::jacobian`].
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
///
/// Legacy damped normal-equation steps use Armijo backtracking from [`Options`].
/// Shapes and callback failures are checked at every trial. For better numerical
/// conditioning and adaptive damping use [`levenberg_marquardt_with_options`].
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

/// Adaptive LM in column-scaled coordinates, solved by rectangular QR.
#[derive(Debug, Clone)]
pub struct LmOptions {
    /// Shared stopping; gradient tolerance applies to column-scaled stationarity.
    pub stopping: StopOptions,
    /// Positive finite starting damping, default 1e-3; at least the minimum.
    pub initial_damping: f64,
    /// Positive finite damping floor, default 1e-15.
    pub minimum_damping: f64,
    /// Accept if actual/predicted reduction exceeds this finite value in `[0,1)`;
    /// default 1e-4.
    pub acceptance_ratio: f64,
    /// Maximum rejected damped models at each accepted iterate.
    /// Positive, default 40; exhaustion returns an explicit damping-trial error.
    pub max_damping_trials: usize,
}

impl Default for LmOptions {
    fn default() -> Self {
        Self {
            stopping: StopOptions::default(),
            initial_damping: 1e-3,
            minimum_damping: 1e-15,
            acceptance_ratio: 1e-4,
            max_damping_trials: 40,
        }
    }
}

/// Minimize `||r||²/2` without forming `J'J`.
///
/// The reported stationarity is `||D⁻¹ J'r||`, where D contains the
/// nonzero column norms of J (zero columns use one). Residual calls,
/// including rejected trials and failed callbacks, consume the evaluation
/// budget; Jacobian calls are reported separately.
///
/// The augmented QR system solves in scaled coordinates with adaptive damping,
/// supporting rectangular and rank-deficient Jacobians. Storage is dense,
/// O((m+n)*n), not sparse/matrix-free. `x0` must be finite and nonempty.
/// Callback, shape, QR/damping failures preserve the last fully evaluated report
/// when one exists; an exhausted objective budget returns a non-converged checkpoint.
pub fn levenberg_marquardt_with_options<P: LeastSquares + ?Sized>(
    problem: &P,
    x0: &Vector,
    options: &LmOptions,
) -> Result<Solution, Error> {
    options.stopping.validate()?;
    crate::positive(options.initial_damping, "initial damping")?;
    crate::positive(options.minimum_damping, "minimum damping")?;
    if options.initial_damping < options.minimum_damping
        || !options.acceptance_ratio.is_finite()
        || !(0.0..1.0).contains(&options.acceptance_ratio)
        || options.max_damping_trials == 0
    {
        return Err(Error::InvalidInput("LM damping options"));
    }
    initial(x0, &options.stopping)?;
    let mut run = Run::new(&options.stopping)?;
    let result = adaptive_impl(problem, x0, options, &mut run);
    run.finish(result, "adaptive QR Levenberg-Marquardt")
}

fn adaptive_impl<P: LeastSquares + ?Sized>(
    p: &P,
    x0: &Vector,
    options: &LmOptions,
    run: &mut Run,
) -> Result<Solution, Error> {
    let mut x = x0.clone();
    let mut r = residuals(p, &x, run)?;
    let m = r.len();
    if m == 0 {
        return Err(Error::InvalidInput("empty residuals"));
    }
    let n = x.len();
    let mut damping = options.initial_damping;
    for k in 0..=options.stopping.max_iterations {
        vector(&r, m, "residuals")?;
        let value = crate::finite(0.5 * r.norm_squared(), "least-squares objective")?;
        run.counts.jacobian += 1;
        run.counts.gradient += 1;
        let j = p.try_jacobian(&x).map_err(|source| Error::Evaluation {
            operation: "Jacobian",
            source,
        })?;
        matrix(&j, m, n, "Jacobian")?;
        let scales = Vector::from_iterator(
            n,
            j.column_iter().map(|c| {
                let norm = c.iter().fold(0.0_f64, |norm, value| norm.hypot(*value));
                if norm > 0.0 { norm } else { 1.0 }
            }),
        );
        vector(&scales, n, "Jacobian column scales")?;
        let mut scaled = j.clone();
        for i in 0..n {
            for row in 0..m {
                scaled[(row, i)] /= scales[i];
            }
        }
        let gradient = scaled.transpose() * &r;
        let norm = crate::finite(gradient.norm(), "scaled least-squares stationarity")?;
        let reason = run.reason(
            &x,
            value,
            k,
            norm <= options.stopping.gradient_tolerance,
            TerminationReason::GradientTolerance,
            k > 0,
        );
        let current = run.record(snapshot(x.clone(), value, norm, k, reason));
        if reason.is_some() {
            return Ok(current);
        }
        let mut growth = 2.0;
        let mut accepted = false;
        for _ in 0..options.max_damping_trials {
            let mut a = Matrix::zeros(m + n, n);
            a.rows_mut(0, m).copy_from(&scaled);
            for i in 0..n {
                a[(m + i, i)] = damping.sqrt();
            }
            let mut rhs = Vector::zeros(m + n);
            rhs.rows_mut(0, m).copy_from(&(-&r));
            let qr = a.qr();
            qr.q_tr_mul(&mut rhs);
            let z = qr
                .r()
                .solve_upper_triangular(&rhs.rows(0, n).into_owned())
                .ok_or(Error::SingularSystem)?;
            vector(&z, n, "LM scaled step")?;
            let d = z.component_div(&scales);
            let trial = &x + &d;
            vector(&trial, n, "LM trial")?;
            let trial_r = residuals(p, &trial, run)?;
            vector(&trial_r, m, "residuals")?;
            let trial_value =
                crate::finite(0.5 * trial_r.norm_squared(), "least-squares objective")?;
            let model_change = &scaled * &z;
            let predicted = -gradient.dot(&z) - 0.5 * model_change.norm_squared();
            let ratio = if predicted > 0.0 {
                (value - trial_value) / predicted
            } else {
                -1.0
            };
            if ratio > options.acceptance_ratio && trial_value < value {
                x = trial;
                r = trial_r;
                damping = (damping * (1.0 - (2.0 * ratio - 1.0).powi(3)).max(1.0 / 3.0))
                    .max(options.minimum_damping);
                accepted = true;
                break;
            }
            damping = crate::finite(damping * growth, "LM damping")?;
            growth *= 2.0;
        }
        if !accepted {
            return Err(Error::IterationLimit {
                solver: "LM damping trials",
                iterations: options.max_damping_trials,
            });
        }
    }
    unreachable!()
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
