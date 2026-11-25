//! Convex dense quadratic programming with a feasible active set and phase I.
//! All inequalities use A_i x >= b_i; multipliers use L = f - lambda' (Ax-b).
//!
//! Minimize `x'Hx/2 + c'x` with `Ae*x=be`, `Ai*x>=bi` and a symmetric
//! positive-semidefinite `H`. Positive-definite well-scaled models are preferred.
//! An active set handles inequalities; phase I permits infeasible initial points.
//! Redundant consistent constraints use a checked least-norm KKT fallback.
//! This is not a general LP/unboundedness-certification solver.
//!
//! ```
//! use nlopt4r::{Matrix, Vector, StopOptions, quadratic::{QuadraticProgram, minimize}};
//! let mut qp = QuadraticProgram::unconstrained(
//!     Matrix::identity(2, 2), Vector::from_vec(vec![-2.0, -4.0]),
//! );
//! qp.equalities = Matrix::from_row_slice(1, 2, &[1.0, 1.0]);
//! qp.equality_rhs = Vector::from_vec(vec![3.0]);
//! let result = minimize(&qp, &Vector::zeros(2), &StopOptions::default())?;
//! assert!(result.solution.converged());
//! assert!((result.solution.x.sum()-3.0).abs() < 1e-6);
//! # Ok::<(), nlopt4r::Error>(())
//! ```
use crate::{
    Error, Matrix, Solution, Status, StopOptions, TerminationReason, Vector, finite, matrix,
    solution,
    stopping::{Report, Run},
    symmetric, vector,
};

/// Dense convex QP data, validated on solver entry rather than construction.
///
/// All arrays must be finite. For `n` variables, absent constraint matrices must
/// have shape `0 x n`, not `0 x 0`. No bounds are implied unless supplied as rows.
pub struct QuadraticProgram {
    /// Symmetric positive-semidefinite `n x n` matrix H in `x'Hx/2`.
    pub hessian: Matrix,
    /// Length-n linear coefficient c.
    pub linear: Vector,
    /// Equality rows Ae, shape `equality_rhs.len() x n`.
    pub equalities: Matrix,
    /// Equality right-hand side be.
    pub equality_rhs: Vector,
    /// Inequality rows Ai, shape `inequality_rhs.len() x n`, using `Ai*x>=bi`.
    pub inequalities: Matrix,
    /// Inequality right-hand side bi.
    pub inequality_rhs: Vector,
}

/// QP state and canonical equality/inequality multipliers.
#[derive(Debug, Clone)]
pub struct QuadraticSolution {
    /// Iterate/value/stationarity report; inspect convergence before use.
    pub solution: Solution,
    /// Equality multipliers in `L=f-mu'(Ae*x-be)-lambda'(Ai*x-bi)`.
    pub equality_multipliers: Vector,
    /// Nonnegative active inequality multipliers; inactive rows have zero multiplier.
    pub inequality_multipliers: Vector,
    /// Maximum equality residual or negative-part inequality residual.
    pub constraint_violation: f64,
}
impl Report for QuadraticSolution {
    fn solution(&self) -> &Solution {
        &self.solution
    }
    fn solution_mut(&mut self) -> &mut Solution {
        &mut self.solution
    }
}

impl QuadraticProgram {
    /// Create a model with empty constraint arrays inferred from `linear.len()`.
    /// Dimensions/finiteness/convexity are checked later by the solver.
    pub fn unconstrained(hessian: Matrix, linear: Vector) -> Self {
        let n = linear.len();
        Self {
            hessian,
            linear,
            equalities: Matrix::zeros(0, n),
            equality_rhs: Vector::zeros(0),
            inequalities: Matrix::zeros(0, n),
            inequality_rhs: Vector::zeros(0),
        }
    }

    fn validate(&self) -> Result<(), Error> {
        let n = self.linear.len();
        if n == 0 {
            return Err(Error::InvalidInput("empty quadratic program"));
        }
        vector(&self.linear, n, "linear objective")?;
        symmetric(&self.hessian, n)?;
        matrix(&self.equalities, self.equality_rhs.len(), n, "equalities")?;
        matrix(
            &self.inequalities,
            self.inequality_rhs.len(),
            n,
            "inequalities",
        )?;
        vector(&self.equality_rhs, self.equality_rhs.len(), "equality rhs")?;
        vector(
            &self.inequality_rhs,
            self.inequality_rhs.len(),
            "inequality rhs",
        )?;
        let eigen = self.hessian.clone().symmetric_eigen().eigenvalues;
        if eigen.min() < -1e-10 * (1.0 + self.hessian.norm()) {
            return Err(Error::InvalidInput(
                "quadratic Hessian must be positive semidefinite",
            ));
        }
        Ok(())
    }
}

fn least_norm(a: &Matrix, b: &Vector) -> Result<Vector, Error> {
    if a.is_square()
        && let Some(x) = a.clone().full_piv_lu().solve(b)
        && x.iter().all(|v| v.is_finite())
        && (a * &x - b).norm() <= 1e-10 * (1.0 + b.norm())
    {
        return Ok(x);
    }
    let tolerance = 1e-12 * a.norm().max(1.0);
    let x = a
        .clone()
        .svd(true, true)
        .solve(b, tolerance)
        .map_err(|_| Error::SingularSystem)?;
    vector(&x, a.ncols(), "KKT solution")?;
    if (a * &x - b).norm() > 1e-7 * (1.0 + b.norm()) {
        return Err(Error::SingularSystem);
    }
    Ok(x)
}

/// Solve an equality-constrained quadratic KKT system, including consistent
/// rank-deficient systems. An unbounded/unsolvable system returns an error.
///
/// Solves `min x'Hx/2+c'x` subject to `a*x=b`; `H` must be symmetric PSD.
/// The returned equality multipliers satisfy `H*x+c-a'*mu=0`. This direct
/// operation has no iterative objective allowance and reports one model state.
/// Inconsistent constraints or an unacceptable solve return explicit errors.
pub fn equality_constrained(
    hessian: &Matrix,
    linear: &Vector,
    a: &Matrix,
    b: &Vector,
) -> Result<QuadraticSolution, Error> {
    let qp = QuadraticProgram {
        hessian: hessian.clone(),
        linear: linear.clone(),
        equalities: a.clone(),
        equality_rhs: b.clone(),
        inequalities: Matrix::zeros(0, linear.len()),
        inequality_rhs: Vector::zeros(0),
    };
    qp.validate()?;
    let (x, multipliers) = kkt(hessian, linear, a, b)?;
    let mut result = report(&qp, x, multipliers, Vector::zeros(0), 1, Status::Converged)?;
    result.solution.terminate(TerminationReason::KktTolerance);
    result.solution.evaluations.objective = 1;
    result.solution.evaluations.gradient = 1;
    result.solution.evaluations.constraints = 1;
    Ok(result)
}

pub(crate) fn kkt(
    h: &Matrix,
    g: &Vector,
    a: &Matrix,
    b: &Vector,
) -> Result<(Vector, Vector), Error> {
    let n = g.len();
    let m = b.len();
    let mut system = Matrix::zeros(n + m, n + m);
    system.view_mut((0, 0), (n, n)).copy_from(h);
    system.view_mut((0, n), (n, m)).copy_from(&(-a.transpose()));
    system.view_mut((n, 0), (m, n)).copy_from(a);
    let mut rhs = Vector::zeros(n + m);
    rhs.rows_mut(0, n).copy_from(&(-g));
    rhs.rows_mut(n, m).copy_from(b);
    let z = least_norm(&system, &rhs)?;
    Ok((z.rows(0, n).into_owned(), z.rows(n, m).into_owned()))
}

fn violation(qp: &QuadraticProgram, x: &Vector) -> f64 {
    let e = &qp.equalities * x - &qp.equality_rhs;
    let i = &qp.inequalities * x - &qp.inequality_rhs;
    e.iter()
        .map(|v| v.abs())
        .chain(i.iter().map(|v| (-v).max(0.0)))
        .fold(0.0, f64::max)
}

fn report(
    qp: &QuadraticProgram,
    x: Vector,
    mu: Vector,
    lambda: Vector,
    iterations: usize,
    status: Status,
) -> Result<QuadraticSolution, Error> {
    let gradient = &qp.hessian * &x + &qp.linear
        - qp.equalities.transpose() * &mu
        - qp.inequalities.transpose() * &lambda;
    let value = finite(
        0.5 * x.dot(&(&qp.hessian * &x)) + qp.linear.dot(&x),
        "quadratic objective",
    )?;
    let constraint_violation = violation(qp, &x);
    Ok(QuadraticSolution {
        solution: solution(
            x,
            value,
            gradient.norm(),
            iterations,
            status == Status::Converged,
        ),
        equality_multipliers: mu,
        inequality_multipliers: lambda,
        constraint_violation,
    })
}

fn independent(a: &Matrix, row: &Vector) -> bool {
    let mut basis: Vec<Vector> = Vec::new();
    for r in a.row_iter() {
        let mut v = r.transpose().into_owned();
        for q in &basis {
            v -= q.dot(&v) * q;
        }
        if v.norm() > 1e-10 * r.norm().max(1.0) {
            basis.push(v.normalize());
        }
    }
    let mut v = row.clone();
    for q in &basis {
        v -= q.dot(&v) * q;
    }
    v.norm() > 1e-10 * row.norm().max(1.0)
}

fn working_matrix(qp: &QuadraticProgram, active: &[usize]) -> Matrix {
    let ne = qp.equality_rhs.len();
    let mut a = Matrix::zeros(ne + active.len(), qp.linear.len());
    a.rows_mut(0, ne).copy_from(&qp.equalities);
    for (j, &i) in active.iter().enumerate() {
        a.row_mut(ne + j).copy_from(&qp.inequalities.row(i));
    }
    a
}

fn active_set(
    qp: &QuadraticProgram,
    mut x: Vector,
    options: &StopOptions,
    run: &mut Run<QuadraticSolution>,
) -> Result<QuadraticSolution, Error> {
    let tol = options.gradient_tolerance;
    let ne = qp.equality_rhs.len();
    let ni = qp.inequality_rhs.len();
    let mut active = Vec::new();
    let slack = &qp.inequalities * &x - &qp.inequality_rhs;
    for i in 0..ni {
        if slack[i].abs() <= tol
            && independent(
                &working_matrix(qp, &active),
                &qp.inequalities.row(i).transpose().into_owned(),
            )
        {
            active.push(i);
        }
    }
    for k in 0..=options.max_iterations {
        run.charge()?;
        run.counts.gradient += 1;
        run.counts.constraints += 1;
        let accepted = run.last.as_ref().is_some_and(|last| last.solution.x != x);
        let mut current = report(
            qp,
            x.clone(),
            Vector::zeros(ne),
            Vector::zeros(ni),
            k,
            Status::MaxIterations,
        )?;
        run.checkpoint(current.clone());
        let a = working_matrix(qp, &active);
        let g = &qp.hessian * &x + &qp.linear;
        let mut rhs = Vector::zeros(a.nrows());
        rhs.rows_mut(0, ne).copy_from(&qp.equality_rhs);
        for (j, &i) in active.iter().enumerate() {
            rhs[ne + j] = qp.inequality_rhs[i];
        }
        rhs -= &a * &x;
        let (d, lam) = kkt(&qp.hessian, &g, &a, &rhs)?;
        let mu = lam.rows(0, ne).into_owned();
        let mut lambda = Vector::zeros(ni);
        for (j, &i) in active.iter().enumerate() {
            lambda[i] = lam[ne + j];
        }
        let negative = active
            .iter()
            .enumerate()
            .filter(|(j, _)| lam[ne + j] < -tol)
            .min_by(|(j, _), (l, _)| lam[ne + j].total_cmp(&lam[ne + l]));
        let stationary = (&g - a.transpose() * &lam).norm() <= tol;
        let converged = stationary && negative.is_none() && violation(qp, &x) <= tol;
        current.equality_multipliers = mu;
        current.inequality_multipliers = lambda;
        current.solution.gradient_norm =
            finite((&g - a.transpose() * &lam).norm(), "QP stationarity")?;
        let reason = run.reason(
            &x,
            current.solution.value,
            k,
            converged,
            TerminationReason::KktTolerance,
            accepted,
        );
        if let Some(reason) = reason {
            current.solution.terminate(reason);
        }
        let current = run.record(current);
        if reason.is_some() {
            return Ok(current);
        }
        if stationary && let Some((index, _)) = negative {
            active.remove(index);
            continue;
        }
        let slack = &qp.inequalities * &x - &qp.inequality_rhs;
        let ad = &qp.inequalities * &d;
        let mut alpha = 1.0;
        let mut blocking = None;
        for i in 0..ni {
            if !active.contains(&i) && ad[i] < -1e-12 {
                let candidate = (slack[i] / -ad[i]).max(0.0);
                if candidate < alpha
                    && independent(&a, &qp.inequalities.row(i).transpose().into_owned())
                {
                    alpha = candidate;
                    blocking = Some(i);
                }
            }
        }
        x += alpha * d;
        if let Some(i) = blocking {
            active.push(i);
        }
    }
    unreachable!()
}

/// Solve a convex QP. The initial point need not be feasible; phase I first
/// finds feasibility. Each phase has its own `max_iterations` allowance, but
/// both share the strict `max_evaluations` limit.
///
/// Input data and `x0` must have the same nonzero variable dimension. Counts
/// describe checked model states, not external callbacks. Outer budget stops
/// return a non-converged original-dimension checkpoint; failed phase-I or KKT
/// solves remain errors. The returned iteration count covers the main solve.
pub fn minimize(
    qp: &QuadraticProgram,
    x0: &Vector,
    options: &StopOptions,
) -> Result<QuadraticSolution, Error> {
    qp.validate()?;
    crate::initial(x0, options)?;
    vector(x0, qp.linear.len(), "QP initial point")?;
    let mut run = Run::new(options)?;
    let result = minimize_impl(qp, x0, options, &mut run);
    run.finish(result, "quadratic programming")
}

fn minimize_impl(
    qp: &QuadraticProgram,
    x0: &Vector,
    options: &StopOptions,
    run: &mut Run<QuadraticSolution>,
) -> Result<QuadraticSolution, Error> {
    run.charge()?;
    run.counts.gradient += 1;
    run.counts.constraints += 1;
    let mut initial = report(
        qp,
        x0.clone(),
        Vector::zeros(qp.equality_rhs.len()),
        Vector::zeros(qp.inequality_rhs.len()),
        0,
        Status::MaxIterations,
    )?;
    if initial.solution.gradient_norm <= options.gradient_tolerance
        && initial.constraint_violation <= options.gradient_tolerance
    {
        initial.solution.terminate(TerminationReason::KktTolerance);
        return Ok(run.record(initial));
    }
    run.record(initial);
    let n = x0.len();
    let mut x = x0.clone();
    if qp.equalities.nrows() > 0 {
        let residual = &qp.equality_rhs - &qp.equalities * &x;
        let correction = least_norm(&qp.equalities, &residual).map_err(|e| {
            if e == Error::SingularSystem {
                Error::Infeasible
            } else {
                e
            }
        })?;
        x += correction;
    }
    if violation(qp, &x) > options.gradient_tolerance {
        let ni = qp.inequality_rhs.len();
        let ne = qp.equality_rhs.len();
        let mut phase =
            QuadraticProgram::unconstrained(Matrix::zeros(n + 1, n + 1), Vector::zeros(n + 1));
        phase.hessian[(n, n)] = 1.0;
        phase.linear[n] = 1.0;
        phase.equalities = Matrix::zeros(ne, n + 1);
        phase.equalities.columns_mut(0, n).copy_from(&qp.equalities);
        phase.equality_rhs = qp.equality_rhs.clone();
        phase.inequalities = Matrix::zeros(ni + 1, n + 1);
        phase
            .inequalities
            .view_mut((0, 0), (ni, n))
            .copy_from(&qp.inequalities);
        for i in 0..=ni {
            phase.inequalities[(i, n)] = 1.0;
        }
        phase.inequality_rhs = Vector::zeros(ni + 1);
        phase
            .inequality_rhs
            .rows_mut(0, ni)
            .copy_from(&qp.inequality_rhs);
        let mut start = Vector::zeros(n + 1);
        start.rows_mut(0, n).copy_from(&x);
        start[n] = violation(qp, &x) + 1.0;
        let phase_options = StopOptions {
            max_evaluations: options
                .max_evaluations
                .map(|n| n.saturating_sub(run.counts.objective)),
            ..options.clone()
        };
        if phase_options.max_evaluations == Some(0) {
            return Err(Error::EvaluationBudgetExceeded);
        }
        let mut phase_run = Run::new(&phase_options)?;
        let phase_result = active_set(&phase, start, &phase_options, &mut phase_run);
        let phase_counts = phase_run.counts;
        let result = phase_run.finish(phase_result, "QP phase I");
        run.counts += phase_counts;
        let result = result?;
        if result.solution.status == Status::MaxEvaluations {
            return Err(Error::EvaluationBudgetExceeded);
        }
        if !result.solution.converged() {
            return Err(Error::SubproblemStopped {
                solver: "QP phase I",
                reason: result.solution.termination_reason,
                iterations: result.solution.iterations,
            });
        }
        if result.solution.x[n] > options.gradient_tolerance {
            return Err(Error::Infeasible);
        }
        x = result.solution.x.rows(0, n).into_owned();
    }
    active_set(qp, x, options, run)
}
