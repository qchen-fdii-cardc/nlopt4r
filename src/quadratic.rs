//! Convex dense quadratic programming with a feasible active set and phase I.
//! All inequalities use A_i x >= b_i; multipliers use L = f - lambda' (Ax-b).
use crate::{
    Error, Matrix, Options, Solution, Status, Vector, finite, matrix, solution, symmetric, vector,
};

pub struct QuadraticProgram {
    pub hessian: Matrix,
    pub linear: Vector,
    pub equalities: Matrix,
    pub equality_rhs: Vector,
    pub inequalities: Matrix,
    pub inequality_rhs: Vector,
}

#[derive(Debug, Clone)]
pub struct QuadraticSolution {
    pub solution: Solution,
    pub equality_multipliers: Vector,
    pub inequality_multipliers: Vector,
    pub constraint_violation: f64,
}

impl QuadraticProgram {
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
    report(&qp, x, multipliers, Vector::zeros(0), 1, Status::Converged)
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
    options: &Options,
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
        let a = working_matrix(qp, &active);
        let g = &qp.hessian * &x + &qp.linear;
        let (d, lam) = kkt(&qp.hessian, &g, &a, &Vector::zeros(a.nrows()))?;
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
        if stationary && negative.is_none() && violation(qp, &x) <= tol {
            return report(qp, x, mu, lambda, k, Status::Converged);
        }
        if k == options.max_iterations {
            return report(qp, x, mu, lambda, k, Status::MaxIterations);
        }
        if stationary && negative.is_some() {
            active.remove(negative.expect("negative multiplier checked").0);
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
/// finds feasibility. Phase I and optimization each have the supplied budget.
pub fn minimize(
    qp: &QuadraticProgram,
    x0: &Vector,
    options: &Options,
) -> Result<QuadraticSolution, Error> {
    qp.validate()?;
    crate::initial(x0, options)?;
    vector(x0, qp.linear.len(), "QP initial point")?;
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
        let result = active_set(&phase, start, options)?;
        if !result.solution.converged() {
            return Err(Error::IterationLimit("QP phase I"));
        }
        if result.solution.x[n] > options.gradient_tolerance {
            return Err(Error::Infeasible);
        }
        x = result.solution.x.rows(0, n).into_owned();
    }
    active_set(qp, x, options)
}
