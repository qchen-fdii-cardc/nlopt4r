//! Nonlinear constraints h(x)=0, g(x)>=0, using L=f-mu'h-lambda'g.
//! Jacobians have one row per constraint, unlike the transposed PHR MATLAB inputs.
use crate::{
    Error, Matrix, Objective, Options, Solution, Vector, evaluate, finite, initial, line_search,
    matrix, positive, quadratic, solution, symmetric, vector,
};

pub struct Constraints {
    pub equalities: Vector,
    pub inequalities: Vector,
    pub equality_jacobian: Matrix,
    pub inequality_jacobian: Matrix,
}

impl Constraints {
    pub fn empty(n: usize) -> Self {
        Self {
            equalities: Vector::zeros(0),
            inequalities: Vector::zeros(0),
            equality_jacobian: Matrix::zeros(0, n),
            inequality_jacobian: Matrix::zeros(0, n),
        }
    }
}

pub trait ConstrainedProblem: Objective {
    fn constraints(&self, x: &Vector) -> Constraints;
    /// Hessian of f-mu'h-lambda'g. Required by the two exact-Hessian methods.
    fn lagrangian_hessian(&self, _x: &Vector, _mu: &Vector, _lambda: &Vector) -> Option<Matrix> {
        None
    }
}

#[derive(Debug, Clone)]
pub struct ConstrainedOptions {
    pub solver: Options,
    pub feasibility_tolerance: f64,
    /// Per-outer-iteration budget for PHR's BFGS subproblem.
    pub max_inner_iterations: usize,
    pub initial_penalty: f64,
}

impl Default for ConstrainedOptions {
    fn default() -> Self {
        Self {
            solver: Options::default(),
            feasibility_tolerance: 1e-6,
            max_inner_iterations: 1000,
            initial_penalty: 2.0,
        }
    }
}

impl ConstrainedOptions {
    fn validate(&self) -> Result<(), Error> {
        self.solver.validate()?;
        positive(self.feasibility_tolerance, "feasibility tolerance")?;
        positive(self.initial_penalty, "initial penalty")?;
        if self.max_inner_iterations == 0 {
            return Err(Error::InvalidInput("inner iteration limit"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ConstrainedSolution {
    pub solution: Solution,
    pub equality_multipliers: Vector,
    pub inequality_multipliers: Vector,
    pub constraint_violation: f64,
    pub complementarity: f64,
    pub inner_iterations: usize,
}

struct State {
    f: f64,
    gradient: Vector,
    constraints: Constraints,
}

fn state<P: ConstrainedProblem + ?Sized>(
    p: &P,
    x: &Vector,
    counts: Option<(usize, usize)>,
) -> Result<State, Error> {
    let (f, gradient) = evaluate(p, x)?;
    let c = p.constraints(x);
    let (ne, ni) = counts.unwrap_or((c.equalities.len(), c.inequalities.len()));
    vector(&c.equalities, ne, "equalities")?;
    vector(&c.inequalities, ni, "inequalities")?;
    matrix(&c.equality_jacobian, ne, x.len(), "equality Jacobian")?;
    matrix(&c.inequality_jacobian, ni, x.len(), "inequality Jacobian")?;
    Ok(State {
        f,
        gradient,
        constraints: c,
    })
}

fn lag_gradient(s: &State, mu: &Vector, lambda: &Vector) -> Vector {
    &s.gradient
        - s.constraints.equality_jacobian.transpose() * mu
        - s.constraints.inequality_jacobian.transpose() * lambda
}

fn violation(c: &Constraints) -> f64 {
    c.equalities
        .iter()
        .map(|v| v.abs())
        .chain(c.inequalities.iter().map(|v| (-v).max(0.0)))
        .fold(0.0, f64::max)
}

fn l1(c: &Constraints) -> f64 {
    c.equalities
        .iter()
        .map(|v| v.abs())
        .chain(c.inequalities.iter().map(|v| (-v).max(0.0)))
        .sum()
}

fn complementarity(c: &Constraints, lambda: &Vector) -> f64 {
    c.inequalities
        .iter()
        .zip(lambda.iter())
        .map(|(&g, &l)| g.min(l).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn done(s: &State, mu: &Vector, lambda: &Vector, options: &ConstrainedOptions) -> bool {
    lag_gradient(s, mu, lambda).norm() <= options.solver.gradient_tolerance
        && violation(&s.constraints) <= options.feasibility_tolerance
        && complementarity(&s.constraints, lambda) <= options.feasibility_tolerance
}

fn report(
    x: Vector,
    s: &State,
    mu: Vector,
    lambda: Vector,
    iterations: usize,
    inner_iterations: usize,
    converged: bool,
) -> ConstrainedSolution {
    ConstrainedSolution {
        solution: solution(
            x,
            s.f,
            lag_gradient(s, &mu, &lambda).norm(),
            iterations,
            converged,
        ),
        constraint_violation: violation(&s.constraints),
        complementarity: complementarity(&s.constraints, &lambda),
        equality_multipliers: mu,
        inequality_multipliers: lambda,
        inner_iterations,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Powell-damped BFGS SQP with an l1 exact-penalty merit function.
    Sqp,
    /// Equality-constrained Newton method on the KKT equations.
    NewtonLagrange,
    /// Equality SQP using the stabilized exact Lagrangian Hessian.
    LagrangeSqp,
}

pub fn minimize<P: ConstrainedProblem + ?Sized>(
    p: &P,
    x0: &Vector,
    method: Method,
    options: &ConstrainedOptions,
) -> Result<ConstrainedSolution, Error> {
    options.validate()?;
    initial(x0, &options.solver)?;
    let first = state(p, x0, None)?;
    let counts = (
        first.constraints.equalities.len(),
        first.constraints.inequalities.len(),
    );
    if method != Method::Sqp && counts.1 > 0 {
        return Err(Error::InvalidInput(
            "Newton-Lagrange and Lagrangian SQP require equality constraints",
        ));
    }
    let mut x = x0.clone();
    let n = x.len();
    let mut mu = Vector::zeros(counts.0);
    let mut lambda = Vector::zeros(counts.1);
    let mut b = Matrix::identity(n, n);
    let mut penalty: f64 = 1.0;
    for k in 0..=options.solver.max_iterations {
        let s = state(p, &x, Some(counts))?;
        let converged = done(&s, &mu, &lambda, options);
        if converged || k == options.solver.max_iterations {
            return Ok(report(x, &s, mu, lambda, k, 0, converged));
        }
        if method != Method::Sqp {
            let mut h = p
                .lagrangian_hessian(&x, &mu, &lambda)
                .ok_or(Error::MissingHessian)?;
            symmetric(&h, n)?;
            let a = &s.constraints.equality_jacobian;
            let stabilization = if method == Method::LagrangeSqp {
                1.0 / 3.1
            } else {
                0.0
            };
            h += stabilization * a.transpose() * a;
            let (d, new_mu) = quadratic::kkt(&h, &s.gradient, a, &(-&s.constraints.equalities))?;
            let dmu = &new_mu - &mu - stabilization * a * &d;
            let old_merit = lag_gradient(&s, &mu, &lambda).norm_squared()
                + s.constraints.equalities.norm_squared();
            let step = line_search::backtrack(
                old_merit,
                -old_merit,
                &options.solver.line_search,
                |alpha| {
                    let next = state(p, &(&x + alpha * &d), Some(counts))?;
                    let grad = lag_gradient(&next, &(&mu + alpha * &dmu), &lambda);
                    finite(
                        grad.norm_squared() + next.constraints.equalities.norm_squared(),
                        "KKT merit",
                    )
                },
            )?;
            x += step.alpha * d;
            mu += step.alpha * dmu;
            continue;
        }
        let qp = quadratic::QuadraticProgram {
            hessian: b.clone(),
            linear: s.gradient.clone(),
            equalities: s.constraints.equality_jacobian.clone(),
            equality_rhs: -&s.constraints.equalities,
            inequalities: s.constraints.inequality_jacobian.clone(),
            inequality_rhs: -&s.constraints.inequalities,
        };
        let qp_options = Options {
            gradient_tolerance: (0.01 * options.solver.gradient_tolerance).min(1e-8),
            ..options.solver.clone()
        };
        let q = quadratic::minimize(&qp, &Vector::zeros(n), &qp_options)?;
        if !q.solution.converged() {
            return Err(Error::IterationLimit("SQP quadratic subproblem"));
        }
        if done(
            &s,
            &q.equality_multipliers,
            &q.inequality_multipliers,
            options,
        ) {
            return Ok(report(
                x,
                &s,
                q.equality_multipliers,
                q.inequality_multipliers,
                k,
                0,
                true,
            ));
        }
        let d = q.solution.x;
        penalty = penalty.max(
            q.equality_multipliers
                .amax()
                .max(q.inequality_multipliers.amax())
                + 1.0,
        );
        let merit = s.f + penalty * l1(&s.constraints);
        let slope = s.gradient.dot(&d) - penalty * l1(&s.constraints);
        let step = line_search::backtrack(merit, slope, &options.solver.line_search, |alpha| {
            let next = state(p, &(&x + alpha * &d), Some(counts))?;
            finite(next.f + penalty * l1(&next.constraints), "SQP merit")
        })?;
        let next_x = &x + step.alpha * d;
        let next = state(p, &next_x, Some(counts))?;
        mu = (1.0 - step.alpha) * mu + step.alpha * q.equality_multipliers;
        lambda = (1.0 - step.alpha) * lambda + step.alpha * q.inequality_multipliers;
        let sk = &next_x - &x;
        let y = lag_gradient(&next, &mu, &lambda) - lag_gradient(&s, &mu, &lambda);
        let bs = &b * &sk;
        let sbs = sk.dot(&bs);
        let sy = sk.dot(&y);
        if sbs > 0.0 {
            let theta = if sy >= 0.2 * sbs {
                1.0
            } else {
                0.8 * sbs / (sbs - sy)
            };
            let z = theta * y + (1.0 - theta) * &bs;
            b += &z * z.transpose() / sk.dot(&z) - &bs * bs.transpose() / sbs;
            b = 0.5 * (&b + b.transpose());
        }
        x = next_x;
    }
    unreachable!()
}

fn augmented(
    s: &State,
    mu: &Vector,
    lambda: &Vector,
    penalty: f64,
) -> Result<(f64, Vector), Error> {
    let c = &s.constraints;
    let eq_weight = penalty * &c.equalities - mu;
    let projected = (lambda - penalty * &c.inequalities).map(|v| v.max(0.0));
    let value = s.f - mu.dot(&c.equalities)
        + 0.5 * penalty * c.equalities.norm_squared()
        + (projected.norm_squared() - lambda.norm_squared()) / (2.0 * penalty);
    let gradient = &s.gradient + c.equality_jacobian.transpose() * eq_weight
        - c.inequality_jacobian.transpose() * projected;
    finite(value, "augmented objective")?;
    vector(&gradient, s.gradient.len(), "augmented gradient")?;
    Ok((value, gradient))
}

/// Powell-Hestenes-Rockafellar multiplier method with BFGS inner minimization.
pub fn augmented_lagrangian<P: ConstrainedProblem + ?Sized>(
    p: &P,
    x0: &Vector,
    options: &ConstrainedOptions,
) -> Result<ConstrainedSolution, Error> {
    options.validate()?;
    initial(x0, &options.solver)?;
    let first = state(p, x0, None)?;
    let counts = (
        first.constraints.equalities.len(),
        first.constraints.inequalities.len(),
    );
    let mut x = x0.clone();
    let mut mu = Vector::zeros(counts.0);
    let mut lambda = Vector::zeros(counts.1);
    let mut penalty = options.initial_penalty;
    let mut previous_residual = f64::INFINITY;
    let mut inner_iterations = 0;
    for k in 0..=options.solver.max_iterations {
        let s = state(p, &x, Some(counts))?;
        let converged = done(&s, &mu, &lambda, options);
        if converged || k == options.solver.max_iterations {
            return Ok(report(x, &s, mu, lambda, k, inner_iterations, converged));
        }
        let mut inverse = Matrix::identity(x.len(), x.len());
        let inner_tolerance = (0.1 * options.solver.gradient_tolerance).max(1e-12);
        for j in 0..=options.max_inner_iterations {
            let current = state(p, &x, Some(counts))?;
            let (f, g) = augmented(&current, &mu, &lambda, penalty)?;
            if g.norm() <= inner_tolerance {
                break;
            }
            if j == options.max_inner_iterations {
                return Err(Error::IterationLimit("PHR inner BFGS"));
            }
            let mut d = -&inverse * &g;
            if g.dot(&d) >= 0.0 {
                d = -&g;
                inverse = Matrix::identity(x.len(), x.len());
            }
            let step =
                line_search::backtrack(f, g.dot(&d), &options.solver.line_search, |alpha| {
                    augmented(
                        &state(p, &(&x + alpha * &d), Some(counts))?,
                        &mu,
                        &lambda,
                        penalty,
                    )
                    .map(|v| v.0)
                })?;
            let next_x = &x + step.alpha * d;
            let (_, new_g) = augmented(&state(p, &next_x, Some(counts))?, &mu, &lambda, penalty)?;
            crate::unconstrained::update_inverse(
                &mut inverse,
                &(&next_x - &x),
                &(new_g - g),
                crate::Method::Bfgs,
            );
            x = next_x;
            inner_iterations += 1;
        }
        let next = state(p, &x, Some(counts))?;
        mu -= penalty * &next.constraints.equalities;
        lambda = (&lambda - penalty * &next.constraints.inequalities).map(|v| v.max(0.0));
        let residual =
            violation(&next.constraints).max(complementarity(&next.constraints, &lambda));
        if k >= 2 && residual > 0.8 * previous_residual {
            penalty *= 2.0;
            finite(penalty, "PHR penalty")?;
        }
        previous_residual = residual;
    }
    unreachable!()
}
