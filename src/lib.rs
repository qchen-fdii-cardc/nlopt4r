//! Dense, pure-Rust optimization methods based on the accompanying MATLAB programs.
//!
//! See the README for algorithm mapping, conventions and limitations.
//! Functions and closures can carry borrowed application data; no global state is used.
//!
//! ```
//! use nlopt4r::{minimize, Method, Options, Vector};
//! let problem = (
//!     |x: &Vector| (x[0] - 3.0).powi(2),
//!     |x: &Vector| Vector::from_vec(vec![2.0 * (x[0] - 3.0)]),
//! );
//! let result = minimize(&problem, &Vector::zeros(1), Method::Bfgs, &Options::default())?;
//! assert!(result.converged());
//! assert!((result.x[0] - 3.0).abs() < 1e-6);
//! # Ok::<(), nlopt4r::Error>(())
//! ```

pub mod constrained;
pub mod core;
mod evaluation;
pub mod least_squares;
pub mod line_search;
pub mod problem;
pub mod quadratic;
mod stopping;
pub mod trust_region;
mod unconstrained;
pub mod utils;

pub use evaluation::{EvaluationError, Evaluator, Function, value_only};
pub use problem::Bounds;
pub use stopping::{EvaluationCounts, StopOptions, TerminationReason, Tolerance};

pub type Matrix<T = f64> = nalgebra::DMatrix<T>;
pub type Vector<T = f64> = nalgebra::DVector<T>;
pub use unconstrained::{Method, minimize};

/// An objective with analytic derivatives. Hessians are required only by Newton methods.
pub trait Objective {
    fn value(&self, x: &Vector<f64>) -> f64;
    fn gradient(&self, x: &Vector<f64>) -> Vector<f64>;
    fn hessian(&self, _x: &Vector<f64>) -> Option<Matrix<f64>> {
        None
    }
}

impl<F, G> Objective for (F, G)
where
    F: Fn(&Vector<f64>) -> f64,
    G: Fn(&Vector<f64>) -> Vector<f64>,
{
    fn value(&self, x: &Vector<f64>) -> f64 {
        self.0(x)
    }
    fn gradient(&self, x: &Vector<f64>) -> Vector<f64> {
        self.1(x)
    }
}

impl<F, G, H> Objective for (F, G, H)
where
    F: Fn(&Vector<f64>) -> f64,
    G: Fn(&Vector<f64>) -> Vector<f64>,
    H: Fn(&Vector<f64>) -> Matrix<f64>,
{
    fn value(&self, x: &Vector<f64>) -> f64 {
        self.0(x)
    }
    fn gradient(&self, x: &Vector<f64>) -> Vector<f64> {
        self.1(x)
    }
    fn hessian(&self, x: &Vector<f64>) -> Option<Matrix<f64>> {
        Some(self.2(x))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    InvalidInput(&'static str),
    DimensionMismatch {
        context: &'static str,
        expected: (usize, usize),
        actual: (usize, usize),
    },
    NonFinite(&'static str),
    MissingHessian,
    SingularSystem,
    Infeasible,
    LineSearchFailed {
        evaluations: usize,
        last_step: f64,
    },
    IterationLimit {
        solver: &'static str,
        iterations: usize,
    },
    SubproblemStopped {
        solver: &'static str,
        reason: TerminationReason,
        iterations: usize,
    },
    EvaluationBudgetExceeded,
    SearchStopped(core::SearchTermination),
    SearchFailure {
        source: Box<Error>,
        last_point: Box<core::EvaluatedPoint>,
        evaluations: EvaluationCounts,
    },
    Evaluation {
        operation: &'static str,
        source: EvaluationError,
    },
    SolverFailure {
        solver: &'static str,
        source: Box<Error>,
        last_solution: Box<Solution>,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(s) => write!(f, "invalid input: {s}"),
            Self::DimensionMismatch {
                context,
                expected,
                actual,
            } => write!(
                f,
                "dimension mismatch in {context}: expected {expected:?}, got {actual:?}"
            ),
            Self::NonFinite(s) => write!(f, "non-finite value: {s}"),
            Self::MissingHessian => write!(f, "this method requires an analytic Hessian"),
            Self::SingularSystem => write!(f, "singular or numerically unsolvable linear system"),
            Self::Infeasible => write!(f, "constraints are infeasible"),
            Self::LineSearchFailed {
                evaluations,
                last_step,
            } => write!(
                f,
                "line search exhausted after {evaluations} trials (last step {last_step})"
            ),
            Self::IterationLimit { solver, iterations } => write!(
                f,
                "iteration limit exhausted in {solver} after {iterations} iterations"
            ),
            Self::SubproblemStopped {
                solver,
                reason,
                iterations,
            } => write!(
                f,
                "{solver} stopped after {iterations} iterations: {reason:?}"
            ),
            Self::EvaluationBudgetExceeded => write!(f, "objective evaluation budget exhausted"),
            Self::SearchStopped(reason) => write!(f, "search stopped: {reason:?}"),
            Self::SearchFailure {
                source,
                evaluations,
                ..
            } => write!(
                f,
                "search failed after {} objective evaluations: {source}",
                evaluations.objective
            ),
            Self::Evaluation { operation, source } => {
                write!(f, "{operation} evaluation failed: {source}")
            }
            Self::SolverFailure {
                solver,
                source,
                last_solution,
            } => write!(
                f,
                "{solver} failed after {} iterations: {source}",
                last_solution.iterations
            ),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Evaluation { source, .. } => Some(source),
            Self::SolverFailure { source, .. } => Some(source.as_ref()),
            Self::SearchFailure { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}
impl Error {
    pub fn root_cause(&self) -> &Error {
        match self {
            Self::SolverFailure { source, .. } => source.root_cause(),
            Self::SearchFailure { source, .. } => source.root_cause(),
            _ => self,
        }
    }
    pub fn last_solution(&self) -> Option<&Solution> {
        match self {
            Self::SolverFailure { last_solution, .. } => Some(last_solution),
            _ => None,
        }
    }
    pub fn last_point(&self) -> Option<&core::EvaluatedPoint> {
        match self {
            Self::SearchFailure { last_point, .. } => Some(last_point),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Converged,
    MaxIterations,
    MaxEvaluations,
    Stagnated,
    Failed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    pub x: Vector<f64>,
    pub value: f64,
    /// Euclidean gradient norm (Lagrangian stationarity norm for constrained solvers).
    /// NaN in scalar-search error checkpoints, where no gradient is evaluated.
    pub gradient_norm: f64,
    pub iterations: usize,
    pub status: Status,
    pub termination_reason: TerminationReason,
    pub evaluations: EvaluationCounts,
}

impl Solution {
    pub fn converged(&self) -> bool {
        self.status == Status::Converged
    }
    pub(crate) fn terminate(&mut self, reason: TerminationReason) {
        self.termination_reason = reason;
        self.status = match reason {
            TerminationReason::GradientTolerance
            | TerminationReason::KktTolerance
            | TerminationReason::IntervalTolerance => Status::Converged,
            TerminationReason::MaxIterations => Status::MaxIterations,
            TerminationReason::MaxEvaluations => Status::MaxEvaluations,
            TerminationReason::ObjectiveStagnation | TerminationReason::StepStagnation => {
                Status::Stagnated
            }
            TerminationReason::Failed => Status::Failed,
        };
    }
}

#[derive(Debug, Clone, Default)]
pub struct Options {
    pub stopping: StopOptions,
    pub line_search: line_search::ArmijoOptions,
}

impl Options {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        self.stopping.validate()?;
        self.line_search.validate()
    }
}

pub(crate) fn positive(v: f64, name: &'static str) -> Result<(), Error> {
    if !v.is_finite() || v <= 0.0 {
        Err(Error::InvalidInput(name))
    } else {
        Ok(())
    }
}

pub(crate) fn finite(v: f64, name: &'static str) -> Result<f64, Error> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(Error::NonFinite(name))
    }
}

pub(crate) fn vector(v: &Vector<f64>, n: usize, name: &'static str) -> Result<(), Error> {
    if v.len() != n {
        return Err(Error::DimensionMismatch {
            context: name,
            expected: (n, 1),
            actual: (v.len(), 1),
        });
    }
    if !v.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite(name));
    }
    Ok(())
}

pub(crate) fn initial(x: &Vector<f64>, options: &StopOptions) -> Result<(), Error> {
    options.validate()?;
    if x.is_empty() {
        return Err(Error::InvalidInput("empty initial point"));
    }
    vector(x, x.len(), "initial point")
}

pub(crate) fn matrix(
    a: &Matrix<f64>,
    rows: usize,
    cols: usize,
    name: &'static str,
) -> Result<(), Error> {
    if a.shape() != (rows, cols) {
        return Err(Error::DimensionMismatch {
            context: name,
            expected: (rows, cols),
            actual: a.shape(),
        });
    }
    if !a.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite(name));
    }
    Ok(())
}

pub(crate) fn symmetric(a: &Matrix<f64>, n: usize) -> Result<(), Error> {
    matrix(a, n, n, "Hessian")?;
    if (a - a.transpose()).norm() > 1e-10 * (1.0 + a.norm()) {
        return Err(Error::InvalidInput("Hessian must be symmetric"));
    }
    Ok(())
}

pub(crate) fn evaluate<P: Evaluator + ?Sized>(
    p: &P,
    x: &Vector<f64>,
) -> Result<(f64, Vector<f64>), Error> {
    vector(x, x.len(), "iterate")?;
    let mut g = Vector::from_element(x.len(), f64::NAN);
    let f = finite(
        p.evaluate(x.as_slice(), Some(g.as_mut_slice()))?,
        "objective",
    )?;
    vector(&g, x.len(), "gradient")?;
    finite(g.norm(), "gradient norm")?;
    Ok((f, g))
}

pub(crate) fn hessian<P: Evaluator + ?Sized>(p: &P, x: &Vector<f64>) -> Result<Matrix<f64>, Error> {
    let h = p
        .evaluate_hessian(x.as_slice())?
        .ok_or(Error::MissingHessian)?;
    symmetric(&h, x.len())?;
    Ok(h)
}

pub(crate) fn solve(a: &Matrix<f64>, b: &Vector<f64>) -> Result<Vector<f64>, Error> {
    let x = a
        .clone()
        .full_piv_lu()
        .solve(b)
        .ok_or(Error::SingularSystem)?;
    vector(&x, b.len(), "linear solve")?;
    if (a * &x - b).norm() > 1e-7 * (1.0 + b.norm()) {
        return Err(Error::SingularSystem);
    }
    Ok(x)
}

pub(crate) fn solution(
    x: Vector<f64>,
    value: f64,
    gradient_norm: f64,
    iterations: usize,
    converged: bool,
) -> Solution {
    Solution {
        x,
        value,
        gradient_norm,
        iterations,
        status: if converged {
            Status::Converged
        } else {
            Status::MaxIterations
        },
        termination_reason: if converged {
            TerminationReason::GradientTolerance
        } else {
            TerminationReason::MaxIterations
        },
        evaluations: EvaluationCounts::default(),
    }
}
