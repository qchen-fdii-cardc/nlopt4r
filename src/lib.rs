//! Pure-Rust local, global, least-squares and constrained optimization.
//!
//! This is an independent solver library, **not** a binding to the upstream
//! NLopt C library. Solvers use dynamically sized `f64` vectors, borrow problem
//! data and return explicit reports or recoverable errors. No native optimizer,
//! global RNG, automatic differentiation or application-specific objective is
//! required.
//!
//! # Choose a solver
//!
//! | Problem / available information | Entry point | Important limitation |
//! |---|---|---|
//! | Smooth objective and gradient | [`local::lbfgs()`] | Unbounded; O(m*n) history |
//! | Small smooth problem | [`minimize`] with [`Method::Bfgs`] | Dense O(n^2) storage |
//! | Hessian-vector products | [`local::newton_cg()`], [`local::trust_region_cg`] | Symmetric operator; O(n) workspace |
//! | Small problem with dense Hessian | [`trust_region::minimize`] | O(n^3) spectral subproblems |
//! | Finite box and gradient | [`bounded::projected_gradient`] | Not L-BFGS-B |
//! | Finite box, no derivatives, local search | [`derivative_free`] | Mesh/simplex collapse is not optimality |
//! | Finite box, multimodal search | [`global`] DE / PSO | Stochastic; no global-optimality certificate |
//! | Residuals and Jacobian | [`least_squares::levenberg_marquardt_with_options`] | Nonzero residual may be optimal |
//! | Convex quadratic with linear constraints | [`quadratic::minimize`] | Dense; limited flat/unbounded cases |
//! | Smooth nonlinear constraints | [`constrained::sqp_with_restoration`] or [`constrained::augmented_lagrangian`] | Local KKT methods |
//!
//! All methods minimize. To maximize, negate the objective and its derivatives.
//! Scale variables and residuals sensibly; tighter tolerances cannot overcome
//! inaccurate derivatives or floating-point resolution.
//!
//! # Quick start: borrowed closures
//!
//! [`Objective`] is implemented for `(value, gradient)` and
//! `(value, gradient, hessian)` tuples. The gradient has length `n`, and a
//! Hessian is a symmetric `n` by `n` matrix.
//!
//! ```
//! use nlopt4r::{local::{lbfgs, LbfgsOptions}, Vector};
//!
//! let target = Vector::from_vec(vec![3.0, -2.0]);
//! let problem = (
//!     |x: &Vector| 0.5 * (x - &target).norm_squared(),
//!     |x: &Vector| x - &target,
//! );
//! let mut options = LbfgsOptions::default();
//! options.stopping.max_evaluations = Some(200);
//! let result = lbfgs(&problem, &Vector::zeros(2), &options)?;
//! assert!(result.converged());
//! assert!((&result.x - &target).norm() < 1e-6);
//! # Ok::<(), nlopt4r::Error>(())
//! ```
//!
//! # Buffered, fallible callbacks
//!
//! Use [`Function`] or implement [`Evaluator`] to share value/gradient work
//! and write into solver-owned buffers. Fill **every** requested gradient entry;
//! `None` means a value-only trial. Callback failures and non-finite output are
//! errors, not silently rejected steps. Use [`value_only`] for derivative-free
//! solvers; gradients are never approximated automatically.
//!
//! ```
//! use nlopt4r::{EvaluationError, Function, Method, Options, Vector, minimize};
//!
//! let problem = Function::new(|x: &[f64], gradient: Option<&mut [f64]>| {
//!     if let Some(g) = gradient {
//!         for (out, &value) in g.iter_mut().zip(x) { *out = 2.0 * value; }
//!     }
//!     Ok::<_, EvaluationError>(x.iter().map(|v| v * v).sum())
//! });
//! let result = minimize(
//!     &problem, &Vector::from_element(2, 1.0), Method::Bfgs, &Options::default(),
//! )?;
//! assert!(result.converged());
//! # Ok::<(), nlopt4r::Error>(())
//! ```
//!
//! # Stopping is not always success
//!
//! [`StopOptions`] controls gradient/KKT solvers; [`core::BudgetOptions`]
//! controls value-only searches. Their option types are deliberately separate.
//!
//! - `Ok(report)` means the solver returned a report, **not** necessarily an optimum.
//!   Check [`Solution::converged`], [`Solution::termination_reason`] or a
//!   value-only report's [`core::SearchTermination`].
//! - Gradient/KKT convergence is first-order stationarity, not a proof of a
//!   nonconvex local minimum. A search target is not a global-optimality proof.
//! - Budget, cancellation and stagnation stops are not convergence. Objective
//!   budgets include rejected trials and failed callbacks and are never exceeded.
//! - A zero iteration limit still evaluates the initial point. A zero objective
//!   budget is invalid. Valid convergence at the final allowed state takes
//!   precedence over iteration/evaluation exhaustion.
//! - Optional [`Tolerance`] tests detect accepted-point stagnation; they are
//!   disabled by default. Very tight tolerances can produce line-search errors
//!   near numerical plateaus instead of convergence.
//!
//! [`EvaluationCounts`] distinguishes objective, derivative and synthetic-model
//! work. Combined value/gradient callbacks increment both categories.
//! Hessian-vector products are counted as Hessian work, not objective calls.
//!
//! # Recover from failures
//!
//! [`Error::root_cause`] unwraps solver context. [`Error::last_solution`] exposes
//! a fully evaluated local checkpoint; [`Error::last_point`] exposes a best
//! value-only checkpoint. Neither is a certified optimum. Initial/input errors
//! may have no checkpoint. Restoration SQP additionally retains multipliers in
//! [`constrained::SqpError::last_constrained`]. Callback panics are not caught.
//!
//! # Bounds, residuals and constraints
//!
//! [`Bounds`] requires finite, ordered endpoints and finite widths; equal
//! endpoints fix a variable. Unconstrained methods do **not** enforce these
//! bounds. Choose a bounded solver rather than clamping inside an objective.
//!
//! Least squares minimizes `||r||^2 / 2` with Jacobian shape `residuals x variables`.
//! Nonlinear constraints use `h(x)=0`, `g(x)>=0`, with one Jacobian row per
//! constraint. The Lagrangian is `L=f-mu'h-lambda'g`, `lambda>=0`.
//! [`constrained::Constraints::from_nonpositive`] converts the alternative
//! `c(x)<=0` convention explicitly.
//!
//! # Extend and learn more
//!
//! [`core::EvaluationContext`] and [`utils`] provide checked evaluations,
//! strict budgets, observers, seeded sampling and numerical building blocks
//! for external algorithms. Existing function-based entry points remain usable
//! with borrowed trait objects.
//!
//! Repository guides cover
//! [theory and references](https://github.com/qchen-fdii-cardc/nlopt4r/blob/master/docs/THEORY.md),
//! [benchmarks and known limitations](https://github.com/qchen-fdii-cardc/nlopt4r/blob/master/docs/BENCHMARK_RESULTS.md)
//! and [migration / application usage](https://github.com/qchen-fdii-cardc/nlopt4r/blob/master/README.md).

pub mod bounded;
pub mod constrained;
pub mod core;
pub mod derivative_free;
mod evaluation;
pub mod global;
pub mod least_squares;
pub mod line_search;
pub mod local;
pub mod problem;
pub mod quadratic;
mod stopping;
pub mod trust_region;
mod unconstrained;
pub mod utils;

pub use evaluation::{EvaluationError, Evaluator, Function, ValueOnly, value_only};
pub use problem::Bounds;
pub use stopping::{EvaluationCounts, StopOptions, TerminationReason, Tolerance};

/// Dynamically sized nalgebra matrix. Optimization entry points use `T = f64`.
pub type Matrix<T = f64> = nalgebra::DMatrix<T>;
/// Dynamically sized nalgebra column vector. Optimization entry points use `T = f64`.
pub type Vector<T = f64> = nalgebra::DVector<T>;
pub use unconstrained::{Method, minimize};

/// An objective with analytic derivatives. Hessians are required only by Newton methods.
///
/// Implement this for infallible application models, or pass tuples of closures.
/// The blanket [`Evaluator`] adapter allocates returned derivative vectors;
/// choose [`Function`] for buffered/fallible evaluation. Because of this blanket
/// implementation, a type should implement `Objective` **or** `Evaluator`, not both.
pub trait Objective {
    /// Objective value at `x`; must be finite at every evaluated trial.
    fn value(&self, x: &Vector<f64>) -> f64;
    /// Analytic gradient with the same dimension as `x`; every entry must be finite.
    fn gradient(&self, x: &Vector<f64>) -> Vector<f64>;
    /// Symmetric finite Hessian of shape `x.len() x x.len()`, or `None` if unavailable.
    /// The default is `None`; gradient-only methods never require it.
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

/// Invalid inputs, numerical failures or callback failures, optionally with recovery state.
///
/// Solvers attach [`Error::SolverFailure`] or [`Error::SearchFailure`] after a
/// valid checkpoint. Inspect [`Self::root_cause`] for the underlying category.
/// Ordinary outer budget/stagnation stops generally return a non-converged report;
/// nested/subproblem exhaustion can instead be an error.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Invalid options, domain, direction or other precondition.
    InvalidInput(&'static str),
    /// A vector or matrix has the wrong shape; vectors are represented as `(n, 1)`.
    DimensionMismatch {
        /// Name of the input or callback being checked.
        context: &'static str,
        /// Required `(rows, columns)`.
        expected: (usize, usize),
        /// Supplied `(rows, columns)`.
        actual: (usize, usize),
    },
    /// Non-finite input, callback output or intermediate arithmetic.
    NonFinite(&'static str),
    /// The selected dense Newton method requested an unavailable Hessian.
    MissingHessian,
    /// Linear system is singular, inconsistent or numerically unsolvable.
    SingularSystem,
    /// A constraint system or linearized subproblem was found infeasible.
    Infeasible,
    /// No acceptable line-search step was found; no unchecked step is taken.
    LineSearchFailed {
        /// Number of evaluated trials.
        evaluations: usize,
        /// Last attempted step length.
        last_step: f64,
    },
    /// A nested solver exhausted its iteration allowance.
    IterationLimit {
        /// Nested solver name.
        solver: &'static str,
        /// Consumed iterations.
        iterations: usize,
    },
    /// A synthetic subproblem stopped without satisfying its convergence test.
    SubproblemStopped {
        /// Subproblem solver name.
        solver: &'static str,
        /// Specific non-convergence stop.
        reason: TerminationReason,
        /// Consumed iterations.
        iterations: usize,
    },
    /// Internal objective-budget signal, converted to a report when a valid outer state exists.
    EvaluationBudgetExceeded,
    /// A value-only evaluation was requested after the context had stopped.
    SearchStopped(core::SearchTermination),
    /// Failed value-only search with its best valid evaluated point.
    SearchFailure {
        /// Underlying failure.
        source: Box<Error>,
        /// Best fully checked value-only evaluation, not a stationary-point certificate.
        last_point: Box<core::EvaluatedPoint>,
        /// Attempted work through the failure.
        evaluations: EvaluationCounts,
    },
    /// Fallible application callback or observer returned an error.
    Evaluation {
        /// Callback category, such as `"objective"`, `"Hessian"` or `"observer"`.
        operation: &'static str,
        /// Application error's display message.
        source: EvaluationError,
    },
    /// Failed local solver with a fully evaluated checkpoint marked [`Status::Failed`].
    SolverFailure {
        /// Outer solver name.
        solver: &'static str,
        /// Underlying failure; use [`Error::root_cause`] to unwrap nesting.
        source: Box<Error>,
        /// Last valid local checkpoint, not necessarily the smallest observed objective.
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
    /// Unwrap solver/search recovery wrappers, retaining the actual error category.
    /// An [`Error::Evaluation`] still contains its callback message as a source.
    pub fn root_cause(&self) -> &Error {
        match self {
            Self::SolverFailure { source, .. } => source.root_cause(),
            Self::SearchFailure { source, .. } => source.root_cause(),
            _ => self,
        }
    }
    /// Recover a local checkpoint from a solver failure, if available.
    /// This does not return value-only or full constrained multiplier state.
    pub fn last_solution(&self) -> Option<&Solution> {
        match self {
            Self::SolverFailure { last_solution, .. } => Some(last_solution),
            _ => None,
        }
    }
    /// Recover the best value-only point from a search failure, if available.
    pub fn last_point(&self) -> Option<&core::EvaluatedPoint> {
        match self {
            Self::SearchFailure { last_point, .. } => Some(last_point),
            _ => None,
        }
    }
}

/// Coarse local-solver outcome. See [`TerminationReason`] for the exact test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// The solver-specific gradient, KKT or scalar interval criterion was satisfied.
    Converged,
    /// Outer iteration limit reached without convergence.
    MaxIterations,
    /// Strict objective-call allowance exhausted without convergence.
    MaxEvaluations,
    /// An enabled step/value tolerance detected insufficient accepted progress.
    Stagnated,
    /// Recovery checkpoint attached to an error.
    Failed,
}

/// Fully evaluated local state and diagnostics.
///
/// A successful `Result` may contain a budget/stagnation report; inspect
/// [`Self::converged`]. Gradient-based methods retain the last valid accepted
/// state rather than an incomplete lower-value trial.
#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    /// Returned iterate.
    pub x: Vector<f64>,
    /// Objective at `x`; least-squares methods use half the squared residual norm.
    pub value: f64,
    /// Euclidean gradient norm (Lagrangian stationarity norm for constrained solvers).
    /// NaN in scalar-search error checkpoints, where no gradient is evaluated.
    /// Projected gradient reports a box gradient-mapping norm; adaptive QR LM
    /// reports column-scaled `||D^-1 J'r||` instead.
    pub gradient_norm: f64,
    /// Outer iterations; nested work is reported by family-specific diagnostics.
    pub iterations: usize,
    /// Coarse outcome, including non-converged normal stops.
    pub status: Status,
    /// Exact criterion or failure associated with `status`.
    pub termination_reason: TerminationReason,
    /// Attempted objective/derivative/model work, including failed callbacks.
    pub evaluations: EvaluationCounts,
}

impl Solution {
    /// Whether the solver-specific convergence test passed; not a global-minimum certificate.
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

/// Shared stopping and Armijo settings for legacy local solvers.
///
/// Use typed alternatives such as [`local::LbfgsOptions`] or
/// [`least_squares::LmOptions`] when selecting their corresponding algorithms.
/// Defaults are the defaults of the two contained option types.
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// Outer iteration, objective-budget, stationarity and optional stagnation controls.
    pub stopping: StopOptions,
    /// Armijo backtracking settings; not used by strong-Wolfe or trust-region methods.
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
