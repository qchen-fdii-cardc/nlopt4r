//! Dense, pure-Rust optimization methods based on the accompanying MATLAB programs.
//!
//! See the README for algorithm mapping, conventions and limitations.
//! Functions and closures can carry borrowed application data; no global state is used.
//!
//! ```
//! use nlopt::{minimize, Method, Options, Vector};
//! let problem = (
//!     |x: &Vector| (x[0] - 3.0).powi(2),
//!     |x: &Vector| Vector::from_vec(vec![2.0 * (x[0] - 3.0)]),
//! );
//! let result = minimize(&problem, &Vector::zeros(1), Method::Bfgs, &Options::default())?;
//! assert!(result.converged());
//! assert!((result.x[0] - 3.0).abs() < 1e-6);
//! # Ok::<(), nlopt::Error>(())
//! ```

pub mod constrained;
pub mod least_squares;
pub mod line_search;
pub mod quadratic;
pub mod trust_region;
mod unconstrained;

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
    DimensionMismatch(&'static str),
    NonFinite(&'static str),
    MissingHessian,
    SingularSystem,
    Infeasible,
    LineSearchFailed,
    IterationLimit(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(s) => write!(f, "invalid input: {s}"),
            Self::DimensionMismatch(s) => write!(f, "dimension mismatch: {s}"),
            Self::NonFinite(s) => write!(f, "non-finite value: {s}"),
            Self::MissingHessian => write!(f, "this method requires an analytic Hessian"),
            Self::SingularSystem => write!(f, "singular or numerically unsolvable linear system"),
            Self::Infeasible => write!(f, "constraints are infeasible"),
            Self::LineSearchFailed => write!(f, "line search exhausted without an acceptable step"),
            Self::IterationLimit(s) => write!(f, "iteration limit exhausted: {s}"),
        }
    }
}
impl std::error::Error for Error {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Converged,
    MaxIterations,
}

#[derive(Debug, Clone)]
pub struct Solution {
    pub x: Vector<f64>,
    pub value: f64,
    /// Euclidean gradient norm (Lagrangian stationarity norm for constrained solvers).
    pub gradient_norm: f64,
    pub iterations: usize,
    pub status: Status,
}

impl Solution {
    pub fn converged(&self) -> bool {
        self.status == Status::Converged
    }
}

#[derive(Debug, Clone)]
pub struct Options {
    pub max_iterations: usize,
    pub gradient_tolerance: f64,
    pub line_search: line_search::ArmijoOptions,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_iterations: 5000,
            gradient_tolerance: 1e-6,
            line_search: Default::default(),
        }
    }
}

impl Options {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        positive(self.gradient_tolerance, "gradient tolerance")?;
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
        return Err(Error::DimensionMismatch(name));
    }
    if !v.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite(name));
    }
    Ok(())
}

pub(crate) fn initial(x: &Vector<f64>, options: &Options) -> Result<(), Error> {
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
        return Err(Error::DimensionMismatch(name));
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

pub(crate) fn evaluate<P: Objective + ?Sized>(
    p: &P,
    x: &Vector<f64>,
) -> Result<(f64, Vector<f64>), Error> {
    vector(x, x.len(), "iterate")?;
    let f = finite(p.value(x), "objective")?;
    let g = p.gradient(x);
    vector(&g, x.len(), "gradient")?;
    Ok((f, g))
}

pub(crate) fn hessian<P: Objective + ?Sized>(p: &P, x: &Vector<f64>) -> Result<Matrix<f64>, Error> {
    let h = p.hessian(x).ok_or(Error::MissingHessian)?;
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
    }
}
