//! Checked evaluation, strict value-only budgets and external-algorithm building blocks.
//!
//! [`EvaluationContext`] validates callback output, charges attempted calls,
//! retains the best point and supports fallible observers. It does not enforce
//! bounds or choose iteration/stationarity rules: an external solver owns those.
//!
//! ```
//! use nlopt4r::{value_only, core::{BudgetOptions, EvaluationContext}};
//! let problem = value_only(|x: &[f64]| Ok(x[0] * x[0]));
//! let budget = BudgetOptions { max_evaluations: 2, target_value: None };
//! let mut context = EvaluationContext::new(&problem, 1, &budget)?;
//! context.evaluate(&[2.0])?;
//! context.evaluate(&[1.0])?;
//! assert_eq!(context.best().unwrap().value, 1.0);
//! assert_eq!(context.counts().objective, 2);
//! # Ok::<(), nlopt4r::Error>(())
//! ```
pub use crate::{
    Error, EvaluationCounts, EvaluationError, Evaluator, Function, Solution, StopOptions,
};
use crate::{Vector, finite, utils::validate_slice};

/// Value-only search budget and optional objective target, with no gradient tolerance.
#[derive(Debug, Clone)]
pub struct BudgetOptions {
    /// Positive strict objective-call limit; default 10_000.
    pub max_evaluations: usize,
    /// Stop after a valid `f(x) <= target`; finite when present, default `None`.
    pub target_value: Option<f64>,
}
impl Default for BudgetOptions {
    fn default() -> Self {
        Self {
            max_evaluations: 10_000,
            target_value: None,
        }
    }
}
impl BudgetOptions {
    /// Check positivity of the budget and finiteness of the optional target.
    pub fn validate(&self) -> Result<(), Error> {
        if self.max_evaluations == 0 {
            return Err(Error::InvalidInput("evaluation budget must be positive"));
        }
        if self.target_value.is_some_and(|v| !v.is_finite()) {
            return Err(Error::InvalidInput("target value must be finite"));
        }
        Ok(())
    }
}
/// Value-only search outcome; only `TargetReached` asserts a requested value was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchTermination {
    /// Best checked objective is at or below the requested target.
    TargetReached,
    /// Strict objective allowance exhausted.
    MaxEvaluations,
    /// Algorithm iteration/generation limit reached.
    MaxIterations,
    /// Mesh/simplex/population progress collapsed; not an optimality certificate.
    Stagnated,
    /// An observer requested cancellation.
    Cancelled,
}
/// Checked objective value and its point, without a fabricated derivative diagnostic.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluatedPoint {
    /// Point supplied to a successful checked evaluation.
    pub x: Vector,
    /// Finite objective at `x`.
    pub value: f64,
}
/// Observer instruction, applied after the current successful evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Control {
    /// Permit further evaluations while other stop conditions allow them.
    Continue,
    /// Cancel before another objective call.
    Stop,
}
/// Borrowed observation of a successful checked evaluation.
pub struct EvaluationEvent<'a> {
    /// Current trial point, which may differ from the best point.
    pub x: &'a [f64],
    /// Current trial objective.
    pub value: f64,
    /// Lowest checked objective through this event.
    pub best_value: f64,
    /// Attempted work including this evaluation.
    pub evaluations: EvaluationCounts,
}
/// Borrowed mutable callback returning an instruction or explicit application error.
/// Event references are valid only during that invocation.
pub type Observer<'a> =
    dyn for<'b> FnMut(&EvaluationEvent<'b>) -> Result<Control, EvaluationError> + 'a;

/// Checked callback calls, strict budgets, best-ever checkpoints and observation.
/// Errors do not discard previously evaluated points. The caller chooses its
/// own iteration and stationarity criteria.
pub struct EvaluationContext<'a, P: Evaluator + ?Sized> {
    problem: &'a P,
    dimension: usize,
    budget: BudgetOptions,
    counts: EvaluationCounts,
    best: Option<EvaluatedPoint>,
    cancelled: bool,
    observer: Option<&'a mut Observer<'a>>,
}
impl<'a, P: Evaluator + ?Sized> EvaluationContext<'a, P> {
    /// Borrow the problem and copy validated options; `dimension` must be positive.
    /// Construction makes no objective calls.
    pub fn new(problem: &'a P, dimension: usize, budget: &BudgetOptions) -> Result<Self, Error> {
        budget.validate()?;
        if dimension == 0 {
            return Err(Error::InvalidInput("empty problem"));
        }
        Ok(Self {
            problem,
            dimension,
            budget: budget.clone(),
            counts: EvaluationCounts::default(),
            best: None,
            cancelled: false,
            observer: None,
        })
    }
    /// Attach an observer called after output validation and best-point update.
    /// An observer error propagates; [`Self::failure`] can retain that valid point.
    pub fn with_observer(mut self, observer: &'a mut Observer<'a>) -> Self {
        self.observer = Some(observer);
        self
    }
    /// Attempted work so far, including callbacks that returned errors.
    pub fn counts(&self) -> EvaluationCounts {
        self.counts
    }
    /// Lowest valid value seen, or `None` before any successful evaluation.
    /// Equal-value ties retain the earlier point.
    pub fn best(&self) -> Option<&EvaluatedPoint> {
        self.best.as_ref()
    }
    /// Current stop, with cancellation taking precedence over target, then budget.
    /// `None` says only that another evaluation is allowed.
    pub fn stop_reason(&self) -> Option<SearchTermination> {
        if self.cancelled {
            return Some(SearchTermination::Cancelled);
        }
        if let (Some(best), Some(target)) = (&self.best, self.budget.target_value)
            && best.value <= target
        {
            return Some(SearchTermination::TargetReached);
        }
        if self.counts.objective >= self.budget.max_evaluations {
            return Some(SearchTermination::MaxEvaluations);
        }
        None
    }
    /// Perform a checked value-only call; a stopped context makes no further callback.
    /// Returns validation, application or [`Error::SearchStopped`] errors explicitly.
    pub fn evaluate(&mut self, x: &[f64]) -> Result<f64, Error> {
        self.evaluate_into(x, None)
    }
    /// Evaluate value and optionally fill a gradient buffer of the configured dimension.
    ///
    /// Requested gradient entries are initialized to NaN to detect incomplete output.
    /// A valid-sized attempted call charges objective (and gradient, if requested)
    /// before invoking the application. Invalid inputs are rejected without charge.
    /// The best point is updated only after all requested outputs are valid.
    pub fn evaluate_into(
        &mut self,
        x: &[f64],
        mut gradient: Option<&mut [f64]>,
    ) -> Result<f64, Error> {
        validate_slice(x, self.dimension, "evaluation point")?;
        if let Some(g) = gradient.as_mut() {
            if g.len() != self.dimension {
                return Err(Error::DimensionMismatch {
                    context: "gradient buffer",
                    expected: (self.dimension, 1),
                    actual: (g.len(), 1),
                });
            }
            g.fill(f64::NAN);
        }
        if let Some(reason) = self.stop_reason() {
            return Err(Error::SearchStopped(reason));
        }
        self.counts.objective += 1;
        if gradient.is_some() {
            self.counts.gradient += 1;
        }
        let value = finite(
            self.problem.evaluate(x, gradient.as_deref_mut())?,
            "objective",
        )?;
        if let Some(g) = gradient {
            validate_slice(g, self.dimension, "gradient")?;
        }
        if self.best.as_ref().is_none_or(|best| value < best.value) {
            self.best = Some(EvaluatedPoint {
                x: Vector::from_column_slice(x),
                value,
            });
        }
        if let Some(observer) = &mut self.observer {
            let event = EvaluationEvent {
                x,
                value,
                best_value: self.best.as_ref().expect("evaluated point").value,
                evaluations: self.counts,
            };
            self.cancelled = observer(&event).map_err(|source| Error::Evaluation {
                operation: "observer",
                source,
            })? == Control::Stop;
        }
        Ok(value)
    }
    /// Attach the best valid value-only checkpoint to a failed search.
    ///
    /// Call this when propagating an error from a custom algorithm; direct
    /// evaluation errors do not automatically wrap themselves.
    ///
    /// ```
    /// use nlopt4r::{value_only, core::{BudgetOptions, EvaluationContext}};
    /// let problem = value_only(|x: &[f64]| {
    ///     if x[0] < 0.0 { return Err("simulation failed".into()); }
    ///     Ok(x[0]*x[0])
    /// });
    /// let mut context = EvaluationContext::new(&problem, 1, &BudgetOptions::default())?;
    /// context.evaluate(&[2.0])?;
    /// let source = context.evaluate(&[-1.0]).unwrap_err();
    /// let error = context.failure(source);
    /// assert_eq!(error.last_point().unwrap().value, 4.0);
    /// assert_eq!(context.counts().objective, 2); // Failed attempt is charged.
    /// # Ok::<(), nlopt4r::Error>(())
    /// ```
    pub fn failure(&self, source: Error) -> Error {
        if let Some(point) = &self.best {
            Error::SearchFailure {
                source: Box::new(source),
                last_point: Box::new(point.clone()),
                evaluations: self.counts,
            }
        } else {
            source
        }
    }
}
