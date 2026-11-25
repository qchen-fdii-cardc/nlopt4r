//! Budgeted, checked evaluation and value-only search reports.
pub use crate::{
    Error, EvaluationCounts, EvaluationError, Evaluator, Function, Solution, StopOptions,
};
use crate::{Vector, finite, utils::validate_slice};

#[derive(Debug, Clone)]
pub struct BudgetOptions {
    pub max_evaluations: usize,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchTermination {
    TargetReached,
    MaxEvaluations,
    MaxIterations,
    Stagnated,
    Cancelled,
}
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluatedPoint {
    pub x: Vector,
    pub value: f64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Control {
    Continue,
    Stop,
}
pub struct EvaluationEvent<'a> {
    pub x: &'a [f64],
    pub value: f64,
    pub best_value: f64,
    pub evaluations: EvaluationCounts,
}
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
    pub fn with_observer(mut self, observer: &'a mut Observer<'a>) -> Self {
        self.observer = Some(observer);
        self
    }
    pub fn counts(&self) -> EvaluationCounts {
        self.counts
    }
    pub fn best(&self) -> Option<&EvaluatedPoint> {
        self.best.as_ref()
    }
    pub fn stop_reason(&self) -> Option<SearchTermination> {
        if self.cancelled {
            return Some(SearchTermination::Cancelled);
        }
        if let (Some(best), Some(target)) = (&self.best, self.budget.target_value) {
            if best.value <= target {
                return Some(SearchTermination::TargetReached);
            }
        }
        if self.counts.objective >= self.budget.max_evaluations {
            return Some(SearchTermination::MaxEvaluations);
        }
        None
    }
    pub fn evaluate(&mut self, x: &[f64]) -> Result<f64, Error> {
        self.evaluate_into(x, None)
    }
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
