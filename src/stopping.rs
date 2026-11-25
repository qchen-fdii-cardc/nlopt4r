use crate::{
    Error, Evaluator, Matrix, Solution, Vector, evaluate, finite, hessian, positive, solution,
    vector,
};

/// Mixed tolerance: `difference <= absolute + relative * scale`.
///
/// Both terms must be finite and nonnegative, and at least one must be positive.
/// Local stagnation tests use the maximum of consecutive point norms or absolute
/// objective values as `scale`; scalar search uses its interval-specific scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// Absolute allowance in the units of the measured quantity.
    pub absolute: f64,
    /// Dimensionless allowance multiplied by the solver's scale.
    pub relative: f64,
}

impl Tolerance {
    /// Construct an absolute-only tolerance. Validation occurs when a solver uses it.
    pub fn absolute(value: f64) -> Self {
        Self {
            absolute: value,
            relative: 0.0,
        }
    }
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if !self.absolute.is_finite()
            || !self.relative.is_finite()
            || self.absolute < 0.0
            || self.relative < 0.0
            || (self.absolute == 0.0 && self.relative == 0.0)
        {
            return Err(Error::InvalidInput(
                "tolerance must be finite, nonnegative and nonzero",
            ));
        }
        Ok(())
    }
    pub(crate) fn satisfied(&self, difference: f64, scale: f64) -> bool {
        difference <= self.absolute + self.relative * scale
    }
}

/// Common controls for local gradient/KKT solvers.
///
/// Defaults: 5000 outer iterations, no objective cap, gradient tolerance 1e-6,
/// no stagnation tests. Gradient and feasibility criteria are solver-specific;
/// see [`crate::Solution::gradient_norm`]. A zero iteration limit is allowed.
#[derive(Debug, Clone)]
pub struct StopOptions {
    /// Outer iteration limit; zero still permits initial evaluation/convergence.
    pub max_iterations: usize,
    /// Maximum objective evaluations, including line-search trials and failed callbacks.
    pub max_evaluations: Option<usize>,
    /// Positive finite absolute stationarity threshold; default 1e-6.
    pub gradient_tolerance: f64,
    /// Optional stagnation tests; neither certifies optimality.
    pub value_tolerance: Option<Tolerance>,
    /// Optional accepted-point displacement test; reports stagnation, not convergence.
    pub step_tolerance: Option<Tolerance>,
}
impl Default for StopOptions {
    fn default() -> Self {
        Self {
            max_iterations: 5000,
            max_evaluations: None,
            gradient_tolerance: 1e-6,
            value_tolerance: None,
            step_tolerance: None,
        }
    }
}
impl StopOptions {
    /// Reject invalid tolerances or `Some(0)` objective budgets before callback evaluation.
    pub fn validate(&self) -> Result<(), Error> {
        positive(self.gradient_tolerance, "gradient tolerance")?;
        if self.max_evaluations == Some(0) {
            return Err(Error::InvalidInput("evaluation budget must be positive"));
        }
        for t in [self.value_tolerance, self.step_tolerance]
            .into_iter()
            .flatten()
        {
            t.validate()?;
        }
        Ok(())
    }
}

/// Detailed reason for a local solver's returned status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    /// Gradient or method-specific stationarity norm is sufficiently small.
    GradientTolerance,
    /// Required stationarity, feasibility and complementarity conditions passed.
    KktTolerance,
    /// Scalar search bracket satisfied its interval tolerance.
    IntervalTolerance,
    /// Outer iteration limit exhausted.
    MaxIterations,
    /// Objective budget exhausted.
    MaxEvaluations,
    /// Enabled objective-change tolerance detected stagnation.
    ObjectiveStagnation,
    /// Enabled accepted-step tolerance detected stagnation.
    StepStagnation,
    /// Checkpoint belongs to a failed solver.
    Failed,
}

/// Attempted evaluation work, including failed callbacks and rejected trials.
///
/// A combined value/gradient request counts in both categories. Least squares
/// charges residuals as objective calls and Jacobians as gradient calls.
/// Constraint evaluation includes its Jacobians. QP counts represent model states.
/// Only `objective` consumes [`StopOptions::max_evaluations`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EvaluationCounts {
    /// Objective calls, or residual/model-state evaluations in the relevant family.
    pub objective: usize,
    /// Gradient requests; includes least-squares Jacobian requests.
    pub gradient: usize,
    /// Dense Hessian or Hessian-vector requests.
    pub hessian: usize,
    /// Nonlinear constraint evaluations or evaluated QP constraint states.
    pub constraints: usize,
    /// Residual vector requests.
    pub residuals: usize,
    /// Residual/constraint Jacobian requests.
    pub jacobian: usize,
    /// Evaluations of synthetic models, not charged to the outer objective budget.
    pub subproblem: usize,
}
impl std::ops::AddAssign for EvaluationCounts {
    fn add_assign(&mut self, rhs: Self) {
        self.objective += rhs.objective;
        self.gradient += rhs.gradient;
        self.hessian += rhs.hessian;
        self.constraints += rhs.constraints;
        self.residuals += rhs.residuals;
        self.jacobian += rhs.jacobian;
        self.subproblem += rhs.subproblem;
    }
}

pub(crate) trait Report: Clone {
    fn solution(&self) -> &Solution;
    fn solution_mut(&mut self) -> &mut Solution;
}
impl Report for Solution {
    fn solution(&self) -> &Solution {
        self
    }
    fn solution_mut(&mut self) -> &mut Solution {
        self
    }
}

pub(crate) struct Run<R: Report = Solution> {
    pub stop: StopOptions,
    pub counts: EvaluationCounts,
    pub last: Option<R>,
    baseline: Option<Solution>,
}
impl<R: Report> Run<R> {
    pub fn new(stop: &StopOptions) -> Result<Self, Error> {
        stop.validate()?;
        Ok(Self {
            stop: stop.clone(),
            counts: EvaluationCounts::default(),
            last: None,
            baseline: None,
        })
    }
    pub fn charge(&mut self) -> Result<(), Error> {
        if self
            .stop
            .max_evaluations
            .is_some_and(|limit| self.counts.objective >= limit)
        {
            return Err(Error::EvaluationBudgetExceeded);
        }
        self.counts.objective += 1;
        Ok(())
    }
    pub fn evaluate<P: Evaluator + ?Sized>(
        &mut self,
        p: &P,
        x: &Vector,
    ) -> Result<(f64, Vector), Error> {
        vector(x, x.len(), "iterate")?;
        self.charge()?;
        self.counts.gradient += 1;
        evaluate(p, x)
    }
    pub fn value<P: Evaluator + ?Sized>(&mut self, p: &P, x: &Vector) -> Result<f64, Error> {
        vector(x, x.len(), "iterate")?;
        self.charge()?;
        finite(p.evaluate(x.as_slice(), None)?, "objective")
    }
    pub fn hessian<P: Evaluator + ?Sized>(&mut self, p: &P, x: &Vector) -> Result<Matrix, Error> {
        self.counts.hessian += 1;
        hessian(p, x)
    }
    pub fn record(&mut self, mut report: R) -> R {
        report.solution_mut().evaluations = self.counts;
        self.baseline = Some(report.solution().clone());
        self.last = Some(report.clone());
        report
    }
    pub fn checkpoint(&mut self, mut report: R) {
        report.solution_mut().evaluations = self.counts;
        self.last = Some(report);
    }
    pub fn reason(
        &self,
        x: &Vector,
        value: f64,
        iteration: usize,
        converged: bool,
        convergence: TerminationReason,
        accepted_step: bool,
    ) -> Option<TerminationReason> {
        if converged {
            return Some(convergence);
        }
        if self
            .stop
            .max_evaluations
            .is_some_and(|limit| self.counts.objective >= limit)
        {
            return Some(TerminationReason::MaxEvaluations);
        }
        if iteration >= self.stop.max_iterations {
            return Some(TerminationReason::MaxIterations);
        }
        if accepted_step
            && iteration > 0
            && let Some(previous) = &self.baseline
        {
            if let Some(t) = self.stop.step_tolerance
                && t.satisfied((x - &previous.x).norm(), x.norm().max(previous.x.norm()))
            {
                return Some(TerminationReason::StepStagnation);
            }
            if let Some(t) = self.stop.value_tolerance
                && t.satisfied(
                    (value - previous.value).abs(),
                    value.abs().max(previous.value.abs()),
                )
            {
                return Some(TerminationReason::ObjectiveStagnation);
            }
        }
        None
    }
    pub fn finish(mut self, result: Result<R, Error>, solver: &'static str) -> Result<R, Error> {
        match result {
            Ok(mut report) => {
                report.solution_mut().evaluations = self.counts;
                Ok(report)
            }
            Err(Error::EvaluationBudgetExceeded) => {
                let mut last = self.last.take().ok_or(Error::EvaluationBudgetExceeded)?;
                last.solution_mut()
                    .terminate(TerminationReason::MaxEvaluations);
                last.solution_mut().evaluations = self.counts;
                Ok(last)
            }
            Err(source) => {
                if let Some(last) = self.last {
                    let mut last = last.solution().clone();
                    last.terminate(TerminationReason::Failed);
                    last.evaluations = self.counts;
                    Err(Error::SolverFailure {
                        solver,
                        source: Box::new(source),
                        last_solution: Box::new(last),
                    })
                } else {
                    Err(source)
                }
            }
        }
    }
}

pub(crate) fn snapshot(
    x: Vector,
    value: f64,
    norm: f64,
    iteration: usize,
    reason: Option<TerminationReason>,
) -> Solution {
    let mut report = solution(x, value, norm, iteration, false);
    if let Some(reason) = reason {
        report.terminate(reason);
    }
    report
}
