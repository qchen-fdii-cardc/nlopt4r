use nlopt4r::{
    Bounds, Error, EvaluationError, Matrix, Objective, Options, Status, StopOptions, Vector,
    bounded::{projected_gradient, projected_gradient_norm},
    constrained::{self, ConstrainedProblem, Constraints, SqpOptions, sqp_with_restoration},
    least_squares::{LeastSquares, LmOptions, levenberg_marquardt_with_options},
};
use std::cell::Cell;

fn v(x: &[f64]) -> Vector {
    Vector::from_column_slice(x)
}

#[test]
fn qr_lm_handles_rectangular_rank_deficiency_and_extreme_column_scaling() {
    for scale in [1e-200, 1e-10, 1.0, 1e10, 1e200] {
        let p = (
            |x: &Vector| v(&[scale * x[0] + x[1] - 2.0]),
            |_: &Vector| Matrix::from_row_slice(1, 3, &[scale, 1.0, 0.0]),
        );
        let r =
            levenberg_marquardt_with_options(&p, &Vector::zeros(3), &LmOptions::default()).unwrap();
        assert!(r.converged(), "{r:?}");
        assert!((scale * r.x[0] + r.x[1] - 2.0).abs() < 1e-6);
        assert!(r.x[2].abs() < 1e-15);
    }
    let p = (
        |x: &Vector| v(&[x[0] - 1.0, x[0] - 3.0, 2.0 * x[0] - 4.0]),
        |_: &Vector| Matrix::from_column_slice(3, 1, &[1.0, 1.0, 2.0]),
    );
    let r = levenberg_marquardt_with_options(&p, &v(&[0.0]), &LmOptions::default()).unwrap();
    assert!(r.converged(), "{r:?}");
    assert!((r.x[0] - 2.0).abs() < 1e-6);
    assert!((r.value - 1.0).abs() < 1e-12);
}

#[test]
fn qr_lm_rejected_trials_are_charged_and_checkpoint_is_recoverable() {
    let calls = Cell::new(0);
    let p = (
        |x: &Vector| {
            calls.set(calls.get() + 1);
            v(&[x[0].powi(2) - 1.0])
        },
        |x: &Vector| Matrix::from_element(1, 1, 2.0 * x[0]),
    );
    let mut options = LmOptions::default();
    options.stopping.max_evaluations = Some(2);
    let r = levenberg_marquardt_with_options(&p, &v(&[0.1]), &options).unwrap();
    assert_eq!(r.status, Status::MaxEvaluations);
    assert_eq!(r.x[0], 0.1);
    assert_eq!(calls.get(), 2);
    assert_eq!(r.evaluations.objective, 2);
    assert_eq!(r.evaluations.residuals, 2);
    options.stopping.max_evaluations = None;
    let r = levenberg_marquardt_with_options(&p, &v(&[0.1]), &options).unwrap();
    assert!(r.converged(), "{r:?}");
    assert!((r.x[0] - 1.0).abs() < 1e-6);
}

struct FailedResidual;
impl LeastSquares for FailedResidual {
    fn residuals(&self, x: &Vector) -> Vector {
        x.clone()
    }
    fn jacobian(&self, _: &Vector) -> Matrix {
        Matrix::identity(1, 1)
    }
    fn try_residuals(&self, x: &Vector) -> Result<Vector, EvaluationError> {
        if x[0] < 0.5 {
            Err(EvaluationError::new("trial rejected"))
        } else {
            Ok(x.clone())
        }
    }
}

#[test]
fn qr_lm_fallible_trial_retains_last_valid_report() {
    let error =
        levenberg_marquardt_with_options(&FailedResidual, &v(&[1.0]), &LmOptions::default())
            .unwrap_err();
    assert!(matches!(
        error.root_cause(),
        Error::Evaluation {
            operation: "residuals",
            ..
        }
    ));
    let last = error.last_solution().unwrap();
    assert_eq!(last.x[0], 1.0);
    assert_eq!(last.evaluations.residuals, 2);
    assert_eq!(last.status, Status::Failed);
}

#[test]
fn projected_gradient_keeps_every_callback_feasible_and_tests_box_stationarity() {
    let bounds = Bounds::new(vec![0.0, -2.0, 4.0], vec![1.0, 2.0, 4.0]).unwrap();
    let p = (
        |x: &Vector| {
            assert!(bounds.contains(x.as_slice()));
            (x[0] - 3.0).powi(2) + (x[1] - 0.5).powi(2) + x[2].powi(2)
        },
        |x: &Vector| {
            assert!(bounds.contains(x.as_slice()));
            v(&[2.0 * (x[0] - 3.0), 2.0 * (x[1] - 0.5), 2.0 * x[2]])
        },
    );
    let r = projected_gradient(&p, &v(&[-1.0, 8.0, 0.0]), &bounds, &Options::default()).unwrap();
    assert!(r.converged(), "{r:?}");
    assert!((r.x[0] - 1.0).abs() < 1e-6);
    assert!((r.x[1] - 0.5).abs() < 1e-6);
    assert_eq!(r.x[2], 4.0);
    assert!(p.gradient(&r.x).norm() > 1.0);
    assert!(projected_gradient_norm(&r.x, &p.gradient(&r.x), &bounds).unwrap() < 1e-6);
    let options = Options {
        stopping: StopOptions {
            max_evaluations: Some(2),
            ..StopOptions::default()
        },
        ..Options::default()
    };
    let r = projected_gradient(&p, &v(&[0.0, -2.0, 4.0]), &bounds, &options).unwrap();
    assert_eq!(r.status, Status::MaxEvaluations);
    assert_eq!(r.evaluations.objective, 2);
    assert!(bounds.contains(r.x.as_slice()));
}

struct NonlinearInterval;
impl Objective for NonlinearInterval {
    fn value(&self, x: &Vector) -> f64 {
        (x[0] - 1.5).powi(2)
    }
    fn gradient(&self, x: &Vector) -> Vector {
        v(&[2.0 * (x[0] - 1.5)])
    }
}
impl ConstrainedProblem for NonlinearInterval {
    fn constraints(&self, x: &Vector) -> Constraints {
        Constraints::inequalities_only(
            v(&[x[0].powi(2) - 1.0, 2.0 - x[0]]),
            Matrix::from_column_slice(2, 1, &[2.0 * x[0], -1.0]),
        )
        .unwrap()
    }
}

#[test]
fn sqp_restores_an_infeasible_linearization_then_resumes_sqp() {
    let options = SqpOptions::default();
    let legacy = constrained::minimize(
        &NonlinearInterval,
        &v(&[0.1]),
        constrained::Method::Sqp,
        &options.solver,
    )
    .unwrap_err();
    assert_eq!(legacy.root_cause(), &Error::Infeasible);
    for start in [0.1, 0.2] {
        let r = sqp_with_restoration(&NonlinearInterval, &v(&[start]), &options).unwrap();
        assert!(r.constrained.solution.converged(), "{r:?}");
        assert!((r.constrained.solution.x[0] - 1.5).abs() < 1e-5);
        assert!(!r.restoration.is_empty());
        for event in &r.restoration {
            assert_eq!(event.trigger.root_cause(), &Error::Infeasible);
            assert!(event.failure.is_none());
            assert!(event.final_violation < event.initial_violation);
        }
    }
}

struct ImpossibleEquality;
impl Objective for ImpossibleEquality {
    fn value(&self, x: &Vector) -> f64 {
        x.norm_squared()
    }
    fn gradient(&self, x: &Vector) -> Vector {
        2.0 * x
    }
}
impl ConstrainedProblem for ImpossibleEquality {
    fn constraints(&self, _: &Vector) -> Constraints {
        Constraints::equalities_only(v(&[1.0]), Matrix::zeros(1, 1)).unwrap()
    }
}

#[test]
fn sqp_restoration_failure_and_strict_budgets_are_reported() {
    let e =
        sqp_with_restoration(&ImpossibleEquality, &v(&[0.0]), &SqpOptions::default()).unwrap_err();
    assert!(matches!(
        e.source.root_cause(),
        Error::SubproblemStopped {
            reason: nlopt4r::TerminationReason::StepStagnation,
            ..
        }
    ));
    assert!(e.source.last_solution().is_some());
    let full = e.last_constrained.as_ref().unwrap();
    assert_eq!(full.equality_multipliers.len(), 1);
    assert_eq!(full.inequality_multipliers.len(), 0);
    assert_eq!(&full.solution, e.source.last_solution().unwrap());
    assert_eq!(full.solution.status, Status::Failed);
    let source = std::error::Error::source(&e).unwrap();
    assert!(source.downcast_ref::<Error>().is_some());
    assert!(std::error::Error::source(&e.source).is_some());
    assert_eq!(e.restoration.len(), 1);
    assert!(matches!(
        e.restoration[0].failure,
        Some(Error::SubproblemStopped { .. })
    ));
    let mut options = SqpOptions {
        max_subproblem_evaluations: 1,
        ..SqpOptions::default()
    };
    let e = sqp_with_restoration(&NonlinearInterval, &v(&[0.1]), &options).unwrap_err();
    assert!(matches!(
        e.source.root_cause(),
        Error::SubproblemStopped { .. }
    ));
    assert!(e.source.last_solution().unwrap().evaluations.subproblem <= 1);
    options.max_subproblem_evaluations = 10000;
    options.solver.solver.stopping.max_evaluations = Some(2);
    let r = sqp_with_restoration(&NonlinearInterval, &v(&[0.1]), &options).unwrap();
    assert_eq!(r.constrained.solution.status, Status::MaxEvaluations);
    assert_eq!(r.constrained.solution.evaluations.objective, 2);
    assert_eq!(r.restoration.len(), 1);
    assert_eq!(
        r.restoration[0].failure,
        Some(Error::EvaluationBudgetExceeded)
    );
}

struct FallibleEquality;
impl Objective for FallibleEquality {
    fn value(&self, x: &Vector) -> f64 {
        0.5 * x.norm_squared()
    }
    fn gradient(&self, x: &Vector) -> Vector {
        x.clone()
    }
}
impl ConstrainedProblem for FallibleEquality {
    fn constraints(&self, x: &Vector) -> Constraints {
        Constraints::equalities_only(
            v(&[x[0].powi(2) - 1.0]),
            Matrix::from_element(1, 1, 2.0 * x[0]),
        )
        .unwrap()
    }
    fn try_constraints(&self, x: &Vector) -> Result<Constraints, EvaluationError> {
        if x[0] < 1.1 {
            Err(EvaluationError::new("constraint callback failed"))
        } else {
            Ok(self.constraints(x))
        }
    }
}

#[test]
fn sqp_preserves_nonzero_multipliers_on_callback_failure_and_budget_stop() {
    let e =
        sqp_with_restoration(&FallibleEquality, &v(&[2.0]), &SqpOptions::default()).unwrap_err();
    assert!(matches!(
        e.source.root_cause(),
        Error::Evaluation {
            operation: "constraints",
            ..
        }
    ));
    let last = e.last_constrained.unwrap();
    assert!((last.solution.x[0] - 1.25).abs() < 1e-10);
    assert!((last.equality_multipliers[0] - 0.3125).abs() < 1e-10);
    assert_eq!(&last.solution, e.source.last_solution().unwrap());
    assert_eq!(last.solution.evaluations.objective, 3);
    assert_eq!(last.solution.evaluations.constraints, 3);
    let mut options = SqpOptions::default();
    options.solver.solver.stopping.max_evaluations = Some(2);
    let r = sqp_with_restoration(&FallibleEquality, &v(&[2.0]), &options).unwrap();
    assert_eq!(r.constrained.solution.status, Status::MaxEvaluations);
    assert!((r.constrained.solution.x[0] - 1.25).abs() < 1e-10);
    assert!((r.constrained.equality_multipliers[0] - 0.3125).abs() < 1e-10);
    assert_eq!(r.constrained.solution.evaluations.objective, 2);
}
