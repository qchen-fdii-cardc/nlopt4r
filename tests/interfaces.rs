mod common;
use common::{Ellipse, Rosenbrock, close, v};
use nlopt4r::{
    Error, EvaluationError, Evaluator, Function, Matrix, Method, Objective, Options, Status,
    StopOptions, TerminationReason, Tolerance, Vector,
    constrained::{self, ConstrainedOptions, ConstrainedProblem, Constraints},
    least_squares::{self, LeastSquares},
    line_search::{self, ScalarMethod, ScalarOptions},
    minimize,
    quadratic::{self, QuadraticProgram},
    trust_region::{self, TrustRegionOptions},
};
use std::cell::Cell;

fn budget(limit: usize) -> Options {
    Options {
        stopping: StopOptions {
            max_evaluations: Some(limit),
            ..StopOptions::default()
        },
        ..Options::default()
    }
}

#[test]
fn buffered_borrowed_callback_counts_match_actual_calls() {
    let target = [3.0, -2.0];
    let calls = Cell::new(0);
    let gradients = Cell::new(0);
    let p = Function::new(|x: &[f64], gradient: Option<&mut [f64]>| {
        calls.set(calls.get() + 1);
        if let Some(g) = gradient {
            gradients.set(gradients.get() + 1);
            for i in 0..x.len() {
                g[i] = x[i] - target[i];
            }
        }
        Ok::<_, EvaluationError>(
            0.5 * x
                .iter()
                .zip(target)
                .map(|(x, t)| (x - t).powi(2))
                .sum::<f64>(),
        )
    });
    let borrowed: &dyn Evaluator = &p;
    let r = minimize(
        borrowed,
        &Vector::zeros(2),
        Method::Bfgs,
        &Options::default(),
    )
    .unwrap();
    close(&r.x, &target, 1e-12);
    assert_eq!(r.termination_reason, TerminationReason::GradientTolerance);
    assert_eq!(r.evaluations.objective, calls.get());
    assert_eq!(r.evaluations.gradient, gradients.get());
    assert_eq!(calls.get(), 3);
    assert_eq!(gradients.get(), 2);
}

#[test]
fn buffered_hessian_callbacks_work_with_newton_and_trust_regions() {
    let calls = Cell::new(0);
    let hessians = Cell::new(0);
    let p = Function::new(|x: &[f64], gradient: Option<&mut [f64]>| {
        calls.set(calls.get() + 1);
        if let Some(g) = gradient {
            g[0] = x[0] - 2.0;
        }
        Ok::<_, EvaluationError>(0.5 * (x[0] - 2.0).powi(2))
    })
    .with_hessian(|_: &[f64]| {
        hessians.set(hessians.get() + 1);
        Ok(Matrix::identity(1, 1))
    });
    let a = minimize(&p, &v(&[0.0]), Method::DampedNewton, &Options::default()).unwrap();
    let before = (calls.get(), hessians.get());
    let b = trust_region::minimize(&p, &v(&[0.0]), &TrustRegionOptions::default()).unwrap();
    for r in [&a, &b] {
        assert!(r.converged());
        close(&r.x, &[2.0], 1e-12);
    }
    assert_eq!(a.evaluations.objective, before.0);
    assert_eq!(a.evaluations.hessian, before.1);
    assert_eq!(b.evaluations.objective, calls.get() - before.0);
    assert_eq!(b.evaluations.hessian, hessians.get() - before.1);
    assert!(b.evaluations.subproblem > 0);
}

#[test]
fn callback_failure_preserves_last_checkpoint_and_source() {
    let p = Function::new(|x: &[f64], gradient: Option<&mut [f64]>| {
        let Some(g) = gradient else {
            return Err(EvaluationError::from("simulation unavailable"));
        };
        g[0] = x[0];
        Ok(0.5 * x[0] * x[0])
    });
    let error = minimize(&p, &v(&[2.0]), Method::Bfgs, &Options::default()).unwrap_err();
    assert!(
        matches!(error.root_cause(), Error::Evaluation { operation: "objective", source }
        if source.message == "simulation unavailable")
    );
    assert!(std::error::Error::source(&error).is_some());
    let last = error.last_solution().unwrap();
    close(&last.x, &[2.0], 1e-12);
    assert_eq!(last.value, 2.0);
    assert_eq!(last.status, Status::Failed);
    assert_eq!(last.evaluations.objective, 2);
    assert_eq!(last.evaluations.gradient, 1);
}

#[test]
fn incomplete_gradient_and_wrong_hessian_are_explicit_errors() {
    let p = Function::new(|_: &[f64], gradient: Option<&mut [f64]>| {
        if let Some(g) = gradient {
            g[0] = 1.0;
        }
        Ok::<_, EvaluationError>(1.0)
    });
    assert!(matches!(
        p.evaluate(&[0.0, 0.0], Some(&mut [0.0])),
        Err(Error::DimensionMismatch {
            context: "gradient buffer",
            expected: (2, 1),
            actual: (1, 1)
        })
    ));
    let error = minimize(&p, &Vector::zeros(2), Method::Bfgs, &Options::default()).unwrap_err();
    assert!(matches!(error.root_cause(), Error::NonFinite("gradient")));
    let p = Function::new(|_: &[f64], gradient: Option<&mut [f64]>| {
        if let Some(g) = gradient {
            g.fill(1.0);
        }
        Ok::<_, EvaluationError>(1.0)
    })
    .with_hessian(|_: &[f64]| Ok(Matrix::zeros(1, 1)));
    let error = minimize(
        &p,
        &Vector::zeros(2),
        Method::DampedNewton,
        &Options::default(),
    )
    .unwrap_err();
    assert!(matches!(
        error.root_cause(),
        Error::DimensionMismatch {
            context: "Hessian",
            expected: (2, 2),
            actual: (1, 1)
        }
    ));
    assert_eq!(error.last_solution().unwrap().evaluations.hessian, 1);
}

#[test]
fn objective_budgets_include_trials_and_never_claim_convergence() {
    let p = (|x: &Vector| 0.5 * x.norm_squared(), |x: &Vector| x.clone());
    for limit in [1, 2] {
        let r = minimize(&p, &v(&[2.0]), Method::Bfgs, &budget(limit)).unwrap();
        assert_eq!(r.status, Status::MaxEvaluations);
        assert_eq!(r.termination_reason, TerminationReason::MaxEvaluations);
        assert!(!r.converged());
        assert_eq!(r.evaluations.objective, limit);
        assert_eq!(r.value, p.value(&r.x));
        assert_eq!(r.gradient_norm, p.gradient(&r.x).norm());
    }
    let r = minimize(&p, &v(&[2.0]), Method::Bfgs, &budget(3)).unwrap();
    assert!(r.converged());
    assert_eq!(r.evaluations.objective, 3);
    assert_eq!(r.termination_reason, TerminationReason::GradientTolerance);
}

#[test]
fn zero_iteration_budget_evaluates_initial_point_once() {
    let options = Options {
        stopping: StopOptions {
            max_iterations: 0,
            ..StopOptions::default()
        },
        ..Options::default()
    };
    let r = minimize(&Rosenbrock, &v(&[-1.2, 1.0]), Method::Bfgs, &options).unwrap();
    assert_eq!(r.termination_reason, TerminationReason::MaxIterations);
    assert_eq!(r.evaluations.objective, 1);
    assert_eq!(r.evaluations.gradient, 1);
    let r = minimize(&Rosenbrock, &v(&[1.0, 1.0]), Method::Bfgs, &options).unwrap();
    assert!(r.converged());
}

#[test]
fn absolute_and_relative_stagnation_do_not_certify_optimality() {
    let p = (
        |x: &Vector| 1000.0 + 0.5 * x.norm_squared(),
        |x: &Vector| x.clone(),
    );
    for (value_tolerance, step_tolerance, reason) in [
        (
            Some(Tolerance::absolute(0.02)),
            None,
            TerminationReason::ObjectiveStagnation,
        ),
        (
            Some(Tolerance {
                absolute: 0.0,
                relative: 1e-4,
            }),
            None,
            TerminationReason::ObjectiveStagnation,
        ),
        (
            None,
            Some(Tolerance::absolute(0.02)),
            TerminationReason::StepStagnation,
        ),
        (
            None,
            Some(Tolerance {
                absolute: 0.0,
                relative: 0.02,
            }),
            TerminationReason::StepStagnation,
        ),
    ] {
        let mut options = Options::default();
        options.stopping.value_tolerance = value_tolerance;
        options.stopping.step_tolerance = step_tolerance;
        options.line_search.initial_step = 0.01;
        let r = minimize(&p, &v(&[1.0]), Method::Bfgs, &options).unwrap();
        assert_eq!(r.status, Status::Stagnated);
        assert_eq!(r.termination_reason, reason);
        assert_eq!(r.iterations, 1);
        assert!(r.gradient_norm > options.stopping.gradient_tolerance);
    }
}

#[test]
fn rejected_trust_steps_do_not_trigger_step_stagnation() {
    let options = TrustRegionOptions {
        stopping: StopOptions {
            step_tolerance: Some(Tolerance::absolute(0.3)),
            ..StopOptions::default()
        },
        ..TrustRegionOptions::default()
    };
    let r = trust_region::minimize(&Rosenbrock, &Vector::zeros(2), &options).unwrap();
    assert_eq!(r.termination_reason, TerminationReason::StepStagnation);
    assert!(r.iterations >= 3, "{r:?}");
    assert!(r.x.norm() > 0.0);
}

#[test]
fn scalar_budget_and_diagnostics_include_endpoint_evaluations() {
    let calls = Cell::new(0);
    let options = ScalarOptions {
        stopping: budget(1).stopping,
        ..ScalarOptions::default()
    };
    let r = line_search::minimize_scalar_with_options(
        |s| {
            calls.set(calls.get() + 1);
            s * s
        },
        (1.0, 3.0),
        ScalarMethod::Parabolic,
        &options,
    )
    .unwrap();
    assert_eq!(r.status, Status::MaxEvaluations);
    assert_eq!(r.x, 1.0);
    assert_eq!(r.value, 1.0);
    assert_eq!(r.evaluations.objective, calls.get());
    let r = line_search::minimize_scalar_with_options(
        |s| (s - 0.25).powi(2),
        (-1.0, 1.0),
        ScalarMethod::Parabolic,
        &ScalarOptions::default(),
    )
    .unwrap();
    assert_eq!(r.termination_reason, TerminationReason::IntervalTolerance);
    assert!((r.x - 0.25).abs() < 1e-6);
}

#[test]
fn evaluation_budgets_are_shared_across_solver_families() {
    let options = budget(2);
    let r = trust_region::minimize(
        &Rosenbrock,
        &v(&[-1.2, 1.0]),
        &TrustRegionOptions {
            stopping: options.stopping.clone(),
            ..TrustRegionOptions::default()
        },
    )
    .unwrap();
    assert_eq!(r.status, Status::MaxEvaluations);
    assert_eq!(r.evaluations.objective, 2);
    let p = (|x: &Vector| x.clone(), |_: &Vector| Matrix::identity(1, 1));
    let r = least_squares::levenberg_marquardt(&p, &v(&[2.0]), &options).unwrap();
    assert_eq!(r.status, Status::MaxEvaluations);
    assert_eq!(r.evaluations.residuals, 2);
    assert_eq!(r.evaluations.jacobian, 1);
    let constrained = ConstrainedOptions {
        solver: options.clone(),
        ..ConstrainedOptions::default()
    };
    for r in [
        constrained::minimize(
            &Ellipse,
            &v(&[0.5, 0.75]),
            constrained::Method::Sqp,
            &constrained,
        )
        .unwrap(),
        constrained::augmented_lagrangian(&Ellipse, &v(&[0.5, 0.75]), &constrained).unwrap(),
    ] {
        assert_eq!(r.solution.status, Status::MaxEvaluations);
        assert_eq!(r.solution.evaluations.objective, 2);
        assert_eq!(r.solution.evaluations.constraints, 2);
    }
}

#[test]
fn qp_phase_i_budget_preserves_original_problem_dimensions() {
    let mut qp = QuadraticProgram::unconstrained(Matrix::identity(1, 1), v(&[0.0]));
    qp.inequalities = Matrix::identity(1, 1);
    qp.inequality_rhs = v(&[2.0]);
    let r = quadratic::minimize(&qp, &v(&[0.0]), &budget(2).stopping).unwrap();
    assert_eq!(r.solution.status, Status::MaxEvaluations);
    assert_eq!(r.solution.evaluations.objective, 2);
    assert_eq!(r.solution.x.len(), 1);
    assert_eq!(r.inequality_multipliers.len(), 1);
    assert_eq!(r.solution.value, 0.5 * r.solution.x.norm_squared());
    let qp = QuadraticProgram::unconstrained(Matrix::identity(1, 1), v(&[0.0]));
    let r = quadratic::minimize(&qp, &v(&[0.0]), &budget(1).stopping).unwrap();
    assert_eq!(
        r.solution.termination_reason,
        TerminationReason::KktTolerance
    );
    assert!(r.solution.converged());
    assert_eq!(r.solution.evaluations.objective, 1);
}

#[test]
fn phase_i_stagnation_is_not_misreported_as_iteration_exhaustion() {
    let mut qp = QuadraticProgram::unconstrained(Matrix::identity(1, 1), v(&[0.0]));
    qp.inequalities = Matrix::identity(1, 1);
    qp.inequality_rhs = v(&[2.0]);
    let options = StopOptions {
        value_tolerance: Some(Tolerance::absolute(1e9)),
        ..StopOptions::default()
    };
    let error = quadratic::minimize(&qp, &v(&[0.0]), &options).unwrap_err();
    assert!(matches!(
        error.root_cause(),
        Error::SubproblemStopped {
            solver: "QP phase I",
            reason: TerminationReason::ObjectiveStagnation,
            ..
        }
    ));
    assert_eq!(error.last_solution().unwrap().x.len(), 1);
}

#[test]
fn sqp_does_not_validate_unused_phr_options() {
    let options = ConstrainedOptions {
        max_inner_iterations: 0,
        initial_penalty: f64::NAN,
        ..ConstrainedOptions::default()
    };
    let r = constrained::minimize(
        &Ellipse,
        &v(&[0.5, 0.75]),
        constrained::Method::Sqp,
        &options,
    )
    .unwrap();
    assert!(r.solution.converged());
    assert!(constrained::augmented_lagrangian(&Ellipse, &v(&[0.5, 0.75]), &options).is_err());
}

struct Plane {
    linear: bool,
}
impl Objective for Plane {
    fn value(&self, x: &Vector) -> f64 {
        0.5 * x.norm_squared()
    }
    fn gradient(&self, x: &Vector) -> Vector {
        x.clone()
    }
}
impl ConstrainedProblem for Plane {
    fn constraints(&self, x: &Vector) -> Constraints {
        let a = Matrix::from_element(1, 3, 1.0);
        if self.linear {
            Constraints::linear(x, &a, &v(&[3.0]), &Matrix::zeros(0, 3), &Vector::zeros(0)).unwrap()
        } else {
            Constraints::equalities_only(v(&[x.sum() - 3.0]), a).unwrap()
        }
    }
}

#[test]
fn equality_and_linear_constructors_work_in_constrained_solves() {
    for linear in [false, true] {
        let r = constrained::minimize(
            &Plane { linear },
            &Vector::zeros(3),
            constrained::Method::Sqp,
            &ConstrainedOptions::default(),
        )
        .unwrap();
        assert!(r.solution.converged());
        assert_eq!(
            r.solution.termination_reason,
            TerminationReason::KktTolerance
        );
        close(&r.solution.x, &[1.0, 1.0, 1.0], 1e-6);
        close(&r.equality_multipliers, &[1.0], 1e-6);
    }
}

struct UpperBound {
    nonpositive: bool,
}
impl Objective for UpperBound {
    fn value(&self, x: &Vector) -> f64 {
        0.5 * (x[0] - 2.0).powi(2)
    }
    fn gradient(&self, x: &Vector) -> Vector {
        v(&[x[0] - 2.0])
    }
}
impl ConstrainedProblem for UpperBound {
    fn constraints(&self, x: &Vector) -> Constraints {
        if self.nonpositive {
            Constraints::from_nonpositive(
                Vector::zeros(0),
                v(&[x[0] - 1.0]),
                Matrix::zeros(0, 1),
                Matrix::from_element(1, 1, 1.0),
            )
            .unwrap()
        } else {
            Constraints::inequalities_only(v(&[1.0 - x[0]]), Matrix::from_element(1, 1, -1.0))
                .unwrap()
        }
    }
}

#[test]
fn nonpositive_adapter_preserves_solution_and_canonical_multiplier_sign() {
    for nonpositive in [false, true] {
        let r = constrained::minimize(
            &UpperBound { nonpositive },
            &v(&[0.0]),
            constrained::Method::Sqp,
            &ConstrainedOptions::default(),
        )
        .unwrap();
        assert!(r.solution.converged());
        close(&r.solution.x, &[1.0], 1e-6);
        close(&r.inequality_multipliers, &[1.0], 1e-6);
    }
    let error = Constraints::linear(
        &v(&[0.0]),
        &Matrix::zeros(1, 2),
        &v(&[0.0]),
        &Matrix::zeros(0, 1),
        &Vector::zeros(0),
    )
    .err()
    .unwrap();
    assert!(matches!(
        error,
        Error::DimensionMismatch {
            context: "linear equalities",
            expected: (1, 1),
            actual: (1, 2)
        }
    ));
}

#[test]
fn nested_solver_exhaustion_reports_context_and_recoverable_checkpoint() {
    let options = ConstrainedOptions {
        max_inner_iterations: 1,
        ..ConstrainedOptions::default()
    };
    let error =
        constrained::augmented_lagrangian(&Ellipse, &Vector::zeros(2), &options).unwrap_err();
    assert!(matches!(
        error.root_cause(),
        Error::IterationLimit {
            solver: "PHR inner BFGS",
            iterations: 1
        }
    ));
    let last = error.last_solution().unwrap();
    assert_eq!(last.status, Status::Failed);
    assert!(last.evaluations.objective > 1);
    assert_eq!(last.value, Ellipse.value(&last.x));
}

struct FailingResiduals;
impl LeastSquares for FailingResiduals {
    fn residuals(&self, x: &Vector) -> Vector {
        x.clone()
    }
    fn jacobian(&self, _: &Vector) -> Matrix {
        Matrix::identity(1, 1)
    }
    fn try_residuals(&self, x: &Vector) -> Result<Vector, EvaluationError> {
        if x[0] < 1.0 {
            Err("measurement missing".into())
        } else {
            Ok(x.clone())
        }
    }
}

#[test]
fn fallible_least_squares_errors_are_not_silently_rejected_trials() {
    let error =
        least_squares::levenberg_marquardt(&FailingResiduals, &v(&[2.0]), &Options::default())
            .unwrap_err();
    assert!(matches!(
        error.root_cause(),
        Error::Evaluation {
            operation: "residuals",
            ..
        }
    ));
    let last = error.last_solution().unwrap();
    close(&last.x, &[4.0 / 3.0], 1e-12);
    assert_eq!(last.value, 0.5 * last.x.norm_squared());
}

#[test]
fn invalid_stopping_options_fail_before_evaluation() {
    for stopping in [
        StopOptions {
            max_evaluations: Some(0),
            ..StopOptions::default()
        },
        StopOptions {
            value_tolerance: Some(Tolerance::absolute(-1.0)),
            ..StopOptions::default()
        },
        StopOptions {
            step_tolerance: Some(Tolerance {
                absolute: 0.0,
                relative: f64::NAN,
            }),
            ..StopOptions::default()
        },
    ] {
        let calls = Cell::new(0);
        let p = Function::new(|_: &[f64], _: Option<&mut [f64]>| {
            calls.set(calls.get() + 1);
            Ok::<_, EvaluationError>(0.0)
        });
        let options = Options {
            stopping,
            ..Options::default()
        };
        assert!(minimize(&p, &v(&[0.0]), Method::Bfgs, &options).is_err());
        assert_eq!(calls.get(), 0);
    }
}
