mod common;

use std::cell::Cell;

use common::{Rosenbrock, close, v};
use nlopt4r::{
    Error, EvaluationError, Evaluator, Function, StopOptions, TerminationReason, Vector,
    local::{
        HessianVector, LbfgsOptions, NewtonCgOptions, SteihaugOptions, SteihaugTermination,
        StrongWolfeOptions, TrustRegionCgOptions, lbfgs, newton_cg, steihaug, strong_wolfe,
        trust_region_cg,
    },
};

impl HessianVector for Rosenbrock {
    fn hessian_vector(&self, x: &[f64], d: &[f64], out: &mut [f64]) -> Result<(), Error> {
        out[0] = (1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0) * d[0] - 400.0 * x[0] * d[1];
        out[1] = -400.0 * x[0] * d[0] + 200.0 * d[1];
        Ok(())
    }
}

struct Diagonal {
    weights: Vector,
    calls: Cell<usize>,
    products: Cell<usize>,
}

impl Diagonal {
    fn new(weights: Vector) -> Self {
        Self {
            weights,
            calls: Cell::new(0),
            products: Cell::new(0),
        }
    }
    fn reset(&self) {
        self.calls.set(0);
        self.products.set(0);
    }
}

impl Evaluator for Diagonal {
    fn evaluate(&self, x: &[f64], gradient: Option<&mut [f64]>) -> Result<f64, Error> {
        self.calls.set(self.calls.get() + 1);
        if let Some(g) = gradient {
            for i in 0..x.len() {
                g[i] = self.weights[i] * x[i];
            }
        }
        Ok(0.5
            * x.iter()
                .enumerate()
                .map(|(i, x)| self.weights[i] * x * x)
                .sum::<f64>())
    }
}

impl HessianVector for Diagonal {
    fn hessian_vector(&self, _: &[f64], d: &[f64], out: &mut [f64]) -> Result<(), Error> {
        self.products.set(self.products.get() + 1);
        for i in 0..d.len() {
            out[i] = self.weights[i] * d[i];
        }
        Ok(())
    }
}

#[test]
fn strong_wolfe_brackets_zooms_and_satisfies_both_inequalities() {
    let f0 = 9.0;
    let g0 = v(&[-6.0]);
    let d = v(&[1.0]);
    for initial_step in [0.1, 10.0] {
        let options = StrongWolfeOptions {
            curvature: 0.1,
            initial_step,
            ..Default::default()
        };
        let calls = Cell::new(0);
        let step = strong_wolfe(f0, &g0, &d, &options, |a| {
            calls.set(calls.get() + 1);
            Ok(((a - 3.0).powi(2), v(&[2.0 * (a - 3.0)])))
        })
        .unwrap();
        assert!(step.value <= f0 + options.sufficient_decrease * step.alpha * g0.dot(&d));
        assert!(step.gradient.dot(&d).abs() <= -options.curvature * g0.dot(&d));
        assert_eq!(step.evaluations, calls.get());
        assert!(calls.get() > 1);
    }
}

#[test]
fn strong_wolfe_enforces_trial_cap_and_propagates_failures() {
    let options = StrongWolfeOptions {
        max_evaluations: 1,
        ..Default::default()
    };
    let calls = Cell::new(0);
    let error = strong_wolfe(1.0, &v(&[-2.0]), &v(&[1.0]), &options, |_| {
        calls.set(calls.get() + 1);
        Ok((100.0, v(&[10.0])))
    })
    .unwrap_err();
    assert!(matches!(
        error,
        Error::LineSearchFailed { evaluations: 1, .. }
    ));
    assert_eq!(calls.get(), 1);
    let error = strong_wolfe(1.0, &v(&[-2.0]), &v(&[1.0]), &options, |_| {
        Err(Error::NonFinite("simulation"))
    })
    .unwrap_err();
    assert!(matches!(error, Error::NonFinite("simulation")));
    assert!(
        strong_wolfe(1.0, &v(&[2.0]), &v(&[1.0]), &options, |_| {
            panic!("non-descent direction must not evaluate");
        })
        .is_err()
    );
}

#[test]
fn lbfgs_solves_rosenbrock_with_short_history() {
    let options = LbfgsOptions {
        history_size: 5,
        stopping: StopOptions {
            gradient_tolerance: 1e-7,
            max_iterations: 1000,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = lbfgs(&Rosenbrock, &v(&[-1.2, 1.0]), &options).unwrap();
    assert!(result.converged(), "{result:?}");
    close(&result.x, &[1.0, 1.0], 1e-6);
    assert!(result.value < 1e-14);
    assert_eq!(result.evaluations.objective, result.evaluations.gradient);
    assert_eq!(result.evaluations.hessian, 0);
}

#[test]
fn lbfgs_solves_ill_conditioned_spd_without_dense_storage() {
    let p = Diagonal::new(Vector::from_iterator(
        20,
        (0..20).map(|i| 10.0_f64.powf(4.0 * i as f64 / 19.0)),
    ));
    let result = lbfgs(
        &p,
        &Vector::from_element(20, 1.0),
        &LbfgsOptions {
            history_size: 10,
            stopping: StopOptions {
                max_iterations: 5000,
                gradient_tolerance: 1e-6,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    assert!(result.converged(), "{result:?}");
    assert!(result.x.norm() < 1e-6);
    assert_eq!(result.evaluations.objective, p.calls.get());
    assert_eq!(p.products.get(), 0);
}

#[test]
fn lbfgs_reuses_accepted_evaluation_and_preserves_callback_error_checkpoint() {
    let p = Diagonal::new(v(&[1.0]));
    let result = lbfgs(&p, &v(&[2.0]), &LbfgsOptions::default()).unwrap();
    assert!(result.converged());
    assert_eq!(p.calls.get(), 2);
    assert_eq!(result.evaluations.objective, 2);

    let calls = Cell::new(0);
    let failing = Function::new(|x: &[f64], gradient: Option<&mut [f64]>| {
        calls.set(calls.get() + 1);
        if calls.get() > 1 {
            return Err(EvaluationError::from("gradient unavailable"));
        }
        if let Some(g) = gradient {
            g[0] = x[0];
        }
        Ok(0.5 * x[0] * x[0])
    });
    let error = lbfgs(&failing, &v(&[2.0]), &LbfgsOptions::default()).unwrap_err();
    assert!(
        matches!(error.root_cause(), Error::Evaluation { source, .. }
        if source.message == "gradient unavailable")
    );
    let last = error.last_solution().unwrap();
    close(&last.x, &[2.0], 1e-15);
    assert_eq!(last.value, 2.0);
    assert_eq!(last.evaluations.objective, calls.get());
    assert_eq!(last.evaluations.gradient, 2);
    assert_eq!(last.termination_reason, TerminationReason::Failed);
}

#[test]
fn all_scalable_solvers_enforce_strict_objective_budgets() {
    let p = Diagonal::new(v(&[1.0, 10.0]));
    let x = v(&[1.0, 1.0]);
    for budget in [1, 2, 3] {
        let stop = StopOptions {
            max_evaluations: Some(budget),
            ..Default::default()
        };
        let a = lbfgs(
            &p,
            &x,
            &LbfgsOptions {
                stopping: stop.clone(),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(a.evaluations.objective, p.calls.get());
        assert_eq!(a.evaluations.objective, budget);
        assert_eq!(a.termination_reason, TerminationReason::MaxEvaluations);
        p.reset();
        let b = newton_cg(
            &p,
            &x,
            &NewtonCgOptions {
                stopping: stop.clone(),
                max_cg_iterations: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(b.evaluations.objective, p.calls.get());
        assert_eq!(b.evaluations.hessian, p.products.get());
        assert_eq!(b.evaluations.objective, budget);
        assert_eq!(b.termination_reason, TerminationReason::MaxEvaluations);
        p.reset();
        let c = trust_region_cg(
            &p,
            &x,
            &TrustRegionCgOptions {
                stopping: stop,
                subproblem: SteihaugOptions {
                    max_iterations: 1,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(c.evaluations.objective, p.calls.get());
        assert_eq!(c.evaluations.hessian, p.products.get());
        assert_eq!(c.evaluations.objective, budget);
        assert_eq!(c.termination_reason, TerminationReason::MaxEvaluations);
        p.reset();
    }
}

#[test]
fn newton_cg_solves_spd_and_rosenbrock_using_only_products() {
    let p = Diagonal::new(v(&[1.0, 100.0, 10000.0]));
    let result = newton_cg(
        &p,
        &v(&[1.0, 1.0, 1.0]),
        &NewtonCgOptions {
            cg_tolerance: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(result.converged(), "{result:?}");
    assert!(result.x.norm() < 1e-6);
    assert_eq!(result.evaluations.objective, p.calls.get());
    assert_eq!(result.evaluations.hessian, p.products.get());
    assert!(p.products.get() > 0);
    let result = newton_cg(&Rosenbrock, &v(&[-1.2, 1.0]), &NewtonCgOptions::default()).unwrap();
    assert!(result.converged(), "{result:?}");
    close(&result.x, &[1.0, 1.0], 1e-5);
}

#[test]
fn steihaug_handles_interior_boundary_and_negative_curvature() {
    let options = SteihaugOptions {
        relative_tolerance: 1e-10,
        ..Default::default()
    };
    let g = v(&[1.0, 2.0]);
    let a = steihaug(&g, 10.0, &options, |d| Ok(2.0 * d)).unwrap();
    assert_eq!(a.termination, SteihaugTermination::ResidualTolerance);
    assert!((a.step + 0.5 * &g).norm() < 1e-14);
    assert!((a.value + 1.25).abs() < 1e-14);
    let b = steihaug(&g, 0.25, &options, |d| Ok(2.0 * d)).unwrap();
    assert_eq!(b.termination, SteihaugTermination::Boundary);
    assert!((b.step.norm() - 0.25).abs() < 1e-14);
    assert!(b.value < 0.0);
    let c = steihaug(&g, 2.0, &options, |d| Ok(-d)).unwrap();
    assert_eq!(c.termination, SteihaugTermination::NegativeCurvature);
    assert!((c.step.norm() - 2.0).abs() < 1e-14);
    assert!(g.dot(&c.step) < 0.0);
    assert!((c.value - (g.dot(&c.step) - 0.5 * c.step.norm_squared())).abs() < 1e-14);
}

#[test]
fn steihaug_detects_negative_curvature_after_a_positive_cg_iteration() {
    let g = v(&[1.0, 0.2]);
    let h = |d: &Vector| Ok(v(&[2.0 * d[0], -d[1]]));
    let result = steihaug(
        &g,
        10.0,
        &SteihaugOptions {
            relative_tolerance: 1e-12,
            ..Default::default()
        },
        h,
    )
    .unwrap();
    assert_eq!(result.termination, SteihaugTermination::NegativeCurvature);
    assert_eq!(result.iterations, 2);
    assert!((result.step.norm() - 10.0).abs() < 1e-12);
    let p = &result.step;
    assert!((result.value - (g.dot(p) + p[0] * p[0] - 0.5 * p[1] * p[1])).abs() < 1e-12);
    assert!(result.value < -1.0);
}

#[test]
fn steihaug_iteration_cap_and_invalid_operator_are_explicit() {
    let result = steihaug(
        &v(&[1.0, 1.0]),
        100.0,
        &SteihaugOptions {
            max_iterations: 1,
            relative_tolerance: 1e-12,
        },
        |d| Ok(v(&[d[0], 10.0 * d[1]])),
    )
    .unwrap();
    assert_eq!(result.termination, SteihaugTermination::MaxIterations);
    assert_eq!(result.hessian_products, 1);
    assert!(matches!(
        steihaug(&v(&[1.0, 1.0]), 1.0, &SteihaugOptions::default(), |_| Ok(
            v(&[1.0])
        )),
        Err(Error::DimensionMismatch { .. })
    ));
    assert!(matches!(
        steihaug(&v(&[1.0]), 1.0, &SteihaugOptions::default(), |_| Ok(v(&[
            f64::NAN
        ]))),
        Err(Error::NonFinite(_))
    ));
}

#[test]
fn trust_cg_solves_rosenbrock_and_large_matrix_free_quadratic() {
    let result = trust_region_cg(
        &Rosenbrock,
        &v(&[-1.2, 1.0]),
        &TrustRegionCgOptions::default(),
    )
    .unwrap();
    assert!(result.converged(), "{result:?}");
    close(&result.x, &[1.0, 1.0], 1e-5);
    let p = Diagonal::new(Vector::from_iterator(
        2000,
        (0..2000).map(|i| 1.0 + (i % 2) as f64),
    ));
    let result = trust_region_cg(
        &p,
        &Vector::from_element(2000, 1.0),
        &TrustRegionCgOptions::default(),
    )
    .unwrap();
    assert!(result.converged(), "{result:?}");
    assert!(result.x.norm() < 1e-6);
    assert_eq!(result.evaluations.objective, p.calls.get());
    assert_eq!(result.evaluations.hessian, p.products.get());
    p.reset();
    let result = lbfgs(
        &p,
        &Vector::from_element(2000, 1.0),
        &LbfgsOptions {
            history_size: 3,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(result.converged(), "{result:?}");
    assert!(result.x.norm() < 1e-6);
    assert_eq!(result.evaluations.objective, p.calls.get());
    assert_eq!(result.evaluations.hessian, 0);
}

#[test]
fn nonconvex_minimizers_safeguard_negative_curvature() {
    struct DoubleWell;
    impl Evaluator for DoubleWell {
        fn evaluate(&self, x: &[f64], gradient: Option<&mut [f64]>) -> Result<f64, Error> {
            if let Some(g) = gradient {
                g[0] = x[0].powi(3) - x[0];
            }
            Ok(0.25 * x[0].powi(4) - 0.5 * x[0].powi(2))
        }
    }
    impl HessianVector for DoubleWell {
        fn hessian_vector(&self, x: &[f64], d: &[f64], out: &mut [f64]) -> Result<(), Error> {
            out[0] = (3.0 * x[0] * x[0] - 1.0) * d[0];
            Ok(())
        }
    }
    let a = newton_cg(&DoubleWell, &v(&[0.25]), &NewtonCgOptions::default()).unwrap();
    let b = trust_region_cg(&DoubleWell, &v(&[0.25]), &TrustRegionCgOptions::default()).unwrap();
    for result in [a, b] {
        assert!(result.converged(), "{result:?}");
        close(&result.x, &[1.0], 1e-6);
        assert!((result.value + 0.25).abs() < 1e-12);
    }
}

#[test]
fn rejected_trust_trial_at_budget_returns_accepted_checkpoint() {
    struct InexactModel;
    impl Evaluator for InexactModel {
        fn evaluate(&self, x: &[f64], gradient: Option<&mut [f64]>) -> Result<f64, Error> {
            if let Some(g) = gradient {
                g[0] = 100.0 * x[0];
            }
            Ok(50.0 * x[0] * x[0])
        }
    }
    impl HessianVector for InexactModel {
        fn hessian_vector(&self, _: &[f64], d: &[f64], out: &mut [f64]) -> Result<(), Error> {
            out[0] = d[0];
            Ok(())
        }
    }
    let result = trust_region_cg(
        &InexactModel,
        &v(&[0.1]),
        &TrustRegionCgOptions {
            stopping: StopOptions {
                max_evaluations: Some(2),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.termination_reason, TerminationReason::MaxEvaluations);
    close(&result.x, &[0.1], 1e-15);
    assert!((result.value - 0.5).abs() < 1e-14);
    assert_eq!(result.evaluations.objective, 2);
    assert_eq!(result.evaluations.gradient, 2);
    assert_eq!(result.evaluations.hessian, 1);
}
#[test]
fn failed_or_incomplete_hessian_products_preserve_valid_checkpoint() {
    struct BrokenProduct {
        fail: bool,
    }
    impl Evaluator for BrokenProduct {
        fn evaluate(&self, x: &[f64], gradient: Option<&mut [f64]>) -> Result<f64, Error> {
            if let Some(g) = gradient {
                g.copy_from_slice(x);
            }
            Ok(0.5 * x.iter().map(|x| x * x).sum::<f64>())
        }
    }
    impl HessianVector for BrokenProduct {
        fn hessian_vector(&self, _: &[f64], _: &[f64], _: &mut [f64]) -> Result<(), Error> {
            if self.fail {
                Err(Error::Evaluation {
                    operation: "Hessian-vector",
                    source: EvaluationError::from("operator unavailable"),
                })
            } else {
                Ok(())
            }
        }
    }
    for fail in [true, false] {
        let p = BrokenProduct { fail };
        let a = newton_cg(&p, &v(&[2.0]), &NewtonCgOptions::default()).unwrap_err();
        let b = trust_region_cg(&p, &v(&[2.0]), &TrustRegionCgOptions::default()).unwrap_err();
        for error in [a, b] {
            let last = error.last_solution().unwrap();
            close(&last.x, &[2.0], 1e-15);
            assert_eq!(last.value, 2.0);
            assert_eq!(last.evaluations.objective, 1);
            assert_eq!(last.evaluations.hessian, 1);
            if fail {
                assert!(
                    matches!(error.root_cause(), Error::Evaluation { source, .. }
                    if source.message == "operator unavailable")
                );
            } else {
                assert!(matches!(error.root_cause(), Error::NonFinite(_)));
            }
        }
    }
}
