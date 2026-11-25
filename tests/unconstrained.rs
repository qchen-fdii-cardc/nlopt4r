mod common;
use common::{Rosenbrock, close, v};
use nlopt4r::{Error, Matrix, Method, Objective, Options, Status, StopOptions, Vector, minimize};

fn methods() -> [Method; 11] {
    [
        Method::SteepestDescent,
        Method::NormalizedGradient,
        Method::FletcherReeves,
        Method::DampedNewton,
        Method::ModifiedNewton,
        Method::Bfgs,
        Method::Dfp,
        Method::Sr1,
        Method::Broyden { phi: 0.0 },
        Method::Broyden { phi: 0.5 },
        Method::Broyden { phi: 1.0 },
    ]
}

#[test]
fn all_methods_solve_matlab_rosenbrock_from_standard_start() {
    let options = Options {
        stopping: StopOptions {
            max_iterations: 100_000,
            gradient_tolerance: 1e-5,
            ..StopOptions::default()
        },
        ..Options::default()
    };
    for method in methods() {
        let r = minimize(&Rosenbrock, &v(&[-1.2, 1.0]), method, &options)
            .unwrap_or_else(|e| panic!("{method:?}: {e}"));
        assert!(r.converged(), "{method:?}: {r:?}");
        assert!(
            r.gradient_norm <= options.stopping.gradient_tolerance,
            "{method:?}: {r:?}"
        );
        assert!(r.value < 1e-9, "{method:?}: {r:?}");
        close(&r.x, &[1.0, 1.0], 1e-4);
    }
}

#[test]
fn all_methods_accept_initial_optimum_and_zero_budget() {
    for method in methods() {
        let options = Options {
            stopping: StopOptions {
                max_iterations: 0,
                ..StopOptions::default()
            },
            ..Options::default()
        };
        let r = minimize(&Rosenbrock, &v(&[1.0, 1.0]), method, &options).unwrap();
        assert!(r.converged());
        assert_eq!(r.iterations, 0);
        assert_eq!(r.value, 0.0);
        let r = minimize(&Rosenbrock, &v(&[-1.2, 1.0]), method, &options).unwrap();
        assert_eq!(r.status, Status::MaxIterations);
        close(&r.x, &[-1.2, 1.0], 1e-15);
        assert_eq!(r.value, Rosenbrock.value(&r.x));
    }
}

#[test]
fn arbitrary_dimensions_borrowed_closures_and_trait_objects() {
    let target = Vector::from_iterator(7, (0..7).map(|i| i as f64 - 2.0));
    let weights = Vector::from_iterator(7, (0..7).map(|i| (i + 1) as f64));
    let p = (
        |x: &Vector| 0.5 * (x - &target).component_mul(&weights).dot(&(x - &target)),
        |x: &Vector| (x - &target).component_mul(&weights),
        |_: &Vector| Matrix::from_diagonal(&weights),
    );
    let problem: &dyn Objective = &p;
    for method in methods() {
        let result = minimize(problem, &Vector::zeros(7), method, &Options::default()).unwrap();
        assert!(result.converged(), "{method:?}: {result:?}");
        assert!((&result.x - &target).norm() < 1e-6);
    }
}

#[test]
fn quasi_newton_zero_curvature_is_safe() {
    let p = (|x: &Vector| 0.5 * x.norm_squared(), |x: &Vector| x.clone());
    for method in [
        Method::Sr1,
        Method::Dfp,
        Method::Bfgs,
        Method::Broyden { phi: 0.5 },
    ] {
        let r = minimize(&p, &v(&[3.0]), method, &Options::default()).unwrap();
        assert!(r.converged());
        assert_eq!(r.value, 0.0);
    }
}

#[test]
fn newton_safeguards_indefinite_hessian() {
    let p = (
        |x: &Vector| (x[0] * x[0] - 1.0).powi(2),
        |x: &Vector| v(&[4.0 * x[0] * (x[0] * x[0] - 1.0)]),
        |x: &Vector| Matrix::from_element(1, 1, 12.0 * x[0] * x[0] - 4.0),
    );
    for method in [Method::DampedNewton, Method::ModifiedNewton] {
        let r = minimize(&p, &v(&[0.1]), method, &Options::default()).unwrap();
        assert!(r.converged(), "{r:?}");
        close(&r.x, &[1.0], 1e-6);
    }
}

#[test]
fn malformed_callbacks_and_options_return_errors() {
    let x = v(&[0.0, 0.0]);
    let options = Options::default();
    let missing = (
        |x: &Vector| x.sum(),
        |_: &Vector| Vector::from_element(2, 1.0),
    );
    assert_eq!(
        minimize(&missing, &x, Method::DampedNewton, &options)
            .unwrap_err()
            .root_cause(),
        &Error::MissingHessian
    );
    let bad_g = (|_: &Vector| 1.0, |_: &Vector| v(&[1.0]));
    assert!(matches!(
        minimize(&bad_g, &x, Method::Bfgs, &options)
            .unwrap_err()
            .root_cause(),
        Error::DimensionMismatch { .. }
    ));
    let nan = (|_: &Vector| f64::NAN, |_: &Vector| Vector::zeros(2));
    assert!(matches!(
        minimize(&nan, &x, Method::Bfgs, &options)
            .unwrap_err()
            .root_cause(),
        Error::NonFinite(_)
    ));
    let nan_g = (|_: &Vector| 1.0, |_: &Vector| v(&[f64::INFINITY, 0.0]));
    assert!(matches!(
        minimize(&nan_g, &x, Method::Bfgs, &options)
            .unwrap_err()
            .root_cause(),
        Error::NonFinite(_)
    ));
    let bad_h = (
        |x: &Vector| x.sum(),
        |_: &Vector| v(&[1.0, 1.0]),
        |_: &Vector| Matrix::zeros(1, 1),
    );
    assert!(matches!(
        minimize(&bad_h, &x, Method::DampedNewton, &options)
            .unwrap_err()
            .root_cause(),
        Error::DimensionMismatch { .. }
    ));
    assert!(minimize(&Rosenbrock, &Vector::zeros(0), Method::Bfgs, &options).is_err());
    assert!(minimize(&Rosenbrock, &v(&[f64::NAN, 1.0]), Method::Bfgs, &options).is_err());
    for phi in [-0.1, 1.1, f64::NAN] {
        assert!(minimize(&Rosenbrock, &x, Method::Broyden { phi }, &options).is_err());
    }
    let bad = Options {
        stopping: StopOptions {
            gradient_tolerance: -1.0,
            ..StopOptions::default()
        },
        ..options
    };
    assert!(minimize(&Rosenbrock, &x, Method::Bfgs, &bad).is_err());
}

#[test]
fn final_iteration_checks_convergence_instead_of_reporting_exhaustion() {
    let p = (|x: &Vector| 0.5 * x.norm_squared(), |x: &Vector| x.clone());
    let options = Options {
        stopping: StopOptions {
            max_iterations: 1,
            ..StopOptions::default()
        },
        ..Options::default()
    };
    let r = minimize(&p, &v(&[2.0]), Method::Bfgs, &options).unwrap();
    assert!(r.converged());
    assert_eq!(r.iterations, 1);
}
