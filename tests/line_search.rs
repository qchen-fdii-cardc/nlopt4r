mod common;
use common::v;
use nlopt::{Error, Status, Vector, line_search::*};

#[test]
fn armijo_satisfies_actual_sufficient_decrease() {
    let p = (|x: &Vector| 10.0 * x.norm_squared(), |x: &Vector| 20.0 * x);
    let x = v(&[1.0, -2.0]);
    let d = -p.1(&x);
    let options = ArmijoOptions::default();
    let r = armijo(&p, &x, &d, &options).unwrap();
    assert!(r.alpha > 0.0 && r.alpha < 1.0);
    assert!(r.value <= p.0(&x) + options.sufficient_decrease * r.alpha * p.1(&x).dot(&d));
    assert_eq!(r.value, p.0(&(&x + r.alpha * d)));
    assert!(r.evaluations > 1);
}

#[test]
fn armijo_rejects_non_descent_and_exhaustion() {
    let p = (|x: &Vector| 10.0 * x.norm_squared(), |x: &Vector| 20.0 * x);
    assert!(matches!(
        armijo(&p, &v(&[1.0]), &v(&[1.0]), &ArmijoOptions::default()),
        Err(Error::InvalidInput(_))
    ));
    let options = ArmijoOptions {
        max_iterations: 1,
        ..ArmijoOptions::default()
    };
    assert_eq!(
        armijo(&p, &v(&[1.0]), &v(&[-20.0]), &options).unwrap_err(),
        Error::LineSearchFailed
    );
}

#[test]
fn scalar_methods_match_quadratic_and_matlab_phi() {
    for method in [ScalarMethod::GoldenSection, ScalarMethod::Parabolic] {
        let r =
            minimize_scalar(|s| (s - 2.3).powi(2) + 4.0, (-3.0, 5.0), 1e-10, 200, method).unwrap();
        assert_eq!(r.status, Status::Converged);
        assert!((r.x - 2.3).abs() < 1e-6, "{r:?}");
        assert!((r.value - 4.0).abs() < 1e-12);
        let r = minimize_scalar(|s| s * s - s.sin(), (0.0, 1.0), 1e-10, 200, method).unwrap();
        assert_eq!(r.status, Status::Converged);
        assert!((2.0 * r.x - r.x.cos()).abs() < 1e-6, "{r:?}");
        assert!((r.x - 0.4501836113).abs() < 1e-6);
    }
}

#[test]
fn scalar_boundaries_flat_functions_and_budget() {
    for method in [ScalarMethod::GoldenSection, ScalarMethod::Parabolic] {
        let r = minimize_scalar(|s| s, (1.0, 3.0), 1e-9, 200, method).unwrap();
        assert_eq!(r.x, 1.0);
        assert!(r.bracket.0 <= r.x && r.x <= r.bracket.1);
        let r = minimize_scalar(|_| 7.0, (-1.0, 1.0), 1e-9, 200, method).unwrap();
        assert_eq!(r.status, Status::Converged);
        assert_eq!(r.value, 7.0);
        let r = minimize_scalar(|s| s * s, (-1.0, 1.0), 1e-9, 0, method).unwrap();
        assert_eq!(r.status, Status::MaxIterations);
        assert_eq!(r.iterations, 0);
    }
}

#[test]
fn invalid_search_inputs_are_explicit_errors() {
    let method = ScalarMethod::Parabolic;
    for bounds in [
        (1.0, 1.0),
        (3.0, 1.0),
        (f64::NAN, 1.0),
        (0.0, f64::INFINITY),
    ] {
        assert!(minimize_scalar(|s| s * s, bounds, 1e-8, 200, method).is_err());
    }
    assert!(minimize_scalar(|_| f64::NAN, (0.0, 1.0), 1e-8, 200, method).is_err());
    let p = (|x: &Vector| x.norm_squared(), |x: &Vector| 2.0 * x);
    for contraction in [0.0, 1.0, f64::NAN] {
        let options = ArmijoOptions {
            contraction,
            ..ArmijoOptions::default()
        };
        assert!(armijo(&p, &v(&[1.0]), &v(&[-1.0]), &options).is_err());
    }
}
