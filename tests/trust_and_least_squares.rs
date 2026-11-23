mod common;
use common::{Rosenbrock, close, v};
use nlopt::{Error, Matrix, Options, Status, Vector, least_squares::*, trust_region::*};

fn assert_trust_kkt(g: &Vector, b: &Matrix, radius: f64) {
    let r = subproblem(g, b, radius).unwrap();
    assert!(r.step.norm() <= radius + 1e-8, "{r:?}");
    assert!(r.multiplier >= 0.0);
    let stationarity = (b + r.multiplier * Matrix::identity(g.len(), g.len())) * &r.step + g;
    assert!(stationarity.norm() < 1e-7, "{r:?}, residual={stationarity}");
    assert!((r.multiplier * (r.step.norm() - radius)).abs() < 1e-7, "{r:?}");
    assert!(r.value <= 1e-10);
    let eigen = (b + r.multiplier * Matrix::identity(g.len(), g.len())).symmetric_eigen().eigenvalues;
    assert!(eigen.min() >= -1e-8);
}

#[test]
fn trust_subproblem_interior_boundary_indefinite_and_hard_cases() {
    let identity = Matrix::identity(2, 2);
    assert_trust_kkt(&v(&[1.0, 2.0]), &identity, 10.0);
    assert_trust_kkt(&v(&[1.0, 2.0]), &identity, 0.25);
    let indefinite = Matrix::from_diagonal(&v(&[-2.0, 3.0]));
    assert_trust_kkt(&v(&[1.0, 1.0]), &indefinite, 1.0);
    assert_trust_kkt(&v(&[0.0, 1.0]), &indefinite, 1.0);
    assert_trust_kkt(&Vector::zeros(2), &indefinite, 1.0);
    assert_trust_kkt(&Vector::zeros(2), &Matrix::zeros(2, 2), 1.0);
    let r = subproblem(&v(&[0.0, 1.0]), &indefinite, 1.0).unwrap();
    assert!((r.value + 1.1).abs() < 1e-8);
}

#[test]
fn trust_region_solves_rosenbrock_and_reports_limits() {
    let options = TrustRegionOptions::default();
    for start in [&[-1.2, 1.0][..], &[0.0, 0.0], &[2.0, 2.0]] {
        let r = minimize(&Rosenbrock, &v(start), &options).unwrap();
        assert!(r.converged(), "{r:?}");
        close(&r.x, &[1.0, 1.0], 1e-5);
        assert!(r.value < 1e-12);
    }
    let options = TrustRegionOptions { solver: Options { max_iterations: 0, ..Options::default() }, ..options };
    let r = minimize(&Rosenbrock, &v(&[-1.2, 1.0]), &options).unwrap();
    assert_eq!(r.status, Status::MaxIterations);
}

#[test]
fn trust_region_validates_hessians_radii_and_missing_derivatives() {
    assert!(subproblem(&v(&[1.0]), &Matrix::identity(1, 1), 0.0).is_err());
    assert!(subproblem(&v(&[1.0]), &Matrix::identity(2, 2), 1.0).is_err());
    assert!(subproblem(&v(&[1.0, 1.0]), &Matrix::from_row_slice(2, 2, &[1.0, 3.0, 0.0, 1.0]), 1.0).is_err());
    let p = (|x: &Vector| x.norm_squared(), |x: &Vector| 2.0 * x);
    assert_eq!(minimize(&p, &v(&[1.0]), &TrustRegionOptions::default()).unwrap_err(), Error::MissingHessian);
}

fn residual(x: &Vector) -> Vector {
    v(&[x[0] - 0.7 * x[0].sin() - 0.2 * x[1].cos(),
        x[1] - 0.7 * x[0].cos() + 0.2 * x[1].sin()])
}

fn jacobian(x: &Vector) -> Matrix {
    Matrix::from_row_slice(2, 2, &[
        1.0 - 0.7 * x[0].cos(), 0.2 * x[1].sin(),
        0.7 * x[0].sin(), 1.0 + 0.2 * x[1].cos(),
    ])
}

#[test]
fn lm_solves_matlab_nonlinear_equations() {
    for start in [&[0.0, 0.0][..], &[1.0, 1.0], &[-2.0, 3.0]] {
        let r = levenberg_marquardt(&(residual, jacobian), &v(start), &Options::default()).unwrap();
        assert!(r.converged(), "{r:?}");
        assert!(residual(&r.x).norm() < 1e-5);
        assert!((r.value - 0.5 * residual(&r.x).norm_squared()).abs() < 1e-15);
    }
}

#[test]
fn lm_overdetermined_inconsistent_system_and_rank_deficient_jacobian() {
    let p = (
        |x: &Vector| v(&[x[0] - 1.0, x[0] - 3.0, 2.0 * x[0] - 4.0]),
        |_: &Vector| Matrix::from_column_slice(3, 1, &[1.0, 1.0, 2.0]),
    );
    let r = levenberg_marquardt(&p, &v(&[0.0]), &Options::default()).unwrap();
    assert!(r.converged());
    close(&r.x, &[2.0], 1e-6);
    assert!((r.value - 1.0).abs() < 1e-12);
    let p = (
        |x: &Vector| v(&[x[0] + x[1] - 2.0]),
        |_: &Vector| Matrix::from_row_slice(1, 2, &[1.0, 1.0]),
    );
    let r = levenberg_marquardt(&p, &Vector::zeros(2), &Options::default()).unwrap();
    assert!(r.converged());
    close(&r.x, &[1.0, 1.0], 1e-5);
}

#[test]
fn lm_rejects_invalid_callbacks_and_handles_initial_root() {
    let p = (|_: &Vector| Vector::zeros(0), |_: &Vector| Matrix::zeros(0, 1));
    assert!(levenberg_marquardt(&p, &v(&[0.0]), &Options::default()).is_err());
    let p = (|_: &Vector| v(&[1.0]), |_: &Vector| Matrix::zeros(2, 1));
    assert!(matches!(levenberg_marquardt(&p, &v(&[0.0]), &Options::default()), Err(Error::DimensionMismatch(_))));
    let p = (|x: &Vector| x.clone(), |_: &Vector| Matrix::identity(1, 1));
    let r = levenberg_marquardt(&p, &v(&[0.0]), &Options::default()).unwrap();
    assert!(r.converged());
    assert_eq!(r.iterations, 0);
    assert_eq!(r.value, 0.0);
}
