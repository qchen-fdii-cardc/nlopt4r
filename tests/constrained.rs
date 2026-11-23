mod common;
use common::{close, v};
use nlopt::{Error, Matrix, Objective, Options, Status, Vector, constrained::*};

struct Ellipse;
impl Objective for Ellipse {
    fn value(&self, x: &Vector) -> f64 {
        (x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2)
    }
    fn gradient(&self, x: &Vector) -> Vector {
        v(&[2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)])
    }
}
impl ConstrainedProblem for Ellipse {
    fn constraints(&self, x: &Vector) -> Constraints {
        Constraints {
            equalities: v(&[x[0] - 2.0 * x[1] + 1.0]),
            inequalities: v(&[1.0 - 0.25 * x[0] * x[0] - x[1] * x[1]]),
            equality_jacobian: Matrix::from_row_slice(1, 2, &[1.0, -2.0]),
            inequality_jacobian: Matrix::from_row_slice(1, 2, &[-0.5 * x[0], -2.0 * x[1]]),
        }
    }
}

struct Cylinder;
impl Objective for Cylinder {
    fn value(&self, x: &Vector) -> f64 {
        -std::f64::consts::PI * x[0] * x[0] * x[1]
    }
    fn gradient(&self, x: &Vector) -> Vector {
        let pi = std::f64::consts::PI;
        v(&[-2.0 * pi * x[0] * x[1], -pi * x[0] * x[0]])
    }
}
impl ConstrainedProblem for Cylinder {
    fn constraints(&self, x: &Vector) -> Constraints {
        let pi = std::f64::consts::PI;
        Constraints {
            equalities: v(&[pi * x[0] * x[1] + pi * x[0] * x[0] - 150.0]),
            inequalities: x.clone(),
            equality_jacobian: Matrix::from_row_slice(
                1,
                2,
                &[pi * x[1] + 2.0 * pi * x[0], pi * x[0]],
            ),
            inequality_jacobian: Matrix::identity(2, 2),
        }
    }
}

struct EqualityCircle;
impl Objective for EqualityCircle {
    fn value(&self, x: &Vector) -> f64 {
        -x[0] - x[1]
    }
    fn gradient(&self, _: &Vector) -> Vector {
        v(&[-1.0, -1.0])
    }
}
impl ConstrainedProblem for EqualityCircle {
    fn constraints(&self, x: &Vector) -> Constraints {
        let mut c = Constraints::empty(2);
        c.equalities = v(&[x.norm_squared() - 1.0]);
        c.equality_jacobian = Matrix::from_row_slice(1, 2, &[2.0 * x[0], 2.0 * x[1]]);
        c
    }
    fn lagrangian_hessian(&self, _: &Vector, mu: &Vector, _: &Vector) -> Option<Matrix> {
        Some(-2.0 * mu[0] * Matrix::identity(2, 2))
    }
}

fn check(r: &ConstrainedSolution, options: &ConstrainedOptions) {
    assert!(r.solution.converged(), "{r:?}");
    assert!(
        r.solution.gradient_norm <= options.solver.gradient_tolerance,
        "{r:?}"
    );
    assert!(
        r.constraint_violation <= options.feasibility_tolerance,
        "{r:?}"
    );
    assert!(r.complementarity <= options.feasibility_tolerance, "{r:?}");
    assert!(
        r.inequality_multipliers.iter().all(|v| *v >= -1e-8),
        "{r:?}"
    );
}

#[test]
fn phr_matches_matlab_ellipse_analytic_solution() {
    let options = ConstrainedOptions::default();
    let expected = [0.5 * (7.0_f64.sqrt() - 1.0), 0.25 * (7.0_f64.sqrt() + 1.0)];
    for start in [&[0.0, 0.0][..], &[2.0, 1.0], &[0.5, 0.75]] {
        let r = augmented_lagrangian(&Ellipse, &v(start), &options).unwrap();
        check(&r, &options);
        close(&r.solution.x, &expected, 1e-5);
        assert!(r.inner_iterations > 0);
    }
}

#[test]
fn sqp_solves_matlab_ellipse() {
    let options = ConstrainedOptions::default();
    let r = minimize(&Ellipse, &v(&[0.5, 0.75]), Method::Sqp, &options).unwrap();
    check(&r, &options);
    close(
        &r.solution.x,
        &[0.5 * (7.0_f64.sqrt() - 1.0), 0.25 * (7.0_f64.sqrt() + 1.0)],
        1e-5,
    );
}

#[test]
fn sqp_matches_matlab_cylinder_analytic_optimum() {
    let options = ConstrainedOptions::default();
    let radius = (50.0 / std::f64::consts::PI).sqrt();
    for start in [&[3.0, 5.0][..], &[4.0, 8.0]] {
        let r = minimize(&Cylinder, &v(start), Method::Sqp, &options).unwrap();
        check(&r, &options);
        close(&r.solution.x, &[radius, 2.0 * radius], 1e-5);
        assert!((r.solution.value + 100.0 * radius).abs() < 1e-4);
    }
}

#[test]
fn equality_sqp_newton_and_phr_solve_nonlinear_constraint() {
    let options = ConstrainedOptions::default();
    for method in [Method::Sqp, Method::NewtonLagrange, Method::LagrangeSqp] {
        let r = minimize(&EqualityCircle, &v(&[0.8, 0.8]), method, &options).unwrap();
        check(&r, &options);
        close(&r.solution.x, &[0.5_f64.sqrt(), 0.5_f64.sqrt()], 1e-5);
        close(&r.equality_multipliers, &[-0.5_f64.sqrt()], 1e-5);
    }
    let r = augmented_lagrangian(&EqualityCircle, &v(&[0.8, 0.8]), &options).unwrap();
    check(&r, &options);
}

struct InactiveInequality;
impl Objective for InactiveInequality {
    fn value(&self, x: &Vector) -> f64 {
        0.5 * (x[0] - 2.0).powi(2)
    }
    fn gradient(&self, x: &Vector) -> Vector {
        v(&[x[0] - 2.0])
    }
}
impl ConstrainedProblem for InactiveInequality {
    fn constraints(&self, x: &Vector) -> Constraints {
        let mut c = Constraints::empty(1);
        c.inequalities = x.clone();
        c.inequality_jacobian = Matrix::identity(1, 1);
        c
    }
}

#[test]
fn inactive_inequality_uses_projected_phr_gradient() {
    let options = ConstrainedOptions::default();
    for start in [-1.0, 0.0, 5.0] {
        let r = augmented_lagrangian(&InactiveInequality, &v(&[start]), &options).unwrap();
        check(&r, &options);
        close(&r.solution.x, &[2.0], 1e-5);
        close(&r.inequality_multipliers, &[0.0], 1e-8);
        let r = minimize(&InactiveInequality, &v(&[start]), Method::Sqp, &options).unwrap();
        check(&r, &options);
        close(&r.solution.x, &[2.0], 1e-5);
    }
}

#[test]
fn constrained_budget_and_invalid_inputs() {
    let options = ConstrainedOptions {
        solver: Options {
            max_iterations: 0,
            ..Options::default()
        },
        ..ConstrainedOptions::default()
    };
    let r = augmented_lagrangian(&Ellipse, &v(&[0.0, 0.0]), &options).unwrap();
    assert_eq!(r.solution.status, Status::MaxIterations);
    let r = minimize(&Ellipse, &v(&[0.0, 0.0]), Method::Sqp, &options).unwrap();
    assert_eq!(r.solution.status, Status::MaxIterations);
    assert!(matches!(
        minimize(&Ellipse, &v(&[0.5, 0.75]), Method::NewtonLagrange, &options),
        Err(Error::InvalidInput(_))
    ));
}

struct BadConstraint;
impl Objective for BadConstraint {
    fn value(&self, x: &Vector) -> f64 {
        x.norm_squared()
    }
    fn gradient(&self, x: &Vector) -> Vector {
        2.0 * x
    }
}
impl ConstrainedProblem for BadConstraint {
    fn constraints(&self, _: &Vector) -> Constraints {
        Constraints {
            equalities: v(&[0.0]),
            inequalities: Vector::zeros(0),
            equality_jacobian: Matrix::zeros(2, 2),
            inequality_jacobian: Matrix::zeros(0, 2),
        }
    }
}

#[test]
fn malformed_constraint_jacobian_is_error_not_panic() {
    for method in [Method::Sqp, Method::NewtonLagrange] {
        assert!(matches!(
            minimize(
                &BadConstraint,
                &v(&[1.0, 1.0]),
                method,
                &ConstrainedOptions::default()
            ),
            Err(Error::DimensionMismatch(_))
        ));
    }
    assert!(matches!(
        augmented_lagrangian(
            &BadConstraint,
            &v(&[1.0, 1.0]),
            &ConstrainedOptions::default()
        ),
        Err(Error::DimensionMismatch(_))
    ));
}
