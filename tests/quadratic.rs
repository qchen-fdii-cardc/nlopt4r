mod common;
use common::{close, v};
use nlopt4r::{Error, Matrix, Status, StopOptions, Vector, quadratic::*};

fn check_kkt(qp: &QuadraticProgram, r: &QuadraticSolution) {
    assert!(r.solution.converged(), "{r:?}");
    assert!(r.constraint_violation < 1e-6, "{r:?}");
    assert!(r.solution.gradient_norm < 1e-6, "{r:?}");
    assert!(
        r.inequality_multipliers.iter().all(|v| *v >= -1e-7),
        "{r:?}"
    );
    let slack = &qp.inequalities * &r.solution.x - &qp.inequality_rhs;
    assert!(
        slack.component_mul(&r.inequality_multipliers).norm() < 1e-6,
        "{r:?}"
    );
}

#[test]
fn matlab_active_set_example_from_feasible_and_infeasible_starts() {
    let mut qp = QuadraticProgram::unconstrained(
        Matrix::from_row_slice(2, 2, &[2.0, -1.0, -1.0, 4.0]),
        v(&[-1.0, -10.0]),
    );
    qp.inequalities = Matrix::from_row_slice(3, 2, &[-3.0, -2.0, 1.0, 0.0, 0.0, 1.0]);
    qp.inequality_rhs = v(&[-6.0, 0.0, 0.0]);
    for start in [&[0.0, 0.0][..], &[5.0, -3.0], &[1.0, 1.0]] {
        let r = minimize(&qp, &v(start), &StopOptions::default()).unwrap();
        check_kkt(&qp, &r);
        close(&r.solution.x, &[0.5, 2.25], 1e-6);
        close(&r.inequality_multipliers, &[0.75, 0.0, 0.0], 1e-6);
    }
}

#[test]
fn equality_qp_analytic_solution_and_multiplier_sign() {
    let h = Matrix::identity(3, 3);
    let a = Matrix::from_row_slice(1, 3, &[1.0, 1.0, 1.0]);
    let r = equality_constrained(&h, &Vector::zeros(3), &a, &v(&[3.0])).unwrap();
    close(&r.solution.x, &[1.0, 1.0, 1.0], 1e-8);
    close(&r.equality_multipliers, &[1.0], 1e-8);
    assert!((r.solution.value - 1.5).abs() < 1e-10);
}

#[test]
fn active_set_removes_wrong_constraints_and_handles_mixed_constraints() {
    let mut qp = QuadraticProgram::unconstrained(Matrix::identity(3, 3), v(&[-1.0, -2.0, -3.0]));
    qp.equalities = Matrix::from_row_slice(1, 3, &[1.0, 1.0, 1.0]);
    qp.equality_rhs = v(&[3.0]);
    qp.inequalities = Matrix::identity(3, 3);
    qp.inequality_rhs = Vector::zeros(3);
    for start in [&[3.0, 0.0, 0.0][..], &[0.0, 0.0, 0.0], &[-5.0, -2.0, -3.0]] {
        let r = minimize(&qp, &v(start), &StopOptions::default()).unwrap();
        check_kkt(&qp, &r);
        close(&r.solution.x, &[0.0, 1.0, 2.0], 1e-6);
    }
}

#[test]
fn redundant_constraints_and_positive_semidefinite_hessians() {
    let mut qp =
        QuadraticProgram::unconstrained(Matrix::from_diagonal(&v(&[2.0, 0.0])), v(&[-2.0, 0.0]));
    qp.equalities = Matrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 2.0]);
    qp.equality_rhs = v(&[2.0, 4.0]);
    qp.inequalities = Matrix::from_row_slice(3, 2, &[1.0, 0.0, 2.0, 0.0, 1.0, 0.0]);
    qp.inequality_rhs = v(&[0.0, 0.0, 0.0]);
    let r = minimize(&qp, &v(&[0.0, 2.0]), &StopOptions::default()).unwrap();
    check_kkt(&qp, &r);
    close(&r.solution.x, &[1.0, 2.0], 1e-6);
}

#[test]
fn infeasible_equalities_and_inequalities_are_reported() {
    let mut qp = QuadraticProgram::unconstrained(Matrix::identity(1, 1), v(&[0.0]));
    qp.equalities = Matrix::from_element(2, 1, 1.0);
    qp.equality_rhs = v(&[1.0, 2.0]);
    assert_eq!(
        minimize(&qp, &v(&[0.0]), &StopOptions::default())
            .unwrap_err()
            .root_cause(),
        &Error::Infeasible
    );
    qp.equalities = Matrix::zeros(0, 1);
    qp.equality_rhs = Vector::zeros(0);
    qp.inequalities = Matrix::from_column_slice(2, 1, &[1.0, -1.0]);
    qp.inequality_rhs = v(&[2.0, -1.0]);
    assert_eq!(
        minimize(&qp, &v(&[0.0]), &StopOptions::default())
            .unwrap_err()
            .root_cause(),
        &Error::Infeasible
    );
}

#[test]
fn invalid_qp_and_budget_are_not_success() {
    let mut qp = QuadraticProgram::unconstrained(Matrix::identity(1, 1), v(&[-2.0]));
    let options = StopOptions {
        max_iterations: 0,
        ..StopOptions::default()
    };
    assert_eq!(
        minimize(&qp, &v(&[0.0]), &options).unwrap().solution.status,
        Status::MaxIterations
    );
    qp.hessian[(0, 0)] = -1.0;
    assert!(matches!(
        minimize(&qp, &v(&[0.0]), &options),
        Err(Error::InvalidInput(_))
    ));
    qp.hessian[(0, 0)] = f64::NAN;
    assert!(matches!(
        minimize(&qp, &v(&[0.0]), &options),
        Err(Error::NonFinite(_))
    ));
    qp.hessian = Matrix::identity(2, 2);
    assert!(matches!(
        minimize(&qp, &v(&[0.0]), &options),
        Err(Error::DimensionMismatch { .. })
    ));
}

#[test]
fn singular_flat_and_unbounded_systems_have_explicit_contracts() {
    let r = equality_constrained(
        &Matrix::zeros(1, 1),
        &v(&[0.0]),
        &Matrix::zeros(0, 1),
        &Vector::zeros(0),
    )
    .unwrap();
    close(&r.solution.x, &[0.0], 1e-10);
    let e = equality_constrained(
        &Matrix::zeros(1, 1),
        &v(&[1.0]),
        &Matrix::zeros(0, 1),
        &Vector::zeros(0),
    )
    .unwrap_err();
    assert_eq!(e, Error::SingularSystem);
}

#[test]
fn small_step_with_large_gradient_is_not_false_convergence() {
    let qp = QuadraticProgram::unconstrained(Matrix::from_element(1, 1, 1e8), v(&[-1.0]));
    let r = minimize(&qp, &v(&[0.0]), &StopOptions::default()).unwrap();
    check_kkt(&qp, &r);
    assert!((r.solution.x[0] - 1e-8).abs() < 1e-15, "{r:?}");
}
