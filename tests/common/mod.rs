#![allow(dead_code)]
use nlopt4r::constrained::{ConstrainedProblem, Constraints};
use nlopt4r::{Matrix, Objective, Vector};

pub struct Ellipse;
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

pub struct Rosenbrock;
impl Objective for Rosenbrock {
    fn value(&self, x: &Vector) -> f64 {
        100.0 * (x[0] * x[0] - x[1]).powi(2) + (x[0] - 1.0).powi(2)
    }
    fn gradient(&self, x: &Vector) -> Vector {
        Vector::from_vec(vec![
            400.0 * x[0] * (x[0] * x[0] - x[1]) + 2.0 * (x[0] - 1.0),
            -200.0 * (x[0] * x[0] - x[1]),
        ])
    }
    fn hessian(&self, x: &Vector) -> Option<Matrix> {
        Some(Matrix::from_row_slice(
            2,
            2,
            &[
                1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0,
                -400.0 * x[0],
                -400.0 * x[0],
                200.0,
            ],
        ))
    }
}

pub fn v(x: &[f64]) -> Vector {
    Vector::from_column_slice(x)
}

pub fn close(actual: &Vector, expected: &[f64], tolerance: f64) {
    assert!(
        (actual - v(expected)).norm() < tolerance,
        "actual={actual:?}, expected={expected:?}"
    );
}
