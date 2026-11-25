mod common;
use common::v;
use nlopt4r::{
    Matrix, Objective, Vector,
    constrained::{self, ConstrainedOptions, ConstrainedProblem, Constraints, Method},
};

struct FiveVariable;
fn q(x: &Vector) -> f64 {
    x[0].powi(3) + x[1].powi(3) + 1.0
}
fn dq(x: &Vector) -> Vector {
    v(&[3.0 * x[0].powi(2), 3.0 * x[1].powi(2), 0.0, 0.0, 0.0])
}
fn product_except(x: &Vector, exclude: &[usize]) -> f64 {
    x.iter()
        .enumerate()
        .filter(|(i, _)| !exclude.contains(i))
        .map(|(_, x)| x)
        .product()
}
impl Objective for FiveVariable {
    fn value(&self, x: &Vector) -> f64 {
        x.iter().product::<f64>().exp() - 0.5 * q(x).powi(2)
    }
    fn gradient(&self, x: &Vector) -> Vector {
        let exp = x.iter().product::<f64>().exp();
        Vector::from_fn(5, |i, _| exp * product_except(x, &[i])) - q(x) * dq(x)
    }
}
impl ConstrainedProblem for FiveVariable {
    fn constraints(&self, x: &Vector) -> Constraints {
        let mut c = Constraints::empty(5);
        c.equalities = v(&[
            x.norm_squared() - 10.0,
            x[1] * x[2] - 5.0 * x[3] * x[4],
            q(x),
        ]);
        c.equality_jacobian = Matrix::from_row_slice(
            3,
            5,
            &[
                2.0 * x[0],
                2.0 * x[1],
                2.0 * x[2],
                2.0 * x[3],
                2.0 * x[4],
                0.0,
                x[2],
                x[1],
                -5.0 * x[4],
                -5.0 * x[3],
                3.0 * x[0] * x[0],
                3.0 * x[1] * x[1],
                0.0,
                0.0,
                0.0,
            ],
        );
        c
    }
    fn lagrangian_hessian(&self, x: &Vector, mu: &Vector, _: &Vector) -> Option<Matrix> {
        let exp = x.iter().product::<f64>().exp();
        let du = Vector::from_fn(5, |i, _| product_except(x, &[i]));
        let d2u = Matrix::from_fn(5, 5, |i, j| {
            if i == j {
                0.0
            } else {
                product_except(x, &[i, j])
            }
        });
        let d2q = Matrix::from_diagonal(&v(&[6.0 * x[0], 6.0 * x[1], 0.0, 0.0, 0.0]));
        let mut h = exp * (&du * du.transpose() + d2u)
            - dq(x) * dq(x).transpose()
            - (q(x) + mu[2]) * d2q
            - 2.0 * mu[0] * Matrix::identity(5, 5);
        h[(1, 2)] -= mu[1];
        h[(2, 1)] -= mu[1];
        h[(3, 4)] += 5.0 * mu[1];
        h[(4, 3)] += 5.0 * mu[1];
        Some(h)
    }
}

#[test]
fn exact_equality_methods_solve_matlab_five_variable_problem() {
    let options = ConstrainedOptions::default();
    for method in [Method::NewtonLagrange, Method::LagrangeSqp, Method::Sqp] {
        let r = constrained::minimize(
            &FiveVariable,
            &v(&[-1.8, 1.7, 1.9, -0.8, -0.8]),
            method,
            &options,
        )
        .unwrap();
        assert!(r.solution.converged(), "{method:?}: {r:?}");
        assert!(
            r.solution.gradient_norm <= options.solver.stopping.gradient_tolerance,
            "{r:?}"
        );
        assert!(
            r.constraint_violation <= options.feasibility_tolerance,
            "{r:?}"
        );
        assert!((r.solution.value - 0.0539498478).abs() < 1e-7, "{r:?}");
    }
}

struct FourVariable;
impl Objective for FourVariable {
    fn value(&self, x: &Vector) -> f64 {
        x.norm_squared() - 2.0 * x[0] - 3.0 * x[3]
    }
    fn gradient(&self, x: &Vector) -> Vector {
        2.0 * x - v(&[2.0, 0.0, 0.0, 3.0])
    }
}
impl ConstrainedProblem for FourVariable {
    fn constraints(&self, x: &Vector) -> Constraints {
        let a = Matrix::from_row_slice(2, 4, &[2.0, 1.0, 1.0, 4.0, 1.0, 1.0, 2.0, 1.0]);
        Constraints {
            equalities: &a * x - v(&[7.0, 6.0]),
            equality_jacobian: a,
            inequalities: x.clone(),
            inequality_jacobian: Matrix::identity(4, 4),
        }
    }
}

#[test]
fn root_matlab_four_variable_example_agrees_with_direct_qp() {
    let options = ConstrainedOptions::default();
    let c = FourVariable.constraints(&Vector::zeros(4));
    let qp = nlopt4r::quadratic::QuadraticProgram {
        hessian: 2.0 * Matrix::identity(4, 4),
        linear: v(&[-2.0, 0.0, 0.0, -3.0]),
        equalities: c.equality_jacobian,
        equality_rhs: -c.equalities,
        inequalities: c.inequality_jacobian,
        inequality_rhs: -c.inequalities,
    };
    let expected =
        nlopt4r::quadratic::minimize(&qp, &Vector::zeros(4), &options.solver.stopping).unwrap();
    for r in [
        constrained::minimize(&FourVariable, &Vector::zeros(4), Method::Sqp, &options).unwrap(),
        constrained::augmented_lagrangian(&FourVariable, &Vector::zeros(4), &options).unwrap(),
    ] {
        assert!(r.solution.converged(), "{r:?}");
        assert!(r.constraint_violation < 1e-6);
        assert!(
            (&r.solution.x - &expected.solution.x).norm() < 1e-5,
            "{r:?}"
        );
    }
}
