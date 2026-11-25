#[path = "../common/mod.rs"]
mod common;

use common::{Ellipse, Rosenbrock, close, v};
use nlopt::{Algorithm, Nlopt, ObjFn, SuccessState, Target};
use nlopt4r::{
    Matrix, Method, Objective, Options, Solution, Vector,
    constrained::{self, ConstrainedOptions, ConstrainedProblem, Constraints},
    least_squares::{LeastSquares, levenberg_marquardt},
    line_search::{ScalarMethod, minimize_scalar},
    minimize,
    quadratic::{self, QuadraticProgram},
    trust_region::{self, TrustRegionOptions},
};

fn objective<P: Objective>(x: &[f64], gradient: Option<&mut [f64]>, p: &mut &P) -> f64 {
    let x = v(x);
    if let Some(gradient) = gradient {
        gradient.copy_from_slice(p.gradient(&x).as_slice());
    }
    p.value(&x)
}

fn configure<F: ObjFn<T>, T>(opt: &mut Nlopt<F, T>) {
    opt.set_ftol_abs(1e-13).unwrap();
    opt.set_xtol_rel(1e-11).unwrap();
    opt.set_maxeval(20_000).unwrap();
}

fn finish<F: ObjFn<T>, T>(opt: &mut Nlopt<F, T>, x0: &Vector) -> (Vector, f64) {
    let mut x = x0.as_slice().to_vec();
    let (status, value) = opt
        .optimize(&mut x)
        .expect("NLopt reference optimization failed");
    assert!(
        !matches!(
            status,
            SuccessState::MaxEvalReached | SuccessState::MaxTimeReached
        ),
        "NLopt reference exhausted its budget: {status:?}, x={x:?}, f={value}"
    );
    assert!(value.is_finite() && x.iter().all(|x| x.is_finite()));
    (v(&x), value)
}

fn reference<P: Objective>(p: &P, x0: &Vector, algorithm: Algorithm) -> (Vector, f64) {
    let mut opt = Nlopt::new(algorithm, x0.len(), objective::<P>, Target::Minimize, p);
    configure(&mut opt);
    finish(&mut opt, x0)
}

fn reference_constrained<P: ConstrainedProblem>(p: &P, x0: &Vector) -> (Vector, f64) {
    let mut opt = Nlopt::new(
        Algorithm::Slsqp,
        x0.len(),
        objective::<P>,
        Target::Minimize,
        p,
    );
    configure(&mut opt);
    let c = p.constraints(x0);
    for i in 0..c.equalities.len() {
        opt.add_equality_constraint(
            move |x: &[f64], gradient: Option<&mut [f64]>, p: &mut &P| {
                let c = p.constraints(&v(x));
                if let Some(gradient) = gradient {
                    for (j, g) in gradient.iter_mut().enumerate() {
                        *g = c.equality_jacobian[(i, j)];
                    }
                }
                c.equalities[i]
            },
            p,
            1e-9,
        )
        .unwrap();
    }
    for i in 0..c.inequalities.len() {
        opt.add_inequality_constraint(
            move |x: &[f64], gradient: Option<&mut [f64]>, p: &mut &P| {
                let c = p.constraints(&v(x));
                if let Some(gradient) = gradient {
                    for (j, g) in gradient.iter_mut().enumerate() {
                        *g = -c.inequality_jacobian[(i, j)];
                    }
                }
                // NLopt uses c(x)<=0; nlopt4r uses g(x)>=0.
                -c.inequalities[i]
            },
            p,
            1e-9,
        )
        .unwrap();
    }
    let result = finish(&mut opt, x0);
    let c = p.constraints(&result.0);
    assert!(
        c.equalities.iter().all(|h| h.abs() <= 1e-7),
        "NLopt equality infeasibility"
    );
    assert!(
        c.inequalities.iter().all(|g| *g >= -1e-7),
        "NLopt inequality infeasibility"
    );
    result
}

fn compare<P: Objective>(
    p: &P,
    actual: &Solution,
    reference: &(Vector, f64),
    point_tolerance: f64,
    value_tolerance: f64,
) {
    assert!(actual.converged(), "nlopt4r did not converge: {actual:?}");
    assert!((p.value(&actual.x) - actual.value).abs() <= 1e-12 * (1.0 + actual.value.abs()));
    assert!((p.value(&reference.0) - reference.1).abs() <= 1e-12 * (1.0 + reference.1.abs()));
    assert!(
        (&actual.x - &reference.0).norm() <= point_tolerance,
        "point mismatch: nlopt4r={actual:?}, NLopt={reference:?}"
    );
    assert!(
        (actual.value - reference.1).abs() <= value_tolerance,
        "objective mismatch: nlopt4r={actual:?}, NLopt={reference:?}"
    );
}

#[test]
fn all_unconstrained_methods_agree_with_nlopt_on_rosenbrock() {
    let options = Options {
        max_iterations: 100_000,
        gradient_tolerance: 1e-5,
        ..Options::default()
    };
    for start in [&[-1.2, 1.0][..], &[0.0, 0.0]] {
        let x0 = v(start);
        let reference = reference(&Rosenbrock, &x0, Algorithm::Lbfgs);
        close(&reference.0, &[1.0, 1.0], 1e-5);
        assert!(Rosenbrock.gradient(&reference.0).norm() < 1e-5);
        for method in [
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
        ] {
            let r = minimize(&Rosenbrock, &x0, method, &options)
                .unwrap_or_else(|e| panic!("{method:?}, start={start:?}: {e}"));
            assert!(
                r.gradient_norm <= options.gradient_tolerance,
                "{method:?}: {r:?}"
            );
            compare(&Rosenbrock, &r, &reference, 1e-4, 1e-9);
        }
    }
}

#[test]
fn borrowed_multidimensional_quadratics_agree_with_nlopt() {
    for n in [1, 3, 9] {
        let target = Vector::from_fn(n, |i, _| i as f64 - 2.0);
        let weights = Vector::from_fn(n, |i, _| (i + 1) as f64);
        let p = (
            |x: &Vector| 0.5 * (x - &target).component_mul(&weights).dot(&(x - &target)),
            |x: &Vector| (x - &target).component_mul(&weights),
            |_: &Vector| Matrix::from_diagonal(&weights),
        );
        let x0 = Vector::from_element(n, 4.0);
        let reference = reference(&p, &x0, Algorithm::Lbfgs);
        assert!((&reference.0 - &target).norm() < 1e-6);
        for method in [Method::Bfgs, Method::ModifiedNewton, Method::FletcherReeves] {
            let r = minimize(&p, &x0, method, &Options::default()).unwrap();
            compare(&p, &r, &reference, 1e-5, 1e-10);
        }
    }
}

#[test]
fn bounded_scalar_searches_agree_with_nlopt() {
    let p = (
        |x: &Vector| x[0] * x[0] - x[0].sin(),
        |x: &Vector| v(&[2.0 * x[0] - x[0].cos()]),
    );
    let mut opt = Nlopt::new(Algorithm::Lbfgs, 1, objective::<_>, Target::Minimize, &p);
    configure(&mut opt);
    opt.set_lower_bounds(&[0.0]).unwrap();
    opt.set_upper_bounds(&[1.0]).unwrap();
    let reference = finish(&mut opt, &v(&[0.5]));
    assert!(p.gradient(&reference.0).norm() < 1e-6);
    for method in [ScalarMethod::GoldenSection, ScalarMethod::Parabolic] {
        let r = minimize_scalar(|s| p.value(&v(&[s])), (0.0, 1.0), 1e-10, 200, method).unwrap();
        assert_eq!(r.status, nlopt4r::Status::Converged);
        assert!((r.x - reference.0[0]).abs() < 1e-6, "{r:?}");
        assert!((r.value - reference.1).abs() < 1e-12, "{r:?}");
    }
}

#[test]
fn newton_trust_region_agrees_with_nlopt_on_rosenbrock() {
    for start in [&[-1.2, 1.0][..], &[2.0, 2.0]] {
        let x0 = v(start);
        let reference = reference(&Rosenbrock, &x0, Algorithm::TNewtonPrecondRestart);
        let r = trust_region::minimize(&Rosenbrock, &x0, &TrustRegionOptions::default()).unwrap();
        assert!(r.gradient_norm <= 1e-6);
        compare(&Rosenbrock, &r, &reference, 1e-5, 1e-11);
    }
}

struct Ball {
    gradient: Vector,
    hessian: Matrix,
    radius: f64,
}
impl Objective for Ball {
    fn value(&self, x: &Vector) -> f64 {
        self.gradient.dot(x) + 0.5 * x.dot(&(&self.hessian * x))
    }
    fn gradient(&self, x: &Vector) -> Vector {
        &self.gradient + &self.hessian * x
    }
}
impl ConstrainedProblem for Ball {
    fn constraints(&self, x: &Vector) -> Constraints {
        let mut c = Constraints::empty(x.len());
        c.inequalities = v(&[self.radius.powi(2) - x.norm_squared()]);
        c.inequality_jacobian = Matrix::from_fn(1, x.len(), |_, j| -2.0 * x[j]);
        c
    }
}

#[test]
fn trust_subproblems_agree_with_nlopt_interior_boundary_and_indefinite() {
    for (gradient, diagonal, radius) in [
        ([1.0, 2.0], [1.0, 3.0], 10.0),
        ([1.0, 2.0], [1.0, 3.0], 0.25),
        ([1.0, 1.0], [-2.0, 3.0], 1.0),
        ([0.0, 1.0], [-2.0, 3.0], 1.0),
        ([0.0, 0.0], [-2.0, 3.0], 1.0),
    ] {
        let p = Ball {
            gradient: v(&gradient),
            hessian: Matrix::from_diagonal(&v(&diagonal)),
            radius,
        };
        let r = trust_region::subproblem(&p.gradient, &p.hessian, radius).unwrap();
        assert!(r.step.norm() <= radius + 1e-8);
        assert!(r.multiplier >= 0.0);
        let kkt = p.gradient(&r.step) + r.multiplier * &r.step;
        assert!(kkt.norm() < 1e-7);
        assert!((r.multiplier * (r.step.norm() - radius)).abs() < 1e-7);
        // Nonconvex subproblems need both signs to avoid selecting a local nonglobal minimum.
        let references = [[0.5 * radius, 0.0], [-0.5 * radius, 0.0]]
            .map(|start| reference_constrained(&p, &v(&start)));
        let reference = references
            .iter()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        assert!(
            (r.value - reference.1).abs() < 1e-7,
            "gradient={gradient:?}, diagonal={diagonal:?}, nlopt4r={r:?}, NLopt={reference:?}"
        );
        let actual_value = p.value(&r.step);
        assert!((r.value - actual_value).abs() < 1e-12);
    }
}

struct SquaredResiduals<'a, P>(&'a P);
impl<P: LeastSquares> Objective for SquaredResiduals<'_, P> {
    fn value(&self, x: &Vector) -> f64 {
        0.5 * self.0.residuals(x).norm_squared()
    }
    fn gradient(&self, x: &Vector) -> Vector {
        self.0.jacobian(x).transpose() * self.0.residuals(x)
    }
}

#[test]
fn nonlinear_least_squares_agrees_with_nlopt() {
    let p = (
        |x: &Vector| v(&[x[0].exp() - 2.0, x[1] - 0.7 * x[0].sin()]),
        |x: &Vector| Matrix::from_row_slice(2, 2, &[x[0].exp(), 0.0, -0.7 * x[0].cos(), 1.0]),
    );
    let objective = SquaredResiduals(&p);
    for start in [&[0.0, 0.0][..], &[1.0, 2.0]] {
        let x0 = v(start);
        let reference = reference(&objective, &x0, Algorithm::Lbfgs);
        let r = levenberg_marquardt(&p, &x0, &Options::default()).unwrap();
        close(
            &reference.0,
            &[2.0_f64.ln(), 0.7 * 2.0_f64.ln().sin()],
            1e-5,
        );
        assert!(p.residuals(&r.x).norm() < 1e-5);
        compare(&objective, &r, &reference, 1e-5, 1e-11);
    }
}

#[test]
fn overdetermined_inconsistent_least_squares_agrees_with_nlopt() {
    let a = Matrix::from_row_slice(4, 2, &[1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0]);
    let observations = v(&[3.1, 4.9, 7.2, 8.8]);
    let p = (|x: &Vector| &a * x - &observations, |_: &Vector| a.clone());
    let objective = SquaredResiduals(&p);
    let x0 = Vector::zeros(2);
    let reference = reference(&objective, &x0, Algorithm::Lbfgs);
    let r = levenberg_marquardt(&p, &x0, &Options::default()).unwrap();
    close(&reference.0, &[1.94, 1.15], 1e-6);
    assert!(r.value > 0.01);
    assert!(objective.gradient(&r.x).norm() <= 1e-6);
    compare(&objective, &r, &reference, 1e-5, 1e-11);
}

struct LinearConstraints<'a>(&'a QuadraticProgram);
impl Objective for LinearConstraints<'_> {
    fn value(&self, x: &Vector) -> f64 {
        0.5 * x.dot(&(&self.0.hessian * x)) + self.0.linear.dot(x)
    }
    fn gradient(&self, x: &Vector) -> Vector {
        &self.0.hessian * x + &self.0.linear
    }
}
impl ConstrainedProblem for LinearConstraints<'_> {
    fn constraints(&self, x: &Vector) -> Constraints {
        Constraints {
            equalities: &self.0.equalities * x - &self.0.equality_rhs,
            inequalities: &self.0.inequalities * x - &self.0.inequality_rhs,
            equality_jacobian: self.0.equalities.clone(),
            inequality_jacobian: self.0.inequalities.clone(),
        }
    }
    fn lagrangian_hessian(&self, _: &Vector, _: &Vector, _: &Vector) -> Option<Matrix> {
        Some(self.0.hessian.clone())
    }
}

#[test]
fn equality_and_active_set_qp_agree_with_nlopt_slsqp() {
    let mut equality =
        QuadraticProgram::unconstrained(Matrix::identity(3, 3), v(&[-1.0, -2.0, -3.0]));
    equality.equalities = Matrix::from_row_slice(1, 3, &[1.0, 1.0, 1.0]);
    equality.equality_rhs = v(&[3.0]);
    let mut mixed = QuadraticProgram::unconstrained(
        Matrix::from_row_slice(2, 2, &[2.0, -1.0, -1.0, 4.0]),
        v(&[-1.0, -10.0]),
    );
    mixed.inequalities = Matrix::from_row_slice(3, 2, &[-3.0, -2.0, 1.0, 0.0, 0.0, 1.0]);
    mixed.inequality_rhs = v(&[-6.0, 0.0, 0.0]);
    for qp in [&equality, &mixed] {
        let x0 = Vector::zeros(qp.linear.len());
        let p = LinearConstraints(qp);
        let reference = reference_constrained(&p, &x0);
        let r = quadratic::minimize(qp, &x0, &Options::default()).unwrap();
        assert!(r.solution.gradient_norm < 1e-6);
        assert!(r.constraint_violation < 1e-6);
        assert!(
            r.inequality_multipliers
                .iter()
                .all(|lambda| *lambda >= -1e-7)
        );
        let slack = p.constraints(&r.solution.x).inequalities;
        assert!(slack.component_mul(&r.inequality_multipliers).norm() < 1e-6);
        compare(&p, &r.solution, &reference, 1e-5, 1e-9);
    }
    let r = quadratic::equality_constrained(
        &equality.hessian,
        &equality.linear,
        &equality.equalities,
        &equality.equality_rhs,
    )
    .unwrap();
    let p = LinearConstraints(&equality);
    let x0 = Vector::zeros(3);
    let reference = reference_constrained(&p, &x0);
    close(&reference.0, &[0.0, 1.0, 2.0], 1e-6);
    compare(&p, &r.solution, &reference, 1e-5, 1e-9);
    for method in [
        constrained::Method::Sqp,
        constrained::Method::NewtonLagrange,
        constrained::Method::LagrangeSqp,
    ] {
        let r = constrained::minimize(&p, &x0, method, &ConstrainedOptions::default()).unwrap();
        assert!(r.constraint_violation <= 1e-6);
        assert!(r.solution.gradient_norm <= 1e-6);
        compare(&p, &r.solution, &reference, 1e-5, 1e-9);
    }
}

#[test]
fn nonlinear_sqp_and_phr_agree_with_nlopt_slsqp() {
    let options = ConstrainedOptions::default();
    let expected = [0.5 * (7.0_f64.sqrt() - 1.0), 0.25 * (7.0_f64.sqrt() + 1.0)];
    for start in [&[0.5, 0.75][..], &[2.0, 1.0]] {
        let x0 = v(start);
        let reference = reference_constrained(&Ellipse, &x0);
        close(&reference.0, &expected, 1e-5);
        for r in [
            constrained::minimize(&Ellipse, &x0, constrained::Method::Sqp, &options).unwrap(),
            constrained::augmented_lagrangian(&Ellipse, &x0, &options).unwrap(),
        ] {
            assert!(r.solution.gradient_norm <= options.solver.gradient_tolerance);
            assert!(r.constraint_violation <= options.feasibility_tolerance);
            assert!(r.complementarity <= options.feasibility_tolerance);
            assert!(
                r.inequality_multipliers
                    .iter()
                    .all(|lambda| *lambda >= -1e-8)
            );
            compare(&Ellipse, &r.solution, &reference, 1e-5, 1e-5);
        }
    }
}
