use nlopt4r::quadratic::{QuadraticProgram, minimize};
use nlopt4r::utils::{ChaCha8Rng, Rng, SeedableRng};
use nlopt4r::{Matrix, StopOptions, Vector};

#[test]
fn constructed_kkt_problems_preserve_optima_with_redundancy_scaling_and_permutation() {
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    for seed in 0..12 {
        let n = 5;
        let target = Vector::from_fn(n, |_, _| rng.random_range(0.2..2.0));
        let a = Matrix::from_fn(n, n, |_, _| rng.random_range(-1.0..1.0));
        let h: Matrix = a.transpose() * a + 0.5 * Matrix::identity(n, n);
        let ae = Matrix::from_fn(3, n, |i, j| match i {
            0 => {
                if j == 0 {
                    1.0
                } else {
                    0.2 * j as f64
                }
            }
            1 => {
                if j == 1 {
                    1.0
                } else {
                    -0.1 * j as f64
                }
            }
            _ => {
                if j == 0 {
                    2.0
                } else {
                    0.4 * j as f64
                }
            }
        });
        let mu = Vector::from_vec(vec![0.3, -0.7, 0.0]);
        let c = -&h * &target + ae.transpose() * mu;
        for scale in [0.2, 5.0] {
            let perm = |i: usize| if seed % 2 == 0 { n - 1 - i } else { i };
            let qp = QuadraticProgram {
                hessian: Matrix::from_fn(n, n, |i, j| scale * h[(perm(i), perm(j))]),
                linear: Vector::from_fn(n, |i, _| scale * c[perm(i)]),
                equalities: Matrix::from_fn(3, n, |i, j| ae[(i, perm(j))]),
                equality_rhs: &ae * &target,
                inequalities: Matrix::identity(n, n),
                inequality_rhs: Vector::zeros(n),
            };
            let r = minimize(
                &qp,
                &Vector::zeros(n),
                &StopOptions {
                    gradient_tolerance: 1e-7,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(r.solution.converged(), "{r:?}");
            let expected = Vector::from_fn(n, |i, _| target[perm(i)]);
            assert!((&r.solution.x - expected).norm() < 1e-5, "{r:?}");
            assert!(r.constraint_violation < 1e-7);
            assert!(r.solution.x.component_mul(&r.inequality_multipliers).norm() < 1e-7);
            let stationarity = &qp.hessian * &r.solution.x + &qp.linear
                - qp.equalities.transpose() * &r.equality_multipliers
                - qp.inequalities.transpose() * &r.inequality_multipliers;
            assert!(stationarity.norm() < 1e-6);
        }
    }
}
