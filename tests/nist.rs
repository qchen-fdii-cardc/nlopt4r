use nlopt4r::least_squares::{LeastSquares, LmOptions, levenberg_marquardt_with_options};
use nlopt4r::{Matrix, StopOptions, Vector};

struct Misra1a;
const PRESSURE: [f64; 14] = [
    77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4, 434.8, 477.3, 536.8, 593.1, 689.1, 760.0,
];
const VOLUME: [f64; 14] = [
    10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82, 50.76, 55.05, 61.01, 66.40, 75.47,
    81.78,
];
impl LeastSquares for Misra1a {
    fn residuals(&self, b: &Vector) -> Vector {
        Vector::from_iterator(
            14,
            PRESSURE
                .iter()
                .zip(VOLUME)
                .map(|(x, y)| -b[0] * (-b[1] * x).exp_m1() - y),
        )
    }
    fn jacobian(&self, b: &Vector) -> Matrix {
        Matrix::from_fn(14, 2, |i, j| {
            if j == 0 {
                -(-b[1] * PRESSURE[i]).exp_m1()
            } else {
                b[0] * PRESSURE[i] * (-b[1] * PRESSURE[i]).exp()
            }
        })
    }
}
#[test]
fn qr_lm_matches_nist_certified_misra1a_fit_from_both_published_starts() {
    let options = LmOptions {
        stopping: StopOptions {
            max_iterations: 1000,
            max_evaluations: Some(5000),
            gradient_tolerance: 1e-8,
            ..Default::default()
        },
        ..Default::default()
    };
    for start in [[500.0, 0.0001], [250.0, 0.0005]] {
        let r = levenberg_marquardt_with_options(
            &Misra1a,
            &Vector::from_column_slice(&start),
            &options,
        )
        .unwrap();
        assert!(r.converged(), "{r:?}");
        assert!((r.x[0] - 238.94212918).abs() < 1e-4, "{r:?}");
        assert!((r.x[1] - 0.00055015643181).abs() < 1e-9);
        assert!((2.0 * r.value - 0.12455138894).abs() < 1e-10);
    }
}
