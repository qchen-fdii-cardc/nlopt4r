//! Formula-based benchmarks. See docs/BENCHMARKS.md for domains and proofs.
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug)]
struct Dual {
    v: f64,
    d: f64,
}
impl Dual {
    fn constant(v: f64) -> Self {
        Self { v, d: 0.0 }
    }
    fn exp(self) -> Self {
        let v = self.v.exp();
        Self { v, d: v * self.d }
    }
    fn ln(self) -> Self {
        Self {
            v: self.v.ln(),
            d: self.d / self.v,
        }
    }
    fn sin(self) -> Self {
        Self {
            v: self.v.sin(),
            d: self.d * self.v.cos(),
        }
    }
    fn cos(self) -> Self {
        Self {
            v: self.v.cos(),
            d: -self.d * self.v.sin(),
        }
    }
    fn cosh(self) -> Self {
        Self {
            v: self.v.cosh(),
            d: self.d * self.v.sinh(),
        }
    }
    fn pow(self, n: i32) -> Self {
        Self {
            v: self.v.powi(n),
            d: if n == 0 {
                0.0
            } else {
                n as f64 * self.v.powi(n - 1) * self.d
            },
        }
    }
    fn abs(self) -> Self {
        Self {
            v: self.v.abs(),
            d: self.v.signum() * self.d,
        }
    }
    fn sqrt(self) -> Self {
        let v = self.v.sqrt();
        Self {
            v,
            d: if v == 0.0 && self.d == 0.0 {
                0.0
            } else {
                self.d / (2.0 * v)
            },
        }
    }
}
impl From<f64> for Dual {
    fn from(v: f64) -> Self {
        Self::constant(v)
    }
}
impl Add for Dual {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            v: self.v + b.v,
            d: self.d + b.d,
        }
    }
}
impl Sub for Dual {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self {
            v: self.v - b.v,
            d: self.d - b.d,
        }
    }
}
impl Mul for Dual {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            v: self.v * b.v,
            d: self.d * b.v + self.v * b.d,
        }
    }
}
impl Div for Dual {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        Self {
            v: self.v / b.v,
            d: (self.d * b.v - self.v * b.d) / b.v.powi(2),
        }
    }
}
impl Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            v: -self.v,
            d: -self.d,
        }
    }
}
impl Add<f64> for Dual {
    type Output = Self;
    fn add(self, b: f64) -> Self {
        self + Self::from(b)
    }
}
impl Sub<f64> for Dual {
    type Output = Self;
    fn sub(self, b: f64) -> Self {
        self - Self::from(b)
    }
}
impl Mul<f64> for Dual {
    type Output = Self;
    fn mul(self, b: f64) -> Self {
        self * Self::from(b)
    }
}
impl Div<f64> for Dual {
    type Output = Self;
    fn div(self, b: f64) -> Self {
        self / Self::from(b)
    }
}
fn sum(xs: impl Iterator<Item = Dual>) -> Dual {
    xs.fold(0.0.into(), |a, b| a + b)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    Smooth,
    Global,
    LeastSquares,
    Constrained,
    Qp,
    Application,
}
#[derive(Clone, Copy, Debug)]
pub struct Problem {
    pub id: &'static str,
    pub group: Group,
    pub fixed_dimension: Option<usize>,
    pub min_dimension: usize,
    pub bounds: (f64, f64),
    pub analytic: bool,
}
macro_rules! p {
    ($id:literal,$g:ident,$n:expr,$min:expr,$lo:expr,$hi:expr) => {
        Problem {
            id: $id,
            group: Group::$g,
            fixed_dimension: $n,
            min_dimension: $min,
            bounds: ($lo, $hi),
            analytic: true,
        }
    };
}
pub static CATALOG: &[Problem] = &[
    p!("sphere", Smooth, None, 1, -5.0, 5.0),
    p!("ellipsoid", Smooth, None, 2, -5.0, 5.0),
    p!("rosenbrock", Smooth, None, 2, -3.0, 3.0),
    p!("zakharov", Smooth, None, 1, -5.0, 5.0),
    p!("dixon-price", Smooth, None, 2, -3.0, 3.0),
    p!("quartic", Smooth, None, 1, -3.0, 3.0),
    p!("different-powers", Smooth, None, 1, -2.0, 2.0),
    p!("cosh-bowl", Smooth, None, 1, -3.0, 3.0),
    p!("logcosh", Smooth, None, 1, -3.0, 3.0),
    p!("trid", Smooth, None, 2, -10000.0, 10000.0),
    p!("booth", Smooth, Some(2), 2, -10.0, 10.0),
    p!("matyas", Smooth, Some(2), 2, -10.0, 10.0),
    p!("beale", LeastSquares, Some(2), 2, -4.5, 4.5),
    p!("himmelblau", Global, Some(2), 2, -6.0, 6.0),
    p!("three-hump-camel", Smooth, Some(2), 2, -5.0, 5.0),
    p!("powell-singular", LeastSquares, Some(4), 4, -4.0, 5.0),
    p!("rastrigin", Global, None, 1, -5.12, 5.12),
    Problem {
        analytic: false,
        ..p!("ackley", Global, None, 1, -5.0, 5.0)
    },
    p!("griewank", Global, None, 1, -100.0, 100.0),
    p!("levy", Global, None, 2, -10.0, 10.0),
    p!("easom", Global, Some(2), 2, -10.0, 10.0),
    p!("qing", Global, None, 1, -12.0, 12.0),
    p!("brown", Smooth, None, 2, -2.0, 2.0),
    Problem {
        analytic: false,
        ..p!("alpine", Global, None, 1, -10.0, 10.0)
    },
    Problem {
        analytic: false,
        ..p!("schwefel-222", Global, None, 1, -10.0, 10.0)
    },
    p!("linear-fit", LeastSquares, Some(2), 2, -5.0, 5.0),
    p!("exponential-fit", LeastSquares, Some(2), 2, -2.0, 2.0),
    p!("logistic-fit", LeastSquares, Some(2), 2, -3.0, 3.0),
    p!("gaussian-fit", LeastSquares, Some(2), 2, 0.1, 3.0),
    p!("ridge-regression", Application, None, 2, -3.0, 3.0),
    p!("poisson-likelihood", Application, None, 1, -3.0, 3.0),
    p!("spring-chain", Application, None, 2, -3.0, 3.0),
    p!("coupled-qp", Qp, None, 2, -3.0, 3.0),
    p!("simplex-qp", Constrained, None, 2, 0.0, 1.0),
    p!("disk-linear", Constrained, Some(2), 2, -1.0, 1.0),
];

impl Problem {
    pub fn dimension(&self, requested: usize) -> Result<usize, String> {
        let n = self.fixed_dimension.unwrap_or(requested);
        if n < self.min_dimension || n > 128 {
            return Err(format!(
                "{} requires dimension {}..=128",
                self.id, self.min_dimension
            ));
        }
        Ok(n)
    }
    pub fn reference(&self, n: usize) -> (Vec<f64>, f64) {
        let x = match self.id {
            "rosenbrock" | "levy" => vec![1.0; n],
            "dixon-price" => (1..=n)
                .map(|i| 2f64.powf(-(1.0 - 2f64.powi(1 - i as i32))))
                .collect(),
            "trid" => (1..=n).map(|i| (i * (n + 1 - i)) as f64).collect(),
            "booth" => vec![1.0, 3.0],
            "beale" => vec![3.0, 0.5],
            "himmelblau" => vec![3.0, 2.0],
            "easom" => vec![std::f64::consts::PI; 2],
            "qing" => (1..=n).map(|i| (i as f64).sqrt()).collect(),
            "linear-fit" => vec![1.0, 2.0],
            "exponential-fit" => vec![1.0, 0.5],
            "logistic-fit" => vec![0.5, 1.0],
            "gaussian-fit" => vec![1.0, 1.5],
            "simplex-qp" => vec![1.0 / n as f64; n],
            "disk-linear" => vec![1.0 / 2f64.sqrt(); 2],
            _ => vec![0.0; n],
        };
        let value = match self.id {
            "trid" => -(n as f64) * ((n * n + 3 * n - 4) as f64) / 6.0,
            "easom" => -1.0,
            "disk-linear" => -2f64.sqrt(),
            _ => 0.0,
        };
        (x, value)
    }
    pub fn value(&self, x: &[f64]) -> f64 {
        self.formula(&x.iter().map(|&v| v.into()).collect::<Vec<_>>())
            .v
    }
    pub fn gradient(&self, x: &[f64]) -> Vec<f64> {
        (0..x.len())
            .map(|j| {
                let xs = x
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| Dual {
                        v,
                        d: if i == j { 1.0 } else { 0.0 },
                    })
                    .collect::<Vec<_>>();
                self.formula(&xs).d
            })
            .collect()
    }
    pub fn violation(&self, x: &[f64]) -> f64 {
        match self.id {
            "simplex-qp" => (x.iter().sum::<f64>() - 1.0).abs(),
            "disk-linear" => (x.iter().map(|v| v * v).sum::<f64>() - 1.0).max(0.0),
            _ => 0.0,
        }
    }
    fn formula(&self, x: &[Dual]) -> Dual {
        let n = x.len();
        let a = x[0];
        let b = x.get(1).copied().unwrap_or(0.0.into());
        let sq = sum(x.iter().map(|v| v.pow(2)));
        let pi = std::f64::consts::PI;
        match self.id {
            "sphere" => sq,
            "ellipsoid" => sum(x
                .iter()
                .enumerate()
                .map(|(i, v)| v.pow(2) * 10f64.powf(6.0 * i as f64 / (n - 1) as f64))),
            "rosenbrock" => sum(x
                .windows(2)
                .map(|w| (w[1] - w[0].pow(2)).pow(2) * 100.0 + (w[0] - 1.0).pow(2))),
            "zakharov" => {
                let z = sum(x
                    .iter()
                    .enumerate()
                    .map(|(i, v)| *v * (0.5 * (i + 1) as f64)));
                sq + z.pow(2) + z.pow(4)
            }
            "dixon-price" => {
                (a - 1.0).pow(2)
                    + sum((1..n).map(|i| (x[i].pow(2) * 2.0 - x[i - 1]).pow(2) * (i + 1) as f64))
            }
            "quartic" => sum(x.iter().enumerate().map(|(i, v)| v.pow(4) * (i + 1) as f64)),
            "different-powers" => sum(x.iter().enumerate().map(|(i, v)| v.abs().pow(i as i32 + 2))),
            "cosh-bowl" => sum(x.iter().map(|v| v.cosh() - 1.0)),
            "logcosh" => sum(x.iter().map(|v| v.cosh().ln())),
            "trid" => {
                sum(x.iter().map(|v| (*v - 1.0).pow(2))) - sum(x.windows(2).map(|w| w[0] * w[1]))
            }
            "booth" => (a + b * 2.0 - 7.0).pow(2) + (a * 2.0 + b - 5.0).pow(2),
            "matyas" => (a.pow(2) + b.pow(2)) * 0.26 - a * b * 0.48,
            "beale" => sum((1..=3).map(|i| {
                (Dual::from([0.0, 1.5, 2.25, 2.625][i]) - a + a * b.pow(i as i32)).pow(2)
            })),
            "himmelblau" => (a.pow(2) + b - 11.0).pow(2) + (a + b.pow(2) - 7.0).pow(2),
            "three-hump-camel" => {
                a.pow(2) * 2.0 - a.pow(4) * 1.05 + a.pow(6) / 6.0 + a * b + b.pow(2)
            }
            "powell-singular" => {
                (a + b * 10.0).pow(2)
                    + (x[2] - x[3]).pow(2) * 5.0
                    + (b - x[2] * 2.0).pow(4)
                    + (a - x[3]).pow(4) * 10.0
            }
            "rastrigin" => {
                sq + sum(x
                    .iter()
                    .map(|v| Dual::from(10.0) - (*v * (2.0 * pi)).cos() * 10.0))
            }
            "ackley" => {
                -(sq / n as f64).sqrt().mul(0.2).neg().exp() * 20.0
                    - (sum(x.iter().map(|v| (*v * (2.0 * pi)).cos())) / n as f64).exp()
                    + 20.0
                    + std::f64::consts::E
            }
            "griewank" => {
                sq / 4000.0
                    - x.iter().enumerate().fold(Dual::from(1.0), |p, (i, v)| {
                        p * (*v / ((i + 1) as f64).sqrt()).cos()
                    })
                    + 1.0
            }
            "levy" => {
                let w = x.iter().map(|v| (*v - 1.0) / 4.0 + 1.0).collect::<Vec<_>>();
                (w[0] * pi).sin().pow(2)
                    + sum(w[..n - 1]
                        .iter()
                        .map(|v| (*v - 1.0).pow(2) * (((*v * pi + 1.0).sin().pow(2)) * 10.0 + 1.0)))
                    + (w[n - 1] - 1.0).pow(2) * ((w[n - 1] * (2.0 * pi)).sin().pow(2) + 1.0)
            }
            "easom" => -a.cos() * b.cos() * (-((a - pi).pow(2) + (b - pi).pow(2))).exp(),
            "qing" => sum(x
                .iter()
                .enumerate()
                .map(|(i, v)| (v.pow(2) - (i + 1) as f64).pow(2))),
            "brown" => sum(x.windows(2).map(|w| {
                // z^p = exp(p ln z), with the continuous zero extension.
                fn power(z: Dual, p: Dual) -> Dual {
                    if z.v == 0.0 {
                        0.0.into()
                    } else {
                        (p * z.ln()).exp()
                    }
                }
                power(w[0].pow(2), w[1].pow(2) + 1.0) + power(w[1].pow(2), w[0].pow(2) + 1.0)
            })),
            "alpine" => sum(x.iter().map(|v| (*v * v.sin() + *v * 0.1).abs())),
            "schwefel-222" => {
                sum(x.iter().map(|v| v.abs())) + x.iter().fold(Dual::from(1.0), |p, v| p * v.abs())
            }
            "linear-fit" => sum((0..5).map(|i| {
                let t = i as f64 / 4.0;
                (a + b * t - (1.0 + 2.0 * t)).pow(2)
            })),
            "exponential-fit" => sum((0..5).map(|i| {
                let t = i as f64 / 4.0;
                (a * (b * t).exp() - (0.5 * t).exp()).pow(2)
            })),
            "logistic-fit" => sum((0..5).map(|i| {
                let t = i as f64 / 2.0 - 1.0;
                let pred = Dual::from(1.0) / (Dual::from(1.0) + (-(a + b * t)).exp());
                (pred - 1.0 / (1.0 + (-(0.5 + t)).exp())).pow(2)
            })),
            "gaussian-fit" => sum((0..5).map(|i| {
                let t = i as f64 / 2.0 - 1.0;
                (a * (-Dual::from(t * t) / (b.pow(2) * 2.0)).exp()
                    - (-t * t / (2.0 * 1.5f64.powi(2))).exp())
                .pow(2)
            })),
            "ridge-regression" => sum(x.windows(2).map(|w| (w[0] + w[1] * 2.0).pow(2))) + sq * 0.1,
            "poisson-likelihood" => sum(x
                .iter()
                .enumerate()
                .map(|(i, v)| (v.exp() - *v - 1.0) * (i + 1) as f64)),
            "spring-chain" => {
                a.pow(2)
                    + sum(x
                        .windows(2)
                        .enumerate()
                        .map(|(i, w)| (w[1] - w[0]).pow(2) * (i + 1) as f64))
            }
            "coupled-qp" => sq + sum(x.windows(2).map(|w| (w[0] + w[1]).pow(2))),
            "simplex-qp" => sum(x.iter().map(|v| (*v - 1.0 / n as f64).pow(2))),
            "disk-linear" => -a - b,
            _ => unreachable!("catalog dispatch"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn references_and_domains() {
        let mut ids = std::collections::HashSet::new();
        assert!(CATALOG.len() >= 30);
        for p in CATALOG {
            assert!(ids.insert(p.id));
            let n = p.dimension(4).unwrap();
            let (x, v) = p.reference(n);
            assert_eq!(x.len(), n);
            assert!(
                x.iter().all(|&v| v >= p.bounds.0 && v <= p.bounds.1),
                "{}",
                p.id
            );
            assert!(
                (p.value(&x) - v).abs() < 1e-9 * (1.0 + v.abs()),
                "{} {} != {}",
                p.id,
                p.value(&x),
                v
            );
            assert!(p.violation(&x) < 1e-12, "{}", p.id);
        }
    }
    #[test]
    fn independent_directional_differences() {
        for p in CATALOG {
            let n = p.dimension(4).unwrap();
            let x = (0..n).map(|i| 0.31 + 0.17 * i as f64).collect::<Vec<_>>();
            let d = (0..n)
                .map(|i| if i % 2 == 0 { 0.7 } else { -0.4 })
                .collect::<Vec<_>>();
            let h = 1e-6;
            let plus = x.iter().zip(&d).map(|(a, b)| a + h * b).collect::<Vec<_>>();
            let minus = x.iter().zip(&d).map(|(a, b)| a - h * b).collect::<Vec<_>>();
            let fd = (p.value(&plus) - p.value(&minus)) / (2.0 * h);
            let exact = p
                .gradient(&x)
                .iter()
                .zip(&d)
                .map(|(a, b)| a * b)
                .sum::<f64>();
            assert!(
                (exact - fd).abs() < 2e-5 * (1.0 + fd.abs()),
                "{}: {exact} != {fd}",
                p.id
            );
        }
    }
    #[test]
    fn independent_nonoptimal_values() {
        let pi = std::f64::consts::PI;
        let cases = [
            ("sphere", vec![1.0, 2.0], 5.0),
            ("ellipsoid", vec![1.0; 4], 1_010_101.0),
            ("booth", vec![0.0, 0.0], 74.0),
            ("rosenbrock", vec![0.0, 0.0], 1.0),
            ("zakharov", vec![1.0; 4], 654.0),
            ("dixon-price", vec![0.0; 4], 1.0),
            ("quartic", vec![1.0; 4], 10.0),
            ("different-powers", vec![1.0; 4], 4.0),
            ("cosh-bowl", vec![1.0; 4], 4.0 * (1f64.cosh() - 1.0)),
            ("logcosh", vec![1.0; 4], 4.0 * 1f64.cosh().ln()),
            ("matyas", vec![1.0, 2.0], 0.34),
            ("himmelblau", vec![0.0, 0.0], 170.0),
            ("beale", vec![0.0, 0.0], 14.203125),
            ("three-hump-camel", vec![1.0, 0.0], 67.0 / 60.0),
            ("powell-singular", vec![1.0; 4], 122.0),
            ("trid", vec![0.0; 4], 4.0),
            ("rastrigin", vec![1.0; 4], 4.0),
            ("ackley", vec![1.0; 4], 20.0 * (1.0 - (-0.2f64).exp())),
            (
                "griewank",
                vec![1.0; 4],
                1.001
                    - 1f64.cos()
                        * (1.0 / 2f64.sqrt()).cos()
                        * (1.0 / 3f64.sqrt()).cos()
                        * 0.5f64.cos(),
            ),
            (
                "levy",
                vec![0.0; 4],
                0.625 + 3.0 / 16.0 * (1.0 + 10.0 * (0.75 * pi + 1.0).sin().powi(2)),
            ),
            ("easom", vec![0.0; 2], -(-2.0 * pi * pi).exp()),
            ("qing", vec![0.0; 4], 30.0),
            ("brown", vec![1.0; 4], 6.0),
            ("alpine", vec![1.0; 4], 4.0 * (1f64.sin() + 0.1)),
            ("schwefel-222", vec![1.0; 4], 5.0),
            ("linear-fit", vec![0.0, 0.0], 22.5),
            ("exponential-fit", vec![0.0, 0.0], 8.768028532459589),
            ("logistic-fit", vec![0.0, 0.0], 0.18423419028499632),
            ("gaussian-fit", vec![1.0, 1.0], 0.08348751286033611),
            ("ridge-regression", vec![1.0; 4], 27.4),
            (
                "poisson-likelihood",
                vec![1.0; 4],
                10.0 * (std::f64::consts::E - 2.0),
            ),
            ("spring-chain", vec![1.0; 4], 1.0),
            ("coupled-qp", vec![1.0, 2.0], 14.0),
            ("simplex-qp", vec![0.0; 4], 0.25),
            ("disk-linear", vec![0.0; 2], 0.0),
        ];
        assert_eq!(cases.len(), CATALOG.len());
        for (id, x, expected) in cases {
            let p = CATALOG.iter().find(|p| p.id == id).unwrap();
            assert!(
                (p.value(&x) - expected).abs() < 1e-12 * (1.0 + expected.abs()),
                "{id}"
            );
        }
    }
    #[test]
    fn dimensional_constraints_and_large_reference_domains() {
        for p in CATALOG {
            assert!(p.dimension(0).is_ok() == p.fixed_dimension.is_some());
            for requested in [2, 4, 128] {
                if let Ok(n) = p.dimension(requested) {
                    let (x, v) = p.reference(n);
                    assert!(
                        x.iter().all(|&v| v >= p.bounds.0 && v <= p.bounds.1),
                        "{}",
                        p.id
                    );
                    assert!((p.value(&x) - v).abs() < 1e-8 * (1.0 + v.abs()), "{}", p.id);
                }
            }
        }
    }
}
