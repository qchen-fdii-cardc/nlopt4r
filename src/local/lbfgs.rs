//! Limited-memory BFGS with a scaled two-loop inverse-Hessian product.
use std::collections::VecDeque;

use crate::{
    Error, Evaluator, Solution, StopOptions, TerminationReason, Vector, finite, initial,
    stopping::{Run, snapshot},
};

use super::{StrongWolfeOptions, strong_wolfe};

#[derive(Debug, Clone)]
pub struct LbfgsOptions {
    pub stopping: StopOptions,
    pub line_search: StrongWolfeOptions,
    /// Maximum number of correction pairs; storage is O(history_size * dimension).
    pub history_size: usize,
    /// Skip a pair unless s'y > tolerance * ||s|| * ||y||.
    pub curvature_tolerance: f64,
}

impl Default for LbfgsOptions {
    fn default() -> Self {
        Self {
            stopping: StopOptions::default(),
            line_search: StrongWolfeOptions::default(),
            history_size: 10,
            curvature_tolerance: 1e-10,
        }
    }
}

impl LbfgsOptions {
    pub fn validate(&self) -> Result<(), Error> {
        self.stopping.validate()?;
        self.line_search.validate()?;
        if self.history_size == 0
            || !self.curvature_tolerance.is_finite()
            || !(0.0..1.0).contains(&self.curvature_tolerance)
        {
            return Err(Error::InvalidInput("L-BFGS history or curvature tolerance"));
        }
        Ok(())
    }
}

struct Pair {
    s: Vector,
    y: Vector,
    rho: f64,
}

#[derive(Default)]
struct History {
    pairs: VecDeque<Pair>,
}

impl History {
    fn push(&mut self, s: Vector, y: Vector, options: &LbfgsOptions) -> bool {
        let sy = s.dot(&y);
        let yy = y.norm_squared();
        let scale = s.norm() * y.norm();
        if !sy.is_finite()
            || !yy.is_finite()
            || !scale.is_finite()
            || sy <= options.curvature_tolerance * scale
            || yy <= 0.0
            || !(1.0 / sy).is_finite()
            || !(sy / yy).is_finite()
            || sy / yy <= 0.0
        {
            return false;
        }
        if self.pairs.len() == options.history_size {
            self.pairs.pop_front();
        }
        self.pairs.push_back(Pair {
            s,
            y,
            rho: 1.0 / sy,
        });
        true
    }

    fn direction(&self, g: &Vector) -> Vector {
        let mut q = g.clone();
        let mut alpha = vec![0.0; self.pairs.len()];
        for (i, pair) in self.pairs.iter().enumerate().rev() {
            alpha[i] = pair.rho * pair.s.dot(&q);
            q -= alpha[i] * &pair.y;
        }
        if let Some(pair) = self.pairs.back() {
            q *= pair.s.dot(&pair.y) / pair.y.norm_squared();
        }
        for (i, pair) in self.pairs.iter().enumerate() {
            let beta = pair.rho * pair.y.dot(&q);
            q += (alpha[i] - beta) * &pair.s;
        }
        -q
    }
}

pub fn lbfgs<P: Evaluator + ?Sized>(
    problem: &P,
    x0: &Vector,
    options: &LbfgsOptions,
) -> Result<Solution, Error> {
    options.validate()?;
    initial(x0, &options.stopping)?;
    let mut run = Run::new(&options.stopping)?;
    let result = minimize_impl(problem, x0, options, &mut run);
    run.finish(result, "L-BFGS")
}

fn minimize_impl<P: Evaluator + ?Sized>(
    problem: &P,
    x0: &Vector,
    options: &LbfgsOptions,
    run: &mut Run,
) -> Result<Solution, Error> {
    let mut x = x0.clone();
    let (mut f, mut g) = run.evaluate(problem, &x)?;
    let mut history = History::default();
    for k in 0..=options.stopping.max_iterations {
        let norm = finite(g.norm(), "L-BFGS gradient norm")?;
        let reason = run.reason(
            &x,
            f,
            k,
            norm <= options.stopping.gradient_tolerance,
            TerminationReason::GradientTolerance,
            k > 0,
        );
        let report = run.record(snapshot(x.clone(), f, norm, k, reason));
        if reason.is_some() {
            return Ok(report);
        }
        let mut d = history.direction(&g);
        if !d.iter().all(|v| v.is_finite())
            || !g.dot(&d).is_finite()
            || g.dot(&d) >= -1e-12 * norm * d.norm()
        {
            history.pairs.clear();
            d = -&g;
        }
        let step = strong_wolfe(f, &g, &d, &options.line_search, |alpha| {
            run.evaluate(problem, &(&x + alpha * &d))
        })?;
        let next = &x + step.alpha * d;
        history.push(&next - &x, &step.gradient - &g, options);
        x = next;
        f = step.value;
        g = step.gradient;
    }
    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_failed_curvature_without_evicting_history() {
        let mut h = History::default();
        let options = LbfgsOptions {
            history_size: 1,
            ..Default::default()
        };
        let v = Vector::from_vec(vec![1.0, 0.0]);
        assert!(h.push(v.clone(), 2.0 * &v, &options));
        assert!(!h.push(v.clone(), -&v, &options));
        assert!(!h.push(v.clone(), Vector::zeros(2), &options));
        assert!(!h.push(v.clone(), Vector::from_vec(vec![f64::NAN, 0.0]), &options));
        assert_eq!(h.pairs.len(), 1);
        assert!((h.direction(&v) + 0.5 * &v).norm() < 1e-14);
        assert!(h.push(v.clone(), 4.0 * &v, &options));
        assert_eq!(h.pairs.len(), 1);
        assert!((h.direction(&v) + 0.25 * v).norm() < 1e-14);
    }

    #[test]
    fn two_loop_matches_bfgs_updates_after_history_eviction() {
        use crate::Matrix;

        let options = LbfgsOptions {
            history_size: 2,
            ..Default::default()
        };
        let b = Matrix::from_diagonal(&Vector::from_vec(vec![2.0, 3.0, 4.0]));
        let mut history = History::default();
        for s in [
            Vector::from_vec(vec![1.0, 1.0, 0.0]),
            Vector::from_vec(vec![0.0, 1.0, 1.0]),
            Vector::from_vec(vec![-1.0, 0.0, 1.0]),
        ] {
            let y = &b * &s;
            assert!(history.push(s, y, &options));
        }
        assert_eq!(history.pairs.len(), 2);
        let last = history.pairs.back().unwrap();
        let mut inverse = (last.s.dot(&last.y) / last.y.norm_squared()) * Matrix::identity(3, 3);
        for pair in &history.pairs {
            let factor = Matrix::identity(3, 3) - pair.rho * &pair.s * pair.y.transpose();
            inverse =
                &factor * inverse * factor.transpose() + pair.rho * &pair.s * pair.s.transpose();
        }
        let g = Vector::from_vec(vec![3.0, -1.0, 2.0]);
        assert!((history.direction(&g) + inverse * g).norm() < 1e-13);
    }
}
