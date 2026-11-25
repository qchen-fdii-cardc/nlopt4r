//! Newton trust regions with a spectral solution of the quadratic subproblem.
//!
//! Matrix-free alternatives are provided separately; the dense spectral solver
//! and its hard-case behavior below are unchanged.
pub use crate::local::{
    SteihaugOptions, SteihaugStep, SteihaugTermination, TrustRegionCgOptions, steihaug,
    trust_region_cg,
};
use crate::{
    Error, Evaluator, Matrix, Solution, StopOptions, TerminationReason, Vector, finite, initial,
    positive,
    stopping::{Run, snapshot},
    symmetric, vector,
};

#[derive(Debug, Clone)]
pub struct TrustRegionOptions {
    pub stopping: StopOptions,
    pub initial_radius: f64,
    pub maximum_radius: f64,
    pub acceptance_threshold: f64,
}

impl Default for TrustRegionOptions {
    fn default() -> Self {
        Self {
            stopping: StopOptions::default(),
            initial_radius: 1.0,
            maximum_radius: 2.0,
            acceptance_threshold: 0.15,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrustRegionStep {
    pub step: Vector,
    pub value: f64,
    pub multiplier: f64,
}

/// Globally solve min g'd + d'Bd/2 subject to ||d|| <= radius.
/// Handles indefinite B and the zero-gradient "hard case".
pub fn subproblem(g: &Vector, b: &Matrix, radius: f64) -> Result<TrustRegionStep, Error> {
    positive(radius, "trust radius")?;
    if g.is_empty() {
        return Err(Error::InvalidInput("empty gradient"));
    }
    vector(g, g.len(), "gradient")?;
    symmetric(b, g.len())?;
    let eig = b.clone().symmetric_eigen();
    let min_index = eig.eigenvalues.imin();
    let min_eigen = eig.eigenvalues[min_index];
    let c = eig.eigenvectors.transpose() * g;
    let scale = 1.0 + eig.eigenvalues.amax();
    let eps = 1e-12 * scale;
    let lower = (-min_eigen).max(0.0);
    let coordinates = |lambda: f64| -> Option<Vector> {
        let mut z = Vector::zeros(g.len());
        for i in 0..g.len() {
            let denom = eig.eigenvalues[i] + lambda;
            if denom.abs() <= eps {
                if c[i].abs() > 1e-12 * (1.0 + g.norm()) {
                    return None;
                }
            } else {
                z[i] = -c[i] / denom;
            }
        }
        Some(z)
    };
    let (z, multiplier) = if let Some(mut z) = coordinates(lower)
        && z.norm() <= radius
    {
        if lower > 0.0 {
            z[min_index] += ((radius - z.norm()) * (radius + z.norm())).max(0.0).sqrt();
        }
        (z, lower)
    } else {
        let mut lo = lower;
        let mut hi = lower + (g.norm() / radius).max(1.0);
        while coordinates(hi).is_none_or(|z| z.norm() > radius) {
            hi = lower + 2.0 * (hi - lower);
            finite(hi, "trust-region multiplier")?;
        }
        for _ in 0..120 {
            let mid = lo + 0.5 * (hi - lo);
            if coordinates(mid).is_none_or(|z| z.norm() > radius) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (coordinates(hi).ok_or(Error::SingularSystem)?, hi)
    };
    let step = &eig.eigenvectors * z;
    let value = finite(g.dot(&step) + 0.5 * step.dot(&(b * &step)), "trust model")?;
    Ok(TrustRegionStep {
        step,
        value,
        multiplier,
    })
}

pub fn minimize<P: Evaluator + ?Sized>(
    problem: &P,
    x0: &Vector,
    options: &TrustRegionOptions,
) -> Result<Solution, Error> {
    initial(x0, &options.stopping)?;
    positive(options.initial_radius, "initial trust radius")?;
    positive(options.maximum_radius, "maximum trust radius")?;
    if options.initial_radius > options.maximum_radius
        || !(0.0..0.75).contains(&options.acceptance_threshold)
    {
        return Err(Error::InvalidInput("trust region options"));
    }
    let mut run = Run::new(&options.stopping)?;
    let result = minimize_impl(problem, x0, options, &mut run);
    run.finish(result, "trust-region minimization")
}

fn minimize_impl<P: Evaluator + ?Sized>(
    problem: &P,
    x0: &Vector,
    options: &TrustRegionOptions,
    run: &mut Run,
) -> Result<Solution, Error> {
    let mut x = x0.clone();
    let mut radius = options.initial_radius;
    let mut accepted = false;
    for k in 0..=options.stopping.max_iterations {
        let (f, g) = run.evaluate(problem, &x)?;
        let norm = finite(g.norm(), "gradient norm")?;
        let reason = run.reason(
            &x,
            f,
            k,
            norm <= options.stopping.gradient_tolerance,
            TerminationReason::GradientTolerance,
            accepted,
        );
        let report = run.record(snapshot(x.clone(), f, norm, k, reason));
        if reason.is_some() {
            return Ok(report);
        }
        let step = subproblem(&g, &run.hessian(problem, &x)?, radius)?;
        run.counts.subproblem += 1;
        if step.value >= 0.0 {
            return Err(Error::InvalidInput("trust model has no predicted decrease"));
        }
        let trial = &x + &step.step;
        vector(&trial, x.len(), "trust iterate")?;
        let next_f = run.value(problem, &trial)?;
        let ratio = (f - next_f) / -step.value;
        if ratio <= options.acceptance_threshold {
            radius *= 0.5;
        } else if ratio >= 0.75 && step.step.norm() >= 0.99 * radius {
            radius = (2.0 * radius).min(options.maximum_radius);
        }
        accepted = ratio > options.acceptance_threshold;
        if accepted {
            x = trial;
        }
        positive(radius, "trust radius underflow")?;
    }
    unreachable!()
}
