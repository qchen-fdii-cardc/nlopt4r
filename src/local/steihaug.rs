//! Truncated conjugate gradients for Euclidean trust regions.
//!
//! [`steihaug()`] solves a quadratic model approximately using symmetric
//! Hessian-vector products. [`trust_region_cg`] embeds it in an unbounded local
//! minimizer, adapting radius from actual/predicted reduction. Workspace is O(n).
//! For the exact dense spectral hard case use [`crate::trust_region::subproblem`].
//! Reference: Steihaug (1983), *The conjugate gradient method and trust regions*.
use crate::{
    Error, Solution, StopOptions, TerminationReason, Vector, finite, initial, positive,
    stopping::{Run, snapshot},
    vector,
};

use super::{HessianVector, newton_cg::product};

/// Inner truncated-CG residual and iteration controls.
#[derive(Debug, Clone)]
pub struct SteihaugOptions {
    /// Positive CG cap, default 100; effective cap also cannot exceed dimension.
    pub max_iterations: usize,
    /// Relative residual threshold; the origin residual is the supplied gradient.
    /// Finite in `(0,1)`, default 0.1.
    pub relative_tolerance: f64,
}

impl Default for SteihaugOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            relative_tolerance: 0.1,
        }
    }
}

impl SteihaugOptions {
    /// Check positive cap and a finite residual tolerance strictly between zero and one.
    pub fn validate(&self) -> Result<(), Error> {
        positive(self.relative_tolerance, "Steihaug residual tolerance")?;
        if self.relative_tolerance >= 1.0 || self.max_iterations == 0 {
            return Err(Error::InvalidInput("Steihaug options"));
        }
        Ok(())
    }
}

/// Why the approximate trust-model solve stopped; not an outer convergence claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SteihaugTermination {
    /// Residual norm is at most the configured fraction of the origin norm.
    ResidualTolerance,
    /// Positive-curvature CG reached the Euclidean trust boundary.
    Boundary,
    /// A nonpositive-curvature direction was followed to the better boundary endpoint.
    NegativeCurvature,
    /// Effective CG cap reached; the returned step is still an approximate model step.
    MaxIterations,
}

/// Approximate trust-model step and its inner-work diagnostics.
#[derive(Debug, Clone)]
pub struct SteihaugStep {
    /// Model displacement with norm at most the supplied radius.
    pub step: Vector,
    /// Quadratic model change g'p + p'Hp/2, not an objective evaluation.
    pub value: f64,
    /// Consumed inner CG iterations.
    pub iterations: usize,
    /// Calls to the supplied multiply callback; does not include objective work.
    pub hessian_products: usize,
    /// Inner termination condition.
    pub termination: SteihaugTermination,
}

fn boundary_roots(p: &Vector, d: &Vector, radius: f64) -> Result<(f64, f64), Error> {
    let dn = finite(d.norm(), "Steihaug direction norm")?;
    positive(dn, "Steihaug boundary direction")?;
    let pn = p.norm();
    let b = p.dot(&(d / dn));
    let c = (pn - radius) * (pn + radius);
    let root = finite((b * b - c).max(0.0).sqrt(), "Steihaug boundary root")?;
    // Use the well-conditioned root directly and recover the other from the
    // product of the roots. This avoids subtracting nearly equal quantities.
    let (negative, positive_root) = if b >= 0.0 {
        let negative = -b - root;
        let positive_root = if negative == 0.0 { 0.0 } else { c / negative };
        (negative, positive_root)
    } else {
        let positive_root = -b + root;
        let negative = if positive_root == 0.0 {
            0.0
        } else {
            c / positive_root
        };
        (negative, positive_root)
    };
    Ok((
        finite(negative / dn, "Steihaug boundary step")?,
        finite(positive_root / dn, "Steihaug boundary step")?,
    ))
}

fn report(
    g: &Vector,
    step: Vector,
    hp: Vector,
    iterations: usize,
    termination: SteihaugTermination,
) -> Result<SteihaugStep, Error> {
    vector(&step, g.len(), "Steihaug step")?;
    let value = finite(g.dot(&step) + 0.5 * step.dot(&hp), "Steihaug model")?;
    Ok(SteihaugStep {
        step,
        value,
        iterations,
        hessian_products: iterations,
        termination,
    })
}

/// Approximately minimize g'p + p'Hp/2 over ||p|| <= radius.
///
/// `multiply` applies a symmetric Hessian without building a matrix. Stops on
/// the boundary, nonpositive curvature, a small residual, or the iteration cap.
/// This is not the globally exact spectral subproblem solver: in particular a
/// zero gradient returns zero and does not search for hidden negative curvature.
pub fn steihaug<F>(
    g: &Vector,
    radius: f64,
    options: &SteihaugOptions,
    mut multiply: F,
) -> Result<SteihaugStep, Error>
where
    F: FnMut(&Vector) -> Result<Vector, Error>,
{
    options.validate()?;
    positive(radius, "Steihaug radius")?;
    if g.is_empty() {
        return Err(Error::InvalidInput("empty Steihaug gradient"));
    }
    vector(g, g.len(), "Steihaug gradient")?;
    let norm = finite(g.norm(), "Steihaug gradient norm")?;
    let mut p = Vector::zeros(g.len());
    let mut hp = Vector::zeros(g.len());
    if norm == 0.0 {
        return report(g, p, hp, 0, SteihaugTermination::ResidualTolerance);
    }
    let mut r = g.clone();
    let mut d = -&r;
    let mut rr = finite(r.norm_squared(), "Steihaug squared residual")?;
    let limit = options.max_iterations.min(g.len());
    for k in 1..=limit {
        let hd = multiply(&d)?;
        vector(&hd, g.len(), "Steihaug Hessian-vector product")?;
        let curvature = finite(d.dot(&hd), "Steihaug curvature")?;
        if curvature <= 0.0 {
            let (negative, positive_root) = boundary_roots(&p, &d, radius)?;
            let change = |tau: f64| tau * (g.dot(&d) + hp.dot(&d)) + 0.5 * tau * tau * curvature;
            let tau = if change(negative) < change(positive_root) {
                negative
            } else {
                positive_root
            };
            return report(
                g,
                p + tau * &d,
                hp + tau * hd,
                k,
                SteihaugTermination::NegativeCurvature,
            );
        }
        let alpha = finite(rr / curvature, "Steihaug CG step")?;
        let next = &p + alpha * &d;
        let next_norm = finite(next.norm(), "Steihaug trial norm")?;
        if next_norm >= radius {
            let (_, tau) = boundary_roots(&p, &d, radius)?;
            return report(
                g,
                p + tau * &d,
                hp + tau * hd,
                k,
                SteihaugTermination::Boundary,
            );
        }
        p = next;
        hp += alpha * &hd;
        r += alpha * hd;
        let residual = finite(r.norm(), "Steihaug residual")?;
        if residual <= options.relative_tolerance * norm {
            return report(g, p, hp, k, SteihaugTermination::ResidualTolerance);
        }
        let next_rr = finite(residual * residual, "Steihaug squared residual")?;
        d = -&r + (next_rr / rr) * d;
        rr = next_rr;
    }
    report(g, p, hp, limit, SteihaugTermination::MaxIterations)
}

/// Matrix-free trust-region minimizer settings, independent of line-search options.
#[derive(Debug, Clone)]
pub struct TrustRegionCgOptions {
    /// Outer gradient, iteration, objective-budget and stagnation controls.
    pub stopping: StopOptions,
    /// Approximate quadratic model solve.
    pub subproblem: SteihaugOptions,
    /// Positive finite starting radius, default 1.
    pub initial_radius: f64,
    /// Positive finite radius ceiling, at least the initial radius; default 100.
    pub maximum_radius: f64,
    /// Accept when actual/predicted decrease exceeds this value in `[0,0.25)`;
    /// default 0.1.
    pub acceptance_threshold: f64,
}

impl Default for TrustRegionCgOptions {
    fn default() -> Self {
        Self {
            stopping: StopOptions::default(),
            subproblem: SteihaugOptions::default(),
            initial_radius: 1.0,
            maximum_radius: 100.0,
            acceptance_threshold: 0.1,
        }
    }
}

impl TrustRegionCgOptions {
    /// Validate stopping, subproblem, ordered radii and acceptance threshold.
    pub fn validate(&self) -> Result<(), Error> {
        self.stopping.validate()?;
        self.subproblem.validate()?;
        positive(self.initial_radius, "initial trust-CG radius")?;
        positive(self.maximum_radius, "maximum trust-CG radius")?;
        if self.initial_radius > self.maximum_radius
            || !(0.0..0.25).contains(&self.acceptance_threshold)
        {
            return Err(Error::InvalidInput("trust-CG options"));
        }
        Ok(())
    }
}

/// Minimize using matrix-free Steihaug trust steps and actual/model reduction.
///
/// Requires a finite nonempty `x0`, analytic gradient and symmetric
/// [`HessianVector`]. Rejected trials consume objective budget, not accepted-step
/// stagnation progress. Hessian products are counted separately, and the last
/// fully evaluated accepted state is retained on failure/budget interruption.
/// Gradient convergence is only first-order stationarity; hidden negative
/// eigenspaces at zero gradient are not explored.
pub fn trust_region_cg<P: HessianVector + ?Sized>(
    problem: &P,
    x0: &Vector,
    options: &TrustRegionCgOptions,
) -> Result<Solution, Error> {
    options.validate()?;
    initial(x0, &options.stopping)?;
    let mut run = Run::new(&options.stopping)?;
    let result = minimize_impl(problem, x0, options, &mut run);
    run.finish(result, "Steihaug trust-region minimization")
}

fn minimize_impl<P: HessianVector + ?Sized>(
    problem: &P,
    x0: &Vector,
    options: &TrustRegionCgOptions,
    run: &mut Run,
) -> Result<Solution, Error> {
    let mut x = x0.clone();
    let (mut f, mut g) = run.evaluate(problem, &x)?;
    let mut radius = options.initial_radius;
    let mut accepted = false;
    for k in 0..=options.stopping.max_iterations {
        let norm = finite(g.norm(), "trust-CG gradient norm")?;
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
        run.counts.subproblem += 1;
        let step = steihaug(&g, radius, &options.subproblem, |v| {
            product(problem, &x, v, run)
        })?;
        if step.value >= 0.0 {
            return Err(Error::InvalidInput(
                "trust-CG model has no predicted decrease",
            ));
        }
        let trial = &x + &step.step;
        let (next_f, next_g) = run.evaluate(problem, &trial)?;
        let ratio = finite((f - next_f) / -step.value, "trust-CG reduction ratio")?;
        if ratio < 0.25 {
            radius *= 0.25;
        } else if ratio > 0.75
            && matches!(
                step.termination,
                SteihaugTermination::Boundary | SteihaugTermination::NegativeCurvature
            )
        {
            radius = (2.0 * radius).min(options.maximum_radius);
        }
        accepted = ratio > options.acceptance_threshold;
        if accepted {
            x = trial;
            f = next_f;
            g = next_g;
        }
        positive(radius, "trust-CG radius underflow")?;
    }
    unreachable!()
}
