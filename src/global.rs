//! Seeded, bounded, derivative-free population optimization.
//!
//! Choose [`differential_evolution`] for synchronous DE/rand/1/bin or
//! [`particle_swarm`] for inertia-weight global-best PSO. Both require finite
//! [`Bounds`], accept [`crate::value_only`] callbacks, and retain the best checked
//! point if the budget ends during initialization or a partial generation.
//! They use O(population * dimension) storage and never request derivatives.
//!
//! ```
//! use nlopt4r::{Bounds, value_only, core::BudgetOptions,
//!     global::{differential_evolution, DeOptions}};
//! let problem = value_only(|x: &[f64]| Ok(x.iter().map(|v| v * v).sum()));
//! let bounds = Bounds::uniform(2, -5.0, 5.0)?;
//! let options = DeOptions {
//!     seed: 42,
//!     budget: BudgetOptions { max_evaluations: 2000, target_value: Some(1e-6) },
//!     ..Default::default()
//! };
//! let result = differential_evolution(&problem, &bounds, &options)?;
//! assert!(bounds.contains(result.x.as_slice()));
//! assert!(result.evaluations.objective <= 2000);
//! // A missed target is a normal report, not a certified optimum.
//! println!("best={}, stop={:?}", result.value, result.termination_reason);
//! # Ok::<(), nlopt4r::Error>(())
//! ```
//!
//! A seed resets the ChaCha8 stream per run, but is not a promise of bitwise
//! cross-platform floating-point reproducibility. No nonlinear constraints,
//! parallel evaluation, adaptive strategies or automatic local polishing are
//! supplied. Restarts and parameter sensitivity matter on difficult landscapes.
//! Primary references: Storn and Price (1997), *Differential Evolution*;
//! Kennedy and Eberhart (1995), *Particle swarm optimization*.
use crate::core::{BudgetOptions, EvaluationContext, Observer, SearchTermination};
use crate::utils::{ChaCha8Rng, Rng, SeedableRng};
use crate::{Bounds, Error, EvaluationCounts, Evaluator, Vector};

/// Best-ever checked point, search outcome and population diagnostics.
#[derive(Debug, Clone)]
pub struct GlobalSolution {
    /// Lowest-value evaluated position, including partial-generation improvements.
    pub x: Vector,
    /// Finite objective at `x`.
    pub value: f64,
    /// Completed generations; initialization and an unfinished generation do not count.
    pub generations: usize,
    /// Attempted callback work, including initialization.
    pub evaluations: EvaluationCounts,
    /// Target, cancellation or limit stop; none certifies global optimality.
    pub termination_reason: SearchTermination,
    /// Run's RNG seed.
    pub seed: u64,
    /// Maximum coordinate spread divided by its domain width.
    pub diversity: f64,
}
impl GlobalSolution {
    /// Whether the requested objective target was reached; not a convergence theorem.
    pub fn target_reached(&self) -> bool {
        self.termination_reason == SearchTermination::TargetReached
    }
}
/// Synchronous DE/rand/1/bin configuration.
///
/// Mutation uses `a + F*(b-c)` with distinct donors excluding the target.
/// Binomial crossover forces one donor coordinate; selection accepts ties.
/// All donors read the preceding generation.
#[derive(Debug, Clone)]
pub struct DeOptions {
    /// Shared strict budget and optional target.
    pub budget: BudgetOptions,
    /// ChaCha8 seed, default 0.
    pub seed: u64,
    /// Number of candidates; at least 4, default 40.
    pub population_size: usize,
    /// Completed-generation allowance, default 1000; zero still evaluates initialization.
    pub max_generations: usize,
    /// Differential weight F, finite in `(0,2]`, default 0.8.
    pub weight: f64,
    /// Crossover probability CR in `[0,1]`, default 0.9; one coordinate is forced even at zero.
    pub crossover: f64,
    /// Optional exact-size population of finite in-bounds vectors.
    /// `None` uses uniform sampling; invalid populations fail before objective calls.
    pub initial_population: Option<Vec<Vector>>,
}
impl Default for DeOptions {
    fn default() -> Self {
        Self {
            budget: BudgetOptions::default(),
            seed: 0,
            population_size: 40,
            max_generations: 1000,
            weight: 0.8,
            crossover: 0.9,
            initial_population: None,
        }
    }
}
/// Synchronous inertia-weight global-best PSO configuration.
///
/// Velocities start at zero. Each generation snapshots global best; independent
/// coordinate-wise draws multiply cognitive/social terms. Personal bests update
/// only on strict improvement. Projected coordinates have their velocity reset.
#[derive(Debug, Clone)]
pub struct PsoOptions {
    /// Shared strict budget and optional target.
    pub budget: BudgetOptions,
    /// ChaCha8 seed, default 0.
    pub seed: u64,
    /// Number of particles; at least 2, default 40.
    pub population_size: usize,
    /// Completed-generation allowance, default 1000; initialization is separate.
    pub max_generations: usize,
    /// Inertia coefficient, finite in `[0,1)`, default 0.7.
    pub inertia: f64,
    /// Nonnegative finite personal-best attraction, default 1.5.
    pub cognitive: f64,
    /// Nonnegative finite global-best attraction, default 1.5.
    pub social: f64,
    /// Velocity cap as a fraction of each bound width, finite in `(0,1]`, default 0.2.
    pub velocity_limit: f64,
    /// Optional exact-size, finite in-bounds starting positions; `None` samples uniformly.
    pub initial_population: Option<Vec<Vector>>,
}
impl Default for PsoOptions {
    fn default() -> Self {
        Self {
            budget: BudgetOptions::default(),
            seed: 0,
            population_size: 40,
            max_generations: 1000,
            inertia: 0.7,
            cognitive: 1.5,
            social: 1.5,
            velocity_limit: 0.2,
            initial_population: None,
        }
    }
}
fn population(
    bounds: &Bounds,
    size: usize,
    initial: &Option<Vec<Vector>>,
    rng: &mut ChaCha8Rng,
) -> Result<Vec<f64>, Error> {
    let capacity = size
        .checked_mul(bounds.dimension())
        .ok_or(Error::InvalidInput("population size overflow"))?;
    if capacity > isize::MAX as usize / std::mem::size_of::<f64>() {
        return Err(Error::InvalidInput("population byte-size overflow"));
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(capacity)
        .map_err(|_| Error::InvalidInput("population allocation failed"))?;
    if let Some(points) = initial {
        if points.len() != size {
            return Err(Error::InvalidInput("initial population size"));
        }
        for x in points {
            bounds.validate_point(x.as_slice())?;
            result.extend_from_slice(x.as_slice());
        }
    } else {
        result.resize(capacity, 0.0);
        for x in result.chunks_exact_mut(bounds.dimension()) {
            bounds.sample(rng, x)?;
        }
    }
    Ok(result)
}
fn diversity(points: &[f64], bounds: &Bounds) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    (0..bounds.dimension())
        .filter(|i| bounds.width(*i) > 0.0)
        .map(|i| {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for x in points.chunks_exact(bounds.dimension()) {
                lo = lo.min(x[i]);
                hi = hi.max(x[i]);
            }
            (hi - lo) / bounds.width(i)
        })
        .fold(0.0, f64::max)
}
fn report<P: Evaluator + ?Sized>(
    context: &EvaluationContext<'_, P>,
    points: &[f64],
    bounds: &Bounds,
    generations: usize,
    seed: u64,
) -> GlobalSolution {
    let best = context
        .best()
        .expect("positive budget evaluates an initial point");
    GlobalSolution {
        x: best.x.clone(),
        value: best.value,
        generations,
        seed,
        evaluations: context.counts(),
        termination_reason: context
            .stop_reason()
            .unwrap_or(SearchTermination::MaxIterations),
        diversity: diversity(points, bounds),
    }
}
fn initialize<P: Evaluator + ?Sized>(
    context: &mut EvaluationContext<'_, P>,
    points: &[f64],
    n: usize,
) -> Result<Vec<f64>, Error> {
    let mut values = Vec::with_capacity(points.len() / n);
    for point in points.chunks_exact(n) {
        if context.stop_reason().is_some() {
            break;
        }
        values.push(context.evaluate(point).map_err(|e| context.failure(e))?);
    }
    Ok(values)
}
/// DE/rand/1/bin with synchronous generations and midpoint boundary repair.
///
/// Bounds and population dimensions must match; objective callbacks must be finite.
/// Mutation overflow and callback errors return [`Error`] with a best-point
/// checkpoint when available. Budget/target/generation stops return a normal
/// [`GlobalSolution`]. Midpoint repair moves a violated donor toward the target
/// and the relevant bound; it is not arbitrary objective-domain rejection.
pub fn differential_evolution<P: Evaluator + ?Sized>(
    p: &P,
    bounds: &Bounds,
    options: &DeOptions,
) -> Result<GlobalSolution, Error> {
    differential_evolution_with_observer(p, bounds, options, None)
}

/// DE with an optional observer after every valid evaluation, including initialization.
///
/// [`crate::core::Control::Stop`] cancels without another call; observer errors
/// propagate with the best valid point. `None` is identical to [`differential_evolution`].
///
/// ```
/// use nlopt4r::{Bounds, value_only, core::{Control, EvaluationEvent, SearchTermination},
///     global::{DeOptions, differential_evolution_with_observer}};
/// let problem = value_only(|x: &[f64]| Ok(x[0]*x[0]));
/// let mut observer = |event: &EvaluationEvent<'_>| {
///     Ok(if event.evaluations.objective >= 5 { Control::Stop } else { Control::Continue })
/// };
/// let result = differential_evolution_with_observer(
///     &problem, &Bounds::uniform(1, -1.0, 1.0)?, &DeOptions::default(), Some(&mut observer),
/// )?;
/// assert_eq!(result.termination_reason, SearchTermination::Cancelled);
/// assert_eq!(result.evaluations.objective, 5);
/// # Ok::<(), nlopt4r::Error>(())
/// ```
pub fn differential_evolution_with_observer<'a, P: Evaluator + ?Sized>(
    p: &'a P,
    bounds: &Bounds,
    options: &DeOptions,
    observer: Option<&'a mut Observer<'a>>,
) -> Result<GlobalSolution, Error> {
    options.budget.validate()?;
    if options.population_size < 4
        || !options.weight.is_finite()
        || !(0.0..=2.0).contains(&options.weight)
        || options.weight == 0.0
        || !options.crossover.is_finite()
        || !(0.0..=1.0).contains(&options.crossover)
    {
        return Err(Error::InvalidInput(
            "DE requires population >=4, 0<F<=2 and 0<=CR<=1",
        ));
    }
    let n = bounds.dimension();
    let mut rng = ChaCha8Rng::seed_from_u64(options.seed);
    let mut points = population(
        bounds,
        options.population_size,
        &options.initial_population,
        &mut rng,
    )?;
    let mut context = EvaluationContext::new(p, n, &options.budget)?;
    if let Some(observer) = observer {
        context = context.with_observer(observer);
    }
    let mut values = initialize(&mut context, &points, n)?;
    if context.stop_reason().is_some() {
        return Ok(report(
            &context,
            &points[..values.len() * n],
            bounds,
            0,
            options.seed,
        ));
    }
    let mut candidate = vec![0.0; n];
    let mut next = points.clone();
    for generation in 0..options.max_generations {
        for i in 0..options.population_size {
            if context.stop_reason().is_some() {
                return Ok(report(&context, &next, bounds, generation, options.seed));
            }
            let donors = donors(&mut rng, options.population_size, i);
            let forced = rng.random_range(0..n);
            for j in 0..n {
                let target = points[i * n + j];
                let mutant = points[donors[0] * n + j]
                    + options.weight * (points[donors[1] * n + j] - points[donors[2] * n + j]);
                if !mutant.is_finite() {
                    return Err(context.failure(Error::NonFinite("DE mutation")));
                }
                let repaired = if mutant < bounds.lower()[j] {
                    bounds.lower()[j] + 0.5 * (target - bounds.lower()[j])
                } else if mutant > bounds.upper()[j] {
                    bounds.upper()[j] + 0.5 * (target - bounds.upper()[j])
                } else {
                    mutant
                };
                candidate[j] = if j == forced || rng.random::<f64>() < options.crossover {
                    repaired
                } else {
                    target
                };
            }
            bounds
                .validate_point(&candidate)
                .map_err(|e| context.failure(e))?;
            let value = context
                .evaluate(&candidate)
                .map_err(|e| context.failure(e))?;
            if value <= values[i] {
                next[i * n..(i + 1) * n].copy_from_slice(&candidate);
                values[i] = value;
            }
        }
        std::mem::swap(&mut points, &mut next);
        next.copy_from_slice(&points);
        if context.stop_reason().is_some() {
            return Ok(report(
                &context,
                &points,
                bounds,
                generation + 1,
                options.seed,
            ));
        }
    }
    Ok(report(
        &context,
        &points,
        bounds,
        options.max_generations,
        options.seed,
    ))
}
fn donors(rng: &mut ChaCha8Rng, size: usize, target: usize) -> [usize; 3] {
    let mut result = [target; 3];
    for i in 0..3 {
        loop {
            let candidate = rng.random_range(0..size);
            if candidate != target && !result[..i].contains(&candidate) {
                result[i] = candidate;
                break;
            }
        }
    }
    result
}
/// Inertia-weight, synchronous global-best PSO. Search success is not a
/// certificate of global optimality.
///
/// Uses `v = inertia*v + cognitive*r1*(pbest-x) + social*r2*(gbest-x)`,
/// then clips velocities by bound width, projects positions and resets velocity
/// on boundary-projected coordinates. Returns the best evaluated position, not
/// the final swarm centroid. Invalid options, non-finite arithmetic and callbacks
/// are explicit errors.
pub fn particle_swarm<P: Evaluator + ?Sized>(
    p: &P,
    bounds: &Bounds,
    options: &PsoOptions,
) -> Result<GlobalSolution, Error> {
    particle_swarm_with_observer(p, bounds, options, None)
}

/// PSO with a fallible per-evaluation observer; see [`differential_evolution_with_observer`].
pub fn particle_swarm_with_observer<'a, P: Evaluator + ?Sized>(
    p: &'a P,
    bounds: &Bounds,
    options: &PsoOptions,
    observer: Option<&'a mut Observer<'a>>,
) -> Result<GlobalSolution, Error> {
    options.budget.validate()?;
    if options.population_size < 2
        || !options.inertia.is_finite()
        || !(0.0..1.0).contains(&options.inertia)
        || [options.cognitive, options.social]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
        || !options.velocity_limit.is_finite()
        || !(0.0..=1.0).contains(&options.velocity_limit)
        || options.velocity_limit == 0.0
    {
        return Err(Error::InvalidInput(
            "invalid PSO population, inertia, coefficients or velocity limit",
        ));
    }
    let n = bounds.dimension();
    let mut rng = ChaCha8Rng::seed_from_u64(options.seed);
    let mut points = population(
        bounds,
        options.population_size,
        &options.initial_population,
        &mut rng,
    )?;
    let mut context = EvaluationContext::new(p, n, &options.budget)?;
    if let Some(observer) = observer {
        context = context.with_observer(observer);
    }
    let mut personal_values = initialize(&mut context, &points, n)?;
    if context.stop_reason().is_some() {
        return Ok(report(
            &context,
            &points[..personal_values.len() * n],
            bounds,
            0,
            options.seed,
        ));
    }
    let mut personal = points.clone();
    let mut velocity = vec![0.0; points.len()];
    for generation in 0..options.max_generations {
        let global = context.best().expect("initialized population").x.clone();
        for i in 0..options.population_size {
            if context.stop_reason().is_some() {
                return Ok(report(&context, &points, bounds, generation, options.seed));
            }
            for j in 0..n {
                let index = i * n + j;
                let v = options.inertia * velocity[index]
                    + options.cognitive * rng.random::<f64>() * (personal[index] - points[index])
                    + options.social * rng.random::<f64>() * (global[j] - points[index]);
                if !v.is_finite() {
                    return Err(context.failure(Error::NonFinite("PSO velocity")));
                }
                let limit = options.velocity_limit * bounds.width(j);
                velocity[index] = v.clamp(-limit, limit);
                let trial = points[index] + velocity[index];
                if !trial.is_finite() {
                    return Err(context.failure(Error::NonFinite("PSO position")));
                }
                points[index] = trial.clamp(bounds.lower()[j], bounds.upper()[j]);
                if trial != points[index] || bounds.width(j) == 0.0 {
                    velocity[index] = 0.0;
                }
            }
            let value = context
                .evaluate(&points[i * n..(i + 1) * n])
                .map_err(|e| context.failure(e))?;
            if value < personal_values[i] {
                personal_values[i] = value;
                personal[i * n..(i + 1) * n].copy_from_slice(&points[i * n..(i + 1) * n]);
            }
        }
        if context.stop_reason().is_some() {
            return Ok(report(
                &context,
                &points,
                bounds,
                generation + 1,
                options.seed,
            ));
        }
    }
    Ok(report(
        &context,
        &points,
        bounds,
        options.max_generations,
        options.seed,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn donors_exclude_target_and_each_other() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        for i in 0..4 {
            for _ in 0..100 {
                let d = donors(&mut rng, 4, i);
                assert!(!d.contains(&i));
                assert!(d[0] != d[1] && d[0] != d[2] && d[1] != d[2]);
            }
        }
    }
}
