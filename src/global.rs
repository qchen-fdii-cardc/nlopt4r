//! Seeded, bounded, derivative-free population optimization.
use crate::core::{BudgetOptions, EvaluationContext, Observer, SearchTermination};
use crate::utils::{ChaCha8Rng, Rng, SeedableRng};
use crate::{Bounds, Error, EvaluationCounts, Evaluator, Vector};

#[derive(Debug, Clone)]
pub struct GlobalSolution {
    pub x: Vector,
    pub value: f64,
    pub generations: usize,
    pub evaluations: EvaluationCounts,
    pub termination_reason: SearchTermination,
    pub seed: u64,
    /// Maximum coordinate spread divided by its domain width.
    pub diversity: f64,
}
impl GlobalSolution {
    pub fn target_reached(&self) -> bool {
        self.termination_reason == SearchTermination::TargetReached
    }
}
#[derive(Debug, Clone)]
pub struct DeOptions {
    pub budget: BudgetOptions,
    pub seed: u64,
    pub population_size: usize,
    pub max_generations: usize,
    pub weight: f64,
    pub crossover: f64,
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
#[derive(Debug, Clone)]
pub struct PsoOptions {
    pub budget: BudgetOptions,
    pub seed: u64,
    pub population_size: usize,
    pub max_generations: usize,
    pub inertia: f64,
    pub cognitive: f64,
    pub social: f64,
    pub velocity_limit: f64,
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
    let mut result = Vec::with_capacity(capacity);
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
pub fn differential_evolution<P: Evaluator + ?Sized>(
    p: &P,
    bounds: &Bounds,
    options: &DeOptions,
) -> Result<GlobalSolution, Error> {
    differential_evolution_with_observer(p, bounds, options, None)
}

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
pub fn particle_swarm<P: Evaluator + ?Sized>(
    p: &P,
    bounds: &Bounds,
    options: &PsoOptions,
) -> Result<GlobalSolution, Error> {
    particle_swarm_with_observer(p, bounds, options, None)
}

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
