use nlopt4r::core::BudgetOptions;
use nlopt4r::global::{DeOptions, PsoOptions, differential_evolution, particle_swarm};
use nlopt4r::{Bounds, EvaluationError, value_only};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target = [0.3, -0.7, 1.2];
    let objective = value_only(|x: &[f64]| {
        Ok::<_, EvaluationError>(x.iter().zip(target).map(|(x, t)| (x - t).powi(2)).sum())
    });
    let bounds = Bounds::uniform(3, -4.0, 4.0)?;
    let budget = BudgetOptions {
        max_evaluations: 15_000,
        target_value: Some(1e-8),
    };
    let de = differential_evolution(
        &objective,
        &bounds,
        &DeOptions {
            budget: budget.clone(),
            seed: 7,
            ..DeOptions::default()
        },
    )?;
    let pso = particle_swarm(
        &objective,
        &bounds,
        &PsoOptions {
            budget,
            seed: 7,
            ..PsoOptions::default()
        },
    )?;
    for (name, result) in [("DE", de), ("PSO", pso)] {
        if !result.target_reached() {
            return Err(format!("{name} did not reach target: {result:?}").into());
        }
        println!(
            "{name}: f={}, calls={}, seed={}",
            result.value, result.evaluations.objective, result.seed
        );
    }
    Ok(())
}
