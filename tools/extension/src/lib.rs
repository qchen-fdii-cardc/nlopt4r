//! An external solver using only nlopt4r's public utilities.
use nlopt4r::core::{BudgetOptions, EvaluatedPoint, EvaluationContext, SearchTermination};
use nlopt4r::{Bounds, Error, EvaluationCounts, Evaluator};

pub struct PatternResult {
    pub point: EvaluatedPoint,
    pub evaluations: EvaluationCounts,
    pub termination_reason: SearchTermination,
    pub iterations: usize,
}
pub fn pattern_search<P: Evaluator + ?Sized>(
    p: &P,
    bounds: &Bounds,
    budget: &BudgetOptions,
) -> Result<PatternResult, Error> {
    let mut context = EvaluationContext::new(p, bounds.dimension(), budget)?;
    let mut x = bounds.midpoint();
    context.evaluate(x.as_slice())?;
    let mut step = 0.5;
    let mut iterations = 0;
    while step > 1e-7 && context.stop_reason().is_none() {
        let before = context.best().expect("initial evaluation").value;
        for i in 0..x.len() {
            if bounds.width(i) == 0.0 {
                continue;
            }
            for sign in [-1.0, 1.0] {
                if context.stop_reason().is_some() {
                    break;
                }
                let mut trial = x.clone();
                trial[i] += sign * step * bounds.width(i);
                bounds
                    .project(trial.as_mut_slice())
                    .map_err(|e| context.failure(e))?;
                context
                    .evaluate(trial.as_slice())
                    .map_err(|e| context.failure(e))?;
            }
        }
        let best = context.best().expect("evaluated point");
        x.copy_from(&best.x);
        if best.value >= before {
            step *= 0.5;
        }
        iterations += 1;
    }
    Ok(PatternResult {
        point: context.best().expect("initial evaluation").clone(),
        evaluations: context.counts(),
        termination_reason: context
            .stop_reason()
            .unwrap_or(SearchTermination::Stagnated),
        iterations,
    })
}
