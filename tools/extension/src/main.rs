use nlopt4r::core::{BudgetOptions, EvaluatedPoint, EvaluationContext};
use nlopt4r::{Bounds, Error, EvaluationError, Function};

fn pattern_search<P: nlopt4r::Evaluator + ?Sized>(
    p: &P,
    bounds: &Bounds,
) -> Result<EvaluatedPoint, Error> {
    let mut context = EvaluationContext::new(p, bounds.dimension(), &BudgetOptions::default())?;
    let mut x = bounds.midpoint();
    context.evaluate(x.as_slice())?;
    let mut step = 0.5;
    while step > 1e-7 && context.stop_reason().is_none() {
        let before = context.best().expect("initial evaluation").value;
        for i in 0..x.len() {
            for sign in [-1.0, 1.0] {
                if context.stop_reason().is_some() {
                    break;
                }
                let mut trial = x.clone();
                trial[i] += sign * step * bounds.width(i);
                bounds.project(trial.as_mut_slice())?;
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
    }
    Ok(context.best().expect("initial evaluation").clone())
}
fn main() -> Result<(), Error> {
    let target = [0.3, -0.7];
    let p = Function::new(|x: &[f64], _: Option<&mut [f64]>| {
        Ok::<_, EvaluationError>(x.iter().zip(target).map(|(x, t)| (x - t).powi(2)).sum())
    });
    let result = pattern_search(&p, &Bounds::uniform(2, -2.0, 2.0)?)?;
    assert!(result.value < 1e-10);
    println!("External bounded pattern search: f={}", result.value);
    Ok(())
}
