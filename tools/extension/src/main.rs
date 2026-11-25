use nlopt4r::core::BudgetOptions;
use nlopt4r::{Bounds, Error, EvaluationError, Function};
use nlopt4r_extension_example::pattern_search;
fn main() -> Result<(), Error> {
    let target = [0.3, -0.7];
    let p = Function::new(|x: &[f64], _: Option<&mut [f64]>| {
        Ok::<_, EvaluationError>(x.iter().zip(target).map(|(x, t)| (x - t).powi(2)).sum())
    });
    let result = pattern_search(
        &p,
        &Bounds::uniform(2, -2.0, 2.0)?,
        &BudgetOptions::default(),
    )?;
    assert!(result.point.value < 1e-10);
    println!("External bounded pattern search: f={}", result.point.value);
    Ok(())
}
