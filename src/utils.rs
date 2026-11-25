//! Small numerical building blocks for external solver implementations.
use crate::{Error, Matrix, Vector};
pub use rand::{Rng, SeedableRng};
pub use rand_chacha::ChaCha8Rng;

pub fn validate_slice(x: &[f64], n: usize, context: &'static str) -> Result<(), Error> {
    if x.len() != n {
        return Err(Error::DimensionMismatch {
            context,
            expected: (n, 1),
            actual: (x.len(), 1),
        });
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite(context));
    }
    Ok(())
}
pub fn solve_dense(a: &Matrix, b: &Vector) -> Result<Vector, Error> {
    crate::matrix(a, b.len(), b.len(), "linear system")?;
    crate::vector(b, b.len(), "linear right-hand side")?;
    crate::solve(a, b)
}
/// Central directional difference. Both evaluations consume the context budget.
pub fn directional_difference<P: crate::Evaluator + ?Sized>(
    context: &mut crate::core::EvaluationContext<'_, P>,
    x: &[f64],
    direction: &[f64],
    step: f64,
) -> Result<f64, Error> {
    validate_slice(direction, x.len(), "difference direction")?;
    crate::positive(step, "difference step")?;
    let plus: Vec<_> = x.iter().zip(direction).map(|(x, d)| x + step * d).collect();
    let minus: Vec<_> = x.iter().zip(direction).map(|(x, d)| x - step * d).collect();
    crate::finite(
        (context.evaluate(&plus)? - context.evaluate(&minus)?) / (2.0 * step),
        "directional difference",
    )
}
