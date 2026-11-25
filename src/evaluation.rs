//! Buffered, fallible callbacks, with adapters for the original objective traits.
use crate::{Error, Matrix, Objective, Vector, vector};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationError {
    pub message: String,
}

impl EvaluationError {
    pub fn new(error: impl std::fmt::Display) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}
impl From<&str> for EvaluationError {
    fn from(message: &str) -> Self {
        Self::new(message)
    }
}
impl From<String> for EvaluationError {
    fn from(message: String) -> Self {
        Self { message }
    }
}
impl std::fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for EvaluationError {}

/// Implement directly for application types, or use `Function` for closures.
/// Fill every gradient entry when requested; no derivative is needed for value-only trials.
pub trait Evaluator {
    fn evaluate(&self, x: &[f64], gradient: Option<&mut [f64]>) -> Result<f64, Error>;
    fn evaluate_hessian(&self, _x: &[f64]) -> Result<Option<Matrix>, Error> {
        Ok(None)
    }
}

impl<P: Objective + ?Sized> Evaluator for P {
    fn evaluate(&self, x: &[f64], gradient: Option<&mut [f64]>) -> Result<f64, Error> {
        let x = Vector::from_column_slice(x);
        let value = self.value(&x);
        if let Some(buffer) = gradient {
            let g = self.gradient(&x);
            vector(&g, buffer.len(), "gradient")?;
            buffer.copy_from_slice(g.as_slice());
        }
        Ok(value)
    }
    fn evaluate_hessian(&self, x: &[f64]) -> Result<Option<Matrix>, Error> {
        Ok(self.hessian(&Vector::from_column_slice(x)))
    }
}

type HessianCallback = fn(&[f64]) -> Result<Matrix, EvaluationError>;

/// Wrap a fallible value-only callback. Gradient requests are explicit errors.
pub fn value_only<F>(callback: F) -> ValueOnly<F>
where
    F: Fn(&[f64]) -> Result<f64, EvaluationError>,
{
    ValueOnly(callback)
}

pub struct ValueOnly<F>(F);
impl<F> Evaluator for ValueOnly<F>
where
    F: Fn(&[f64]) -> Result<f64, EvaluationError>,
{
    fn evaluate(&self, x: &[f64], gradient: Option<&mut [f64]>) -> Result<f64, Error> {
        if gradient.is_some() {
            return Err(Error::Evaluation {
                operation: "gradient",
                source: EvaluationError::from("value-only callback cannot provide a gradient"),
            });
        }
        self.0(x).map_err(|source| Error::Evaluation {
            operation: "objective",
            source,
        })
    }
}

pub struct Function<F, H = HessianCallback> {
    callback: F,
    hessian: Option<H>,
}

impl<F> Function<F> {
    pub fn new(callback: F) -> Self {
        Self {
            callback,
            hessian: None,
        }
    }
}
impl<F, H> Function<F, H> {
    pub fn with_hessian<J>(self, hessian: J) -> Function<F, J> {
        Function {
            callback: self.callback,
            hessian: Some(hessian),
        }
    }
}
impl<F, H> Evaluator for Function<F, H>
where
    F: Fn(&[f64], Option<&mut [f64]>) -> Result<f64, EvaluationError>,
    H: Fn(&[f64]) -> Result<Matrix, EvaluationError>,
{
    fn evaluate(&self, x: &[f64], gradient: Option<&mut [f64]>) -> Result<f64, Error> {
        if let Some(g) = &gradient
            && g.len() != x.len()
        {
            return Err(Error::DimensionMismatch {
                context: "gradient buffer",
                expected: (x.len(), 1),
                actual: (g.len(), 1),
            });
        }
        (self.callback)(x, gradient).map_err(|source| Error::Evaluation {
            operation: "objective",
            source,
        })
    }
    fn evaluate_hessian(&self, x: &[f64]) -> Result<Option<Matrix>, Error> {
        self.hessian
            .as_ref()
            .map(|h| h(x))
            .transpose()
            .map_err(|source| Error::Evaluation {
                operation: "Hessian",
                source,
            })
    }
}
