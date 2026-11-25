use nlopt4r::core::{
    BudgetOptions, Control, EvaluationContext, EvaluationEvent, SearchTermination,
};
use nlopt4r::utils::{ChaCha8Rng, Rng, SeedableRng, directional_difference};
use nlopt4r::{Bounds, Error, EvaluationError, Function};
use std::cell::Cell;

#[test]
fn finite_bounds_scaling_projection_and_seeded_sampling() {
    assert!(Bounds::uniform(2, -f64::MAX, f64::MAX).is_err());
    let bounds = Bounds::new(vec![-2.0, 3.0], vec![4.0, 3.0]).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let mut x = [0.0; 2];
    for _ in 0..50 {
        bounds.sample(&mut rng, &mut x).unwrap();
        assert!(bounds.contains(&x));
        assert_eq!(x[1], 3.0);
    }
    x = [20.0, -4.0];
    bounds.project(&mut x).unwrap();
    assert_eq!(x, [4.0, 3.0]);
    let unit = bounds.to_unit(&[1.0, 3.0]).unwrap();
    assert_eq!(
        bounds.from_unit(unit.as_slice()).unwrap().as_slice(),
        [1.0, 3.0]
    );
    assert!(bounds.project(&mut [f64::NAN, 0.0]).is_err());
    assert!(rng.random::<f64>().is_finite());
}
#[test]
fn context_strict_budget_attempts_and_checkpoints() {
    let calls = Cell::new(0);
    let p = Function::new(|x: &[f64], _: Option<&mut [f64]>| {
        calls.set(calls.get() + 1);
        if x[0] < 0.0 {
            Err(EvaluationError::from("domain"))
        } else {
            Ok(x[0] * x[0])
        }
    });
    let mut context = EvaluationContext::new(
        &p,
        1,
        &BudgetOptions {
            max_evaluations: 2,
            target_value: None,
        },
    )
    .unwrap();
    context.evaluate(&[2.0]).unwrap();
    let e = context.evaluate(&[-1.0]).unwrap_err();
    let e = context.failure(e);
    assert_eq!(e.last_point().unwrap().value, 4.0);
    assert_eq!(context.counts().objective, 2);
    assert_eq!(calls.get(), 2);
    assert!(matches!(
        context.evaluate(&[1.0]),
        Err(Error::SearchStopped(SearchTermination::MaxEvaluations))
    ));
}
#[test]
fn target_observer_and_difference_accounting() {
    let p = Function::new(|x: &[f64], _: Option<&mut [f64]>| Ok::<_, EvaluationError>(x[0] * x[0]));
    let mut context = EvaluationContext::new(&p, 1, &BudgetOptions::default()).unwrap();
    let d = directional_difference(&mut context, &[2.0], &[1.0], 1e-5).unwrap();
    assert!((d - 4.0).abs() < 1e-8);
    assert_eq!(context.counts().objective, 2);
    let mut observer = |event: &EvaluationEvent<'_>| {
        assert!(event.best_value <= event.value);
        Ok(Control::Stop)
    };
    let mut context = EvaluationContext::new(&p, 1, &BudgetOptions::default())
        .unwrap()
        .with_observer(&mut observer);
    context.evaluate(&[1.0]).unwrap();
    assert_eq!(context.stop_reason(), Some(SearchTermination::Cancelled));
    let mut context = EvaluationContext::new(
        &p,
        1,
        &BudgetOptions {
            max_evaluations: 1,
            target_value: Some(0.0),
        },
    )
    .unwrap();
    context.evaluate(&[0.0]).unwrap();
    assert_eq!(
        context.stop_reason(),
        Some(SearchTermination::TargetReached)
    );
}

#[test]
fn failed_observer_and_overflowed_difference_have_explicit_safe_recovery() {
    let calls = Cell::new(0);
    let p = nlopt4r::value_only(|x: &[f64]| {
        calls.set(calls.get() + 1);
        Ok::<_, EvaluationError>(x[0] * x[0])
    });
    let mut context = EvaluationContext::new(&p, 1, &BudgetOptions::default()).unwrap();
    assert!(directional_difference(&mut context, &[0.0], &[0.0], f64::MAX).is_err());
    assert_eq!(calls.get(), 0);
    let mut observer = |_: &EvaluationEvent<'_>| Err(EvaluationError::from("observer unavailable"));
    let mut context = EvaluationContext::new(&p, 1, &BudgetOptions::default())
        .unwrap()
        .with_observer(&mut observer);
    let error = context.evaluate(&[2.0]).unwrap_err();
    let error = context.failure(error);
    assert_eq!(error.last_point().unwrap().value, 4.0);
    assert!(matches!(
        error.root_cause(),
        Error::Evaluation {
            operation: "observer",
            ..
        }
    ));
    assert_eq!(context.counts().objective, 1);
}
