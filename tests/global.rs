use nlopt4r::core::{BudgetOptions, SearchTermination};
use nlopt4r::global::{DeOptions, PsoOptions, differential_evolution, particle_swarm};
use nlopt4r::{Bounds, Error, EvaluationError, Function, Vector};
use std::cell::Cell;

#[test]
fn seeded_methods_reach_targets_without_derivatives() {
    let bounds = Bounds::new(vec![-4.0, -4.0, 2.0], vec![4.0, 4.0, 2.0]).unwrap();
    let p = Function::new(|x: &[f64], gradient: Option<&mut [f64]>| {
        assert!(gradient.is_none());
        assert!(bounds.contains(x));
        Ok::<_, EvaluationError>((x[0] - 0.3).powi(2) + (x[1] + 0.7).powi(2))
    });
    let budget = BudgetOptions {
        max_evaluations: 15_000,
        target_value: Some(1e-9),
    };
    for seed in [0, 7, 42] {
        let de = DeOptions {
            budget: budget.clone(),
            seed,
            ..DeOptions::default()
        };
        let a = differential_evolution(&p, &bounds, &de).unwrap();
        let b = differential_evolution(&p, &bounds, &de).unwrap();
        assert_eq!(a.x, b.x);
        assert_eq!(a.evaluations, b.evaluations);
        assert!(a.target_reached(), "{a:?}");
        let options = PsoOptions {
            budget: budget.clone(),
            seed,
            ..PsoOptions::default()
        };
        let a = particle_swarm(&p, &bounds, &options).unwrap();
        let b = particle_swarm(&p, &bounds, &options).unwrap();
        assert_eq!(a.x, b.x);
        assert!(a.target_reached(), "{a:?}");
        assert_eq!(a.evaluations.gradient, 0);
    }
}
#[test]
fn partial_initialization_and_generations_preserve_best_and_budgets() {
    let bounds = Bounds::uniform(1, -2.0, 2.0).unwrap();
    for limit in [1, 3, 4, 5, 7, 8, 9] {
        let calls = Cell::new(0);
        let p = Function::new(|x: &[f64], _: Option<&mut [f64]>| {
            calls.set(calls.get() + 1);
            Ok::<_, EvaluationError>(x[0] * x[0])
        });
        let initial = vec![
            Vector::from_vec(vec![2.0]),
            Vector::from_vec(vec![1.0]),
            Vector::from_vec(vec![0.5]),
            Vector::from_vec(vec![-1.0]),
        ];
        let de = DeOptions {
            population_size: 4,
            budget: BudgetOptions {
                max_evaluations: limit,
                target_value: None,
            },
            initial_population: Some(initial.clone()),
            ..DeOptions::default()
        };
        let a = differential_evolution(&p, &bounds, &de).unwrap();
        assert_eq!(a.evaluations.objective, limit);
        assert_eq!(calls.get(), limit);
        assert_eq!(a.termination_reason, SearchTermination::MaxEvaluations);
        assert_eq!(a.value, a.x[0] * a.x[0]);
        let pso = PsoOptions {
            population_size: 4,
            budget: de.budget.clone(),
            initial_population: Some(initial),
            ..PsoOptions::default()
        };
        let a = particle_swarm(&p, &bounds, &pso).unwrap();
        assert_eq!(a.evaluations.objective, limit);
        assert_eq!(a.value, a.x[0] * a.x[0]);
        assert!(bounds.contains(a.x.as_slice()));
    }
}
#[test]
fn invalid_populations_and_callback_failures_are_explicit() {
    let bounds = Bounds::uniform(1, -2.0, 2.0).unwrap();
    let calls = Cell::new(0);
    let p = Function::new(|_: &[f64], _: Option<&mut [f64]>| {
        calls.set(calls.get() + 1);
        if calls.get() > 1 {
            Err(EvaluationError::from("simulation failed"))
        } else {
            Ok(1.0)
        }
    });
    let error = differential_evolution(&p, &bounds, &DeOptions::default()).unwrap_err();
    assert!(matches!(error.root_cause(), Error::Evaluation { .. }));
    assert_eq!(error.last_point().unwrap().value, 1.0);
    assert!(
        differential_evolution(
            &p,
            &bounds,
            &DeOptions {
                population_size: 3,
                ..DeOptions::default()
            }
        )
        .is_err()
    );
    assert!(
        particle_swarm(
            &p,
            &bounds,
            &PsoOptions {
                inertia: f64::NAN,
                ..PsoOptions::default()
            }
        )
        .is_err()
    );
    let invalid = DeOptions {
        initial_population: Some(vec![Vector::from_vec(vec![5.0]); 40]),
        ..DeOptions::default()
    };
    assert!(differential_evolution(&p, &bounds, &invalid).is_err());
}
#[test]
fn forced_crossover_changes_a_one_dimensional_target_when_cr_zero() {
    let bounds = Bounds::uniform(1, -4.0, 4.0).unwrap();
    let initial = vec![1.0, 2.0, 3.0, 4.0]
        .into_iter()
        .map(|x| Vector::from_vec(vec![x]))
        .collect();
    let calls = Cell::new(0);
    let changed = Cell::new(false);
    let p = Function::new(|x: &[f64], _: Option<&mut [f64]>| {
        if calls.get() == 4 {
            changed.set(x[0] != 1.0);
        }
        calls.set(calls.get() + 1);
        Ok::<_, EvaluationError>(x[0] * x[0])
    });
    differential_evolution(
        &p,
        &bounds,
        &DeOptions {
            population_size: 4,
            crossover: 0.0,
            max_generations: 1,
            seed: 11,
            initial_population: Some(initial),
            ..DeOptions::default()
        },
    )
    .unwrap();
    assert!(changed.get());
}

#[test]
fn observer_sees_monotone_best_and_cancels_without_an_extra_evaluation() {
    use nlopt4r::core::{Control, EvaluationEvent};
    use nlopt4r::global::differential_evolution_with_observer;
    let bounds = Bounds::uniform(2, -3.0, 3.0).unwrap();
    let p =
        nlopt4r::value_only(|x: &[f64]| Ok::<_, EvaluationError>(x.iter().map(|x| x * x).sum()));
    let mut previous = f64::INFINITY;
    let mut calls = 0;
    let mut observer = |event: &EvaluationEvent<'_>| {
        assert!(event.best_value <= previous);
        previous = event.best_value;
        calls += 1;
        Ok(if calls == 43 {
            Control::Stop
        } else {
            Control::Continue
        })
    };
    let r = differential_evolution_with_observer(
        &p,
        &bounds,
        &DeOptions::default(),
        Some(&mut observer),
    )
    .unwrap();
    assert_eq!(r.termination_reason, SearchTermination::Cancelled);
    assert_eq!(r.evaluations.objective, 43);
    assert_eq!(r.evaluations.gradient, 0);
}

#[test]
fn pso_generations_use_one_global_best_snapshot() {
    use nlopt4r::utils::{ChaCha8Rng, Rng, SeedableRng};
    use std::cell::RefCell;
    let bounds = Bounds::uniform(1, -4.0, 4.0).unwrap();
    let initial = [3.0, -0.5, 2.0, 1.0];
    let mut tested_improving_snapshot = false;
    for seed in 0..30 {
        let observed = RefCell::new(Vec::new());
        let p = nlopt4r::value_only(|x: &[f64]| {
            observed.borrow_mut().push(x[0]);
            Ok::<_, EvaluationError>(x[0] * x[0])
        });
        let options = PsoOptions {
            seed,
            population_size: 4,
            max_generations: 1,
            inertia: 0.0,
            cognitive: 0.0,
            social: 1.5,
            velocity_limit: 1.0,
            initial_population: Some(initial.iter().map(|x| Vector::from_vec(vec![*x])).collect()),
            ..PsoOptions::default()
        };
        particle_swarm(&p, &bounds, &options).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for (i, x) in initial.iter().enumerate() {
            let _r1 = rng.random::<f64>();
            let r2 = rng.random::<f64>();
            let v = (1.5 * r2 * (-0.5 - x)).clamp(-8.0, 8.0);
            let expected = (x + v).clamp(-4.0, 4.0);
            assert_eq!(observed.borrow()[i + 4], expected);
            if i == 0 && expected.abs() < 0.5 {
                tested_improving_snapshot = true;
            }
        }
    }
    assert!(tested_improving_snapshot);
}
