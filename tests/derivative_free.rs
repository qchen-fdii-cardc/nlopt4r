use nlopt4r::core::{BudgetOptions, SearchTermination};
use nlopt4r::derivative_free::{Options, nelder_mead, pattern_search};
use nlopt4r::{Bounds, EvaluationError, Vector, value_only};

#[test]
fn bounded_searches_solve_smooth_and_nonsmooth_problems_without_derivatives() {
    let bounds = Bounds::new(vec![-3.0, -3.0, 2.0], vec![3.0, 3.0, 2.0]).unwrap();
    let p = value_only(|x: &[f64]| {
        assert!(bounds.contains(x));
        Ok::<_, EvaluationError>((x[0] - 0.3).abs() + (x[1] + 0.7).powi(2))
    });
    let options = Options {
        budget: BudgetOptions {
            max_evaluations: 8000,
            target_value: Some(1e-7),
        },
        ..Default::default()
    };
    for method in [pattern_search::<_>, nelder_mead::<_>] {
        let r = method(
            &p,
            &Vector::from_vec(vec![2.0, 2.0, 2.0]),
            &bounds,
            &options,
        )
        .unwrap();
        assert!(r.value <= 1e-7, "{r:?}");
        assert!(bounds.contains(r.x.as_slice()));
    }
}
#[test]
fn search_budget_inside_simplex_initialization_preserves_evaluated_best() {
    let bounds = Bounds::uniform(3, -3.0, 3.0).unwrap();
    let p = value_only(|x: &[f64]| Ok::<_, EvaluationError>(x.iter().map(|x| x * x).sum()));
    for limit in [1, 2, 3, 4, 5, 7] {
        let options = Options {
            budget: BudgetOptions {
                max_evaluations: limit,
                target_value: None,
            },
            ..Default::default()
        };
        let r = nelder_mead(&p, &Vector::from_element(3, 1.0), &bounds, &options).unwrap();
        assert_eq!(r.evaluations.objective, limit);
        assert_eq!(r.termination_reason, SearchTermination::MaxEvaluations);
        assert_eq!(r.value, r.x.norm_squared());
    }
}
#[test]
fn fixed_domain_and_flat_simplex_are_stagnation_not_certified_optima() {
    let p = value_only(|_: &[f64]| Ok::<_, EvaluationError>(1.0));
    let fixed = Bounds::uniform(2, 3.0, 3.0).unwrap();
    let r = nelder_mead(
        &p,
        &Vector::from_element(2, 3.0),
        &fixed,
        &Options::default(),
    )
    .unwrap();
    assert_eq!(r.evaluations.objective, 1);
    assert_eq!(r.termination_reason, SearchTermination::Stagnated);
    let bounds = Bounds::uniform(2, -1.0, 1.0).unwrap();
    let r = nelder_mead(&p, &Vector::zeros(2), &bounds, &Options::default()).unwrap();
    assert_eq!(r.termination_reason, SearchTermination::Stagnated);
    assert!(
        Options {
            initial_step: 0.0,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
}
