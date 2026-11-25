# Reusable solver foundations

Existing root solver functions and objective tuples remain supported.
The `core`, `problem` and `utils` namespaces expose building blocks without
requiring a solver registry or universal configuration object.

## Bounds

`Bounds::new(lower, upper)` validates nonempty matching dimensions, finite
ordered endpoints and finite widths. Equal endpoints fix a coordinate.
Projection rejects non-finite input rather than silently replacing it.
Population methods can therefore require the objective to be defined only
inside this domain.

Normalized coordinates use `x_i = lower_i + u_i*(upper_i-lower_i)`, `u_i` in
`[0,1]`. The inverse map returns zero for fixed coordinates. Use `sample` with an
explicit seeded RNG; no process-global random state is used. Reproducibility
depends on RNG/version and floating-point platform, not seed alone.

## Evaluation context

`core::EvaluationContext` accepts borrowed `Evaluator` implementations. Its
`BudgetOptions` separates objective budgets/targets from gradient tolerances:

```rust
use nlopt4r::{EvaluationError, value_only};
use nlopt4r::core::{BudgetOptions, EvaluationContext};

let objective = value_only(|x: &[f64]| {
    Ok::<_, EvaluationError>(x.iter().map(|v| v*v).sum())
});
let mut context = EvaluationContext::new(
    &objective, 2,
    &BudgetOptions { max_evaluations: 100, target_value: Some(1e-8) },
)?;
context.evaluate(&[1.0, 2.0])?;
```

Every attempted callback consumes one objective call, including failures.
Combined gradient requests also increment the gradient count. Shape, finite
values and complete gradient buffers are checked. `directional_difference`
consumes two context evaluations, making numerical derivative cost explicit.

Inspect `stop_reason()` before the next call. A target hit can stop a search,
but a caller-supplied target is not an optimality certificate. `with_observer`
accepts a borrowed, fallible callback receiving each evaluated point and
counters. `Control::Stop` requests cancellation. Observer errors propagate.

The best valid point is retained even if a later callback fails; `failure`
attaches it to `Error::SearchFailure`, accessible through `last_point()`.
This value-only recovery has no fictitious gradient. Existing gradient-based
solver errors retain `last_solution()` and their original semantics.

## Implement an algorithm outside nlopt4r

Run the independent workspace application:

```sh
cargo run -p nlopt4r-extension-example
```

Its [source](../tools/extension/src/main.rs) implements bounded coordinate
pattern search using only public context/bounds utilities. It evaluates feasible
positive/negative coordinate steps and halves the radius when none improves
the incumbent. This is an extension example, not a global optimum guarantee.

## Test separation

```sh
cargo test                         # pure-Rust solver tests
cargo test -p nlopt4r-reference    # requires CMake/C, not a published dependency
```

The reference target still uses [tests/nlopt](../tests/nlopt/main.rs).
See [ROADMAP](ROADMAP.md) for later local/global methods and release gates.
