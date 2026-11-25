# nlopt4r

A reusable, pure-Rust library implementing the optimization families.
This is a dense solver library, **not** a binding to the unrelated upstream NLopt C library.

See the [improvement roadmap](docs/ROADMAP.md) for the proposed architecture,
PSO/DE algorithms, benchmark suites, theory documentation and publication checklist.
Implementation is tracked there with validated completion gates. Publication has
not been performed; a maintainer-approved license and source-provenance review
remain release blockers.

The [foundation guide](docs/FOUNDATIONS.md) covers validated bounds, shared
evaluation budgets, observers and the external algorithm extension example.

## Use from another Rust application or library

Add a path dependency (adjust the path to your checkout):

```toml
[dependencies]
nlopt4r = { path = "../optimization/nlopt" }
```

```rust
use nlopt4r::{minimize, Method, Options, Vector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target = Vector::from_vec(vec![3.0, -2.0]);
    let objective = (
        |x: &Vector| 0.5 * (x - &target).norm_squared(),
        |x: &Vector| x - &target,
    );
    let result = minimize(
        &objective, &Vector::zeros(2), Method::Bfgs, &Options::default(),
    )?;
    if !result.converged() {
        return Err("optimization exhausted its iteration budget".into());
    }
    println!("x = {}, f = {}", result.x, result.value);
    Ok(())
}
```

Run the included application example:

```sh
cargo run --example rosenbrock
```

`Vector` and `Matrix` are dynamically sized nalgebra `f64` types. There are no
hard-coded objective functions, dimensions, globals, file I/O, or native runtime
libraries. Runtime dependencies are pure-Rust `nalgebra` and seeded RNG crates.
Capturing/borrowing closures work. Implement `Objective`
for application types, or pass `(value, gradient)` closures. Use
`(value, gradient, hessian)` for Newton methods. Solvers accept borrowed trait
objects as well as concrete problem types.

### Stopping controls and reports

`StopOptions` holds shared stopping criteria. `Options` combines them with Armijo
settings for unconstrained and least-squares solvers. Trust regions use
`TrustRegionOptions::stopping` directly, and QP accepts `&StopOptions`; neither
requires unused Armijo settings.

```rust
use nlopt4r::{Options, StopOptions, Tolerance};

let options = Options {
    stopping: StopOptions {
        max_iterations: 2000,
        max_evaluations: Some(10_000),
        gradient_tolerance: 1e-7,
        step_tolerance: Some(Tolerance { absolute: 1e-12, relative: 1e-10 }),
        value_tolerance: None,
    },
    ..Options::default()
};
```

Optional step/value tolerances test `difference <= absolute + relative * scale`.
The scale is the maximum of the consecutive point norms or absolute objective
values. These criteria report **stagnation**, not convergence. They are disabled
by default, preserving the original numerical stopping behavior. Rejected
trust-region steps and QP working-set changes do not count as accepted progress.
Scalar search has `ScalarOptions::interval_tolerance`; use
`minimize_scalar_with_options` for budgets and configurable absolute/relative
tolerances. The original `minimize_scalar` convenience function remains available.

Optimization reports include `termination_reason` and `evaluations` as well as the existing
point, objective, stationarity norm and iteration count. Evaluation counts track
attempted objective, gradient, Hessian, constraint, residual and Jacobian calls,
including failed callbacks. A combined value/gradient call counts once in each
category. Unconstrained/trust-region value-only trials do not request gradients. For least squares, each
residual evaluation counts as an objective evaluation; Jacobians also count as
gradient evaluations. Constraint calls include their Jacobians.
QP counts describe evaluated model states rather than external callback calls.

`max_evaluations` limits objective evaluations, including rejected trials and
PHR inner evaluations. It is never exceeded. QP shares this limit across phase I
and optimization; synthetic SQP/trust-model evaluations are reported separately
as `evaluations.subproblem` and do not consume the outer objective budget.
Convergence takes precedence if the last permitted fully evaluated point
satisfies optimality criteria.

### Buffered and fallible callbacks

The original `Objective` trait and tuple closures still work through an automatic
adapter. For expensive application callbacks, use `Function` or implement
`Evaluator` directly. Input/output slices avoid callback-side gradient allocation;
the solver itself still uses dense vectors and matrices.

```rust
use nlopt4r::{EvaluationError, Function, Method, Options, minimize, Vector};

let target = [3.0, -2.0];
let problem = Function::new(|x: &[f64], gradient: Option<&mut [f64]>| {
    if let Some(g) = gradient {
        for i in 0..x.len() { g[i] = x[i] - target[i]; }
    }
    Ok::<_, EvaluationError>(
        0.5 * x.iter().zip(target).map(|(x, t)| (x - t).powi(2)).sum::<f64>()
    )
});
let result = minimize(&problem, &Vector::zeros(2), Method::Bfgs, &Options::default())?;
```

Fill every requested gradient entry; incomplete/non-finite output is an error.
Use `Function::with_hessian` for a fallible analytic Hessian. Callback errors can
use `"message".into()` or `EvaluationError::new(application_error)`; the latter
retains its display message. `Evaluator::evaluate` permits shared value/gradient
computation, while value-only trials pass `None`. For existing least-squares or
constrained types, override `try_residuals`, `try_jacobian`, `try_constraints` or
`try_lagrangian_hessian` to surface application failures without panics.

**Configuration migration:** move old `Options.max_iterations` and
`Options.gradient_tolerance` into `Options.stopping`. Move
`TrustRegionOptions.solver` stopping fields into `TrustRegionOptions.stopping`.
Pass `&options.stopping` instead of `&options` to QP. For nonlinear constraints,
shared criteria live in `ConstrainedOptions.solver.stopping`; PHR-specific penalty
and inner-iteration settings are validated only by PHR. These are intentional
configuration changes; objective tuples and solver method names are unchanged.

## Algorithms and MATLAB correspondence

| Rust API | MATLAB reference |
| --- | --- |
| `minimize`, `Method::SteepestDescent` / `NormalizedGradient` | `grad.m` / `gradd.m` |
| `Method::DampedNewton` / `ModifiedNewton` | `dampnm.m` / `revisenm.m` |
| `Method::FletcherReeves` | `frcg.m`, including periodic restarts |
| `Method::Bfgs`, `Dfp`, `Sr1`, `Broyden { phi }` | quasi-Newton programs |
| `line_search::armijo` | `armijo.m` |
| `line_search::minimize_scalar`, `GoldenSection` / `Parabolic` | `golds.m` / `qmin.m` |
| `trust_region::minimize` / `subproblem` | `trustm.m` / `trustq.m` |
| `least_squares::levenberg_marquardt` | `lmm.m` |
| `quadratic::equality_constrained` / `minimize` | `qlag.m` / `qpact.m` |
| `constrained::minimize`, `Method::Sqp` | `sqpm.m`, `qpsubp.m` |
| `constrained::Method::NewtonLagrange` / `LagrangeSqp` | `newtlagr.m` / `lagsqp.m` |
| `constrained::augmented_lagrangian` | `multphr.m`, `mpsi.m`, `dmpsi.m` |

These are mathematical implementations rather than bug-for-bug translations:

- Shared Armijo search explicitly fails when no trial is acceptable. It never
  takes the MATLAB fallback's unchecked full step.
- Quasi-Newton methods start with the identity inverse Hessian, avoid unsafe
  curvature divisions, and safeguard descent directions. Modified Newton
  increases its shift until the Hessian is positive definite. Broyden's parameter
  is in `[0, 1]`: zero gives DFP, one gives inverse BFGS.
- Parabolic search uses bounded, safeguarded Brent interpolation instead of the
  MATLAB routine's potentially unbounded bracket expansion. Both scalar methods
  assume a continuous unimodal objective on the supplied closed interval.
  Stopping uses `absolute_tolerance + sqrt(machine_epsilon) * abs(x)`.
- Trust-region subproblems use symmetric eigendecomposition and a multiplier
  search, including indefinite-Hessian hard cases, instead of the reference's
  smoothing Newton system.
- QP and SQP subproblems share an active-set solver rather than duplicating the
  MATLAB smoothing solver. QP phase I allows infeasible initial guesses.
- PHR differentiates the **projected** inequality penalty consistently and updates
  multipliers using the penalty used by the inner minimization.
- Exact equality methods use caller-supplied Lagrangian Hessians, not the
  example-specific (and partly incorrect) MATLAB Hessian expressions.

## Other solver interfaces

### Nonlinear least squares

Pass `(residuals, jacobian)` closures or implement `least_squares::LeastSquares`,
then call `least_squares::levenberg_marquardt(&problem, &x0, &options)`.
The Jacobian is `number_of_residuals x number_of_variables`; square,
overdetermined and underdetermined systems are supported. The reported objective
is **half the squared residual norm**. Convergence tests `||J'r||`, so an
inconsistent system may converge with a nonzero residual.

### Linear constraints and QP

Construct a `quadratic::QuadraticProgram` with objective
`0.5*x'H*x + c'x`, equalities `Ae*x = be`, and inequalities `Ai*x >= bi`.
Use `QuadraticProgram::unconstrained(hessian, linear)` as a starting constructor.
Call `quadratic::minimize(&qp, &x0, &stopping)` with `StopOptions`, or
`quadratic::equality_constrained(&h, &c, &a, &b)`.

The Hessian must be symmetric positive semidefinite. Consistent singular KKT
systems use an SVD least-norm solution; inconsistent or numerically unsolvable
systems return `SingularSystem`. In particular, general linear programs and
unbounded flat-direction subproblems are not supported. Prefer positive definite,
well-scaled QPs. Redundant consistent constraints are supported. Phase I and the
main active-set solve each have `max_iterations` as their budget; the returned
iteration count covers the main solve.

### Nonlinear constraints

Implement `constrained::ConstrainedProblem` (which extends `Evaluator`, with an
automatic adapter for existing `Objective` implementations) and return
`Constraints` from its `constraints` method:

- `equalities`: `h(x) = 0`, `inequalities`: `g(x) >= 0`.
- Jacobians have **one row per constraint**, unlike the transposed derivative
  arrays in some MATLAB scripts.
- No constraints of one kind: use an empty vector and a `0 x n` matrix.
  `Constraints::empty(n)` creates both empty sets.
- `Constraints::equalities_only(values, jacobian)` and `inequalities_only` create
  validated single-kind constraint sets without manually assembling empty arrays.
  `Constraints::linear(x, Ae, be, Ai, bi)` evaluates validated linear constraints.
  `Constraints::from_nonpositive` explicitly converts `c(x)<=0` and its Jacobian
  to the canonical `g(x)>=0` representation; output multipliers keep the canonical
  sign convention.
- Constraint counts must stay constant during the solve.
- Multipliers use `L = f - mu'h - lambda'g`, with `lambda >= 0`.
- For exact equality methods, also implement `lagrangian_hessian(x, mu, lambda)`
  returning the Hessian of this Lagrangian. These methods reject inequalities.

Call `constrained::minimize(&problem, &x0, method, &options)` for SQP,
Newton-Lagrange or stabilized Lagrangian SQP, or
`constrained::augmented_lagrangian(&problem, &x0, &options)` for PHR.
Multipliers start at zero. Reports include stationarity, feasibility,
complementarity (`||min(g, lambda)||`) and, for PHR, inner iteration counts.
Convergence requires all three conditions, not just a small step.

SQP requires feasible linearized constraint subproblems; it does not implement
elastic restoration when those subproblems are infeasible. PHR may be a better
choice for infeasible nonlinear starts. Newton-Lagrange methods find KKT points;
they do not certify second-order optimality.

## Errors and termination

All solvers return `Result`. Invalid options, inconsistent dimensions, missing
derivatives, non-finite callback results and failed searches are explicit errors.
Callbacks must be defined and finite at every evaluated trial point; non-finite
trials are errors, not silently rejected domain violations. Callback panics are
not intercepted. Analytic derivative correctness is the caller's responsibility.

Outer iteration exhaustion returns the last evaluated point with
`Status::MaxIterations`, never success. A zero outer budget evaluates the initial
point and reports convergence only if the stopping criteria already hold.
Nested solver exhaustion returns `Error::IterationLimit` with context.
An optimum reached on the last allowed iteration still reports convergence.
Always inspect `converged()` or `status` before using a result as an optimum.

`Status::MaxEvaluations` and `Status::Stagnated` are also non-converged results.
Reports identify whether a gradient, KKT, interval, budget or stagnation criterion
stopped the solve. Dimension errors include expected and actual shapes; line
search errors include trial counts and the last step; nested iteration errors
name the failing solver and its iteration count.
`Error::SubproblemStopped` distinguishes a phase-I stagnation stop from actual
iteration exhaustion.

When an error occurs after a valid checkpoint, `Error::last_solution()` exposes
that point and its diagnostics with `Status::Failed`. `Error::root_cause()` and
the standard error source chain expose the underlying failure. Checkpoints are
fully evaluated states, not arbitrary trial points; PHR also retains valid inner
states. If an evaluation budget is exhausted during a trial or before its
gradient is evaluated, the returned point is the last valid checkpoint, not
necessarily the smallest objective seen. Invalid input/initial evaluations may
have no checkpoint. Use the recovered point as a restart guess, not as a claimed
optimum.

Scalar error checkpoints also expose the last valid scalar point through
`last_solution()` as a length-one vector; their `gradient_norm` is NaN because
scalar search does not evaluate a gradient.

These dense, local algorithms are intended for small/moderate, reasonably scaled
problems. They do not promise global nonlinear minima, sparse performance,
automatic differentiation, bounds handling beyond explicit constraints, or
production-grade infeasibility/unboundedness certificates.

## Validation

```sh
cargo test
cargo test --release
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

Integration tests cover the MATLAB Rosenbrock objective, scalar `s^2-sin(s)`,
nonlinear residual equations, active-set QP, constrained ellipse, cylinder,
five-variable exact-Hessian equality problem and four-variable constrained
quadratic. Additional tests verify analytic optima, KKT conditions, derivative
shapes, non-finite values, singular systems, indefinite trust regions, zero
curvature, inactive inequalities, arbitrary dimensions and iteration limits.
The tests do not require a MATLAB installation.

### Comparison with crates.io NLopt

The additional integration-test group in [tests/nlopt/](tests/nlopt/) compares
`nlopt4r` against the crates.io [`nlopt`](https://crates.io/crates/nlopt) wrapper:

```sh
cargo test -p nlopt4r-reference
cargo test --release -p nlopt4r-reference
```

The wrapper is a dev-dependency of the **unpublished reference tooling crate**,
not of the solver package. Default `cargo test` and downstream builds are
pure Rust. Only the explicit reference command (or `cargo test --workspace`)
requires **CMake and a C compiler** to build bundled native NLopt; a separate
system NLopt installation is not needed. The test group retains its existing
location and coverage.

The group covers all unconstrained methods, scalar searches, Newton trust regions
(including indefinite and hard-case subproblems), square and rectangular least
squares, equality and active-set QP, exact equality methods, nonlinear SQP and
PHR. Shared problems use identical objective functions and derivatives. The
adapter negates inequalities and their gradients to translate `nlopt4r`'s
`g(x) >= 0` convention into NLopt's `c(x) <= 0`.

Comparisons check convergence, solution coordinates, objective values, and
applicable feasibility/stationarity/complementarity conditions with explicit
tolerances. NLopt failures and evaluation/time-budget exhaustion fail the tests.
Known analytic solutions provide additional checks; indefinite trust subproblems
use starts of both signs and compare objective values rather than demanding a
unique step. NLopt's L-BFGS, truncated Newton and SLSQP methods serve as numerical
references, not identical algorithm implementations. Stopping rules and iteration
counts differ, so these are correctness comparisons, not performance benchmarks.
