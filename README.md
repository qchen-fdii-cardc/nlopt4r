# nlopt4r

A reusable, pure-Rust library implementing the optimization families in
the adjacent [MATLAB programs](../Matlab程序/). This is an educational dense
solver library, **not** a binding to the unrelated upstream NLopt C library.

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
hard-coded objective functions, dimensions, globals, file I/O, native libraries,
or runtime dependencies. Capturing/borrowing closures work. Implement `Objective`
for application types, or pass `(value, gradient)` closures. Use
`(value, gradient, hessian)` for Newton methods. Solvers accept borrowed trait
objects as well as concrete problem types.

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
Call `quadratic::minimize(&qp, &x0, &options)` or
`quadratic::equality_constrained(&h, &c, &a, &b)`.

The Hessian must be symmetric positive semidefinite. Consistent singular KKT
systems use an SVD least-norm solution; inconsistent or numerically unsolvable
systems return `SingularSystem`. In particular, general linear programs and
unbounded flat-direction subproblems are not supported. Prefer positive definite,
well-scaled QPs. Redundant consistent constraints are supported. Phase I and the
main active-set solve each have `max_iterations` as their budget; the returned
iteration count covers the main solve.

### Nonlinear constraints

Implement `constrained::ConstrainedProblem` (which extends `Objective`) and return
`Constraints` from its `constraints` method:

- `equalities`: `h(x) = 0`, `inequalities`: `g(x) >= 0`.
- Jacobians have **one row per constraint**, unlike the transposed derivative
  arrays in some MATLAB scripts.
- No constraints of one kind: use an empty vector and a `0 x n` matrix.
  `Constraints::empty(n)` creates both empty sets.
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
