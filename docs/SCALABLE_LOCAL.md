# Scalable local optimization

`nlopt4r::local` preserves `minimize`, `Method`, and `Options` as compatibility
reexports. Its new entry points are unconstrained, smooth local optimizers:

| Entry point | Derivatives | Auxiliary storage |
| --- | --- | --- |
| `lbfgs` / `LbfgsOptions` | value and gradient | O(m n), m = `history_size` |
| `newton_cg` / `NewtonCgOptions` | value, gradient, Hessian-vector products | O(n) |
| `trust_region_cg` / `TrustRegionCgOptions` | same | O(n) |

The dense `trust_region::minimize` and spectral `trust_region::subproblem`
retain their existing behavior. `steihaug` and `trust_region_cg` are also
reexported from `trust_region`. None of these algorithms is L-BFGS-B: there is
no bound handling, projected gradient, generalized Cauchy point, or subspace
bound-constrained minimization here.

## Strong Wolfe search

Write phi(a) = f(x + a d), with phi'(0) < 0. The accepted step satisfies

```text
phi(a) <= phi(0) + c1 a phi'(0)
|phi'(a)| <= c2 |phi'(0)|,       0 < c1 < c2 < 1.
```

`strong_wolfe` doubles trial steps up to a configured maximum until it finds
a bracket, then bisects strictly inside that bracket. This safeguarded zoom
does not rely on potentially unstable polynomial interpolation. One shared
trial cap covers both phases. Each trial evaluates value and gradient together;
the accepted gradient and value are returned and reused by the outer solver.
The starting point's supplied derivatives are not reevaluated by the search.
Callback errors and nonfinite derivatives propagate immediately.

## Limited-memory BFGS

Let s = x(next) - x and y = g(next) - g. A correction is retained only if
s'y exceeds `curvature_tolerance * ||s|| * ||y||`, and its reciprocal and
initial scaling are finite and positive. Rejected pairs do not evict older
valid corrections. Only the most recent m valid vector pairs are stored.
The scaled initial inverse model is gamma I, gamma = s'y / y'y.
The two-loop recurrence applies the inverse model without forming an n-by-n
matrix:

```text
q = g
newest to oldest: a[i] = rho[i] s[i]'q; q -= a[i] y[i]
r = gamma q
oldest to newest: b = rho[i] y[i]'r; r += (a[i] - b) s[i]
d = -r,          rho[i] = 1 / (s[i]'y[i]).
```

A nonfinite or insufficiently descending direction resets history and falls
back to the negative gradient. Failed curvature updates are skipped, not
replaced with a dense approximation.

## Matrix-free second derivatives

Implement `local::HessianVector`, which extends the existing `Evaluator`:

```rust
use nlopt4r::{Error, Evaluator};
use nlopt4r::local::HessianVector;

struct Quadratic;
impl Evaluator for Quadratic {
    fn evaluate(&self, x: &[f64], gradient: Option<&mut [f64]>) -> Result<f64, Error> {
        if let Some(g) = gradient {
            g.copy_from_slice(x);
        }
        Ok(0.5 * x.iter().map(|v| v * v).sum::<f64>())
    }
}
impl HessianVector for Quadratic {
    fn hessian_vector(&self, _: &[f64], v: &[f64], out: &mut [f64]) -> Result<(), Error> {
        out.copy_from_slice(v);
        Ok(())
    }
}
```

The operator must be symmetric and linear at each fixed x; symmetry cannot be
certified from isolated products. Every output entry must be filled; buffers
start with NaNs so incomplete writes are reported explicitly. These methods
never call `Evaluator::evaluate_hessian`.

Newton-CG approximately solves H p = -g by conjugate gradients with an inner
relative-residual tolerance and iteration cap. On nonpositive curvature it
uses the accumulated direction (or -g on the first inner iteration).
A descent safeguard precedes strong Wolfe search. This is truncated Newton,
not a claim that an indefinite Newton equation is solved exactly.

Steihaug CG approximately minimizes q(p) = g'p + p'Hp/2 subject to
||p|| <= radius. An unconstrained CG step that reaches the boundary is truncated
at the positive root of ||p + tau d|| = radius. On nonpositive curvature both
boundary endpoints are compared and the lower quadratic model is selected.
The model change is accumulated from existing products; no extra Hessian
callback is needed to report it. Residual, boundary, negative-curvature and
iteration-cap exits are distinguished by `SteihaugTermination`.

The outer trust-region solver uses actual/predicted decrease to accept steps,
shrinks the radius for poor agreement, and expands it for good agreement at a
boundary step. Rejected trials leave the last accepted iterate unchanged.
Zero-gradient starts terminate at first-order stationarity: unlike the dense
spectral solver, Steihaug does not resolve hidden negative-curvature hard cases.
No preconditioner is currently applied.

## Stopping, accounting and recovery

All outer methods return the existing `Solution` and use `StopOptions`:
gradient convergence, iteration limits, strict objective budgets, and optional
value/step stagnation. Stagnation is not convergence. `max_evaluations` includes
the starting evaluation, rejected trials and failed objective callbacks.
No callback is made after the objective budget is exhausted.

`evaluations.objective` and `evaluations.gradient` count the combined callbacks.
`evaluations.hessian` counts Hessian-vector callbacks, including failures, not
dense Hessian constructions. Products do not consume the objective budget.
`evaluations.subproblem` counts inner Newton-CG/Steihaug solves. Standalone
`steihaug` reports its own products and iterations.

After a valid accepted checkpoint exists, exhausted budgets return that
checkpoint with `MaxEvaluations`. Other callback/model/search failures wrap
the original error in `SolverFailure`; `root_cause()` and `last_solution()`
recover the source and last valid state with final counters. A failure at the
initial evaluation cannot supply a valid checkpoint.

## Algorithm references

Equations above are restated directly rather than reproducing source text:

- J. Nocedal, *Updating quasi-Newton matrices with limited storage*,
  Mathematics of Computation 35 (1980), 773–782, DOI:
  [10.1090/S0025-5718-1980-0572855-7](https://doi.org/10.1090/S0025-5718-1980-0572855-7).
- J. Nocedal and S. Wright, *Numerical Optimization*, second edition (2006),
  algorithms 3.5–3.6 (Wolfe bracketing/zoom), 7.4 (L-BFGS two-loop recursion),
  and 7.1 (truncated Newton-CG).
- T. Steihaug, *The conjugate gradient method and trust regions in large scale
  optimization*, SIAM Journal on Numerical Analysis 20 (1983), 626–637, DOI:
  [10.1137/0720042](https://doi.org/10.1137/0720042).

Focused regression tests are in `tests/scalable_local.rs`, plus the internal
L-BFGS correction-rejection test. They cover Rosenbrock, ill-conditioned SPD
quadratics, a large matrix-free problem, strong Wolfe inequalities, negative
curvature, boundary truncation, exact callback accounting and error recovery.
