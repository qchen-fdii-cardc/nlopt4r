# Robust local solvers

The new interfaces are opt-in: legacy `least_squares::levenberg_marquardt`
and `constrained::minimize` keep their existing algorithms and reports.
These are dense, local methods with analytic first derivatives; none claims
global optimality or supplies numerical derivatives.

## Adaptive rectangular-QR Levenberg–Marquardt

Use `least_squares::levenberg_marquardt_with_options` with `LmOptions`.
Its `stopping: StopOptions` field controls termination and residual budgets;
the remaining fields configure damping and model acceptance. This interface
neither uses nor validates Armijo line-search settings.
For residuals \(r\in\mathbb R^m\), Jacobian \(J\in\mathbb R^{m\times n}\),
and diagonal column norms \(D\), the method solves

```text
min_z || [J D^-1; sqrt(mu) I] z - [-r; 0] ||_2
d = D^-1 z
```

Householder QR and triangular substitution solve the augmented rectangular
system directly: neither a normal matrix nor `J'J` is formed. Nonzero column
norms are used regardless of their magnitude; identically zero columns use
one. Positive damping makes the augmented system full column rank even for
underdetermined or rank-deficient residual systems. It does not identify the
rank of the undamped Jacobian or certify uniqueness.

An actual/predicted reduction ratio for the **undamped** residual model accepts
or rejects each trial. Successful steps use Nielsen's cubic damping update;
rejected steps increase damping geometrically. Damping stays above the supplied
positive floor. A finite rejection limit or numerical failure returns an
explicit error with the last complete accepted report. Overflow is reported,
not silently accepted. This is a scaled adaptive LM method, not an exact
trust-region-radius solver.

Stationarity and `Solution::gradient_norm` are
`||D^-1 J' r||_2` in this interface, making the test insensitive to independent
variable-unit changes. This is not the unscaled gradient norm; small
stationarity does not imply small residuals. Column scaling does not change
residual weights or address badly scaled residual *rows*.

Every residual invocation, including rejected trials and fallible callbacks,
consumes one `StopOptions::max_evaluations` unit. Jacobian invocations are
counted separately. Accepted residuals are cached. Budget exhaustion returns
the last fully evaluated accepted checkpoint (not a rejected trial); Jacobian
or residual failures preserve that checkpoint in `Error::SolverFailure`.
There is no checkpoint if the initial residual/Jacobian cannot be evaluated.

## Feasible projected gradient

`bounded::projected_gradient(problem, x0, bounds, options)` projects the initial
point, then uses `d = P(x-gradient)-x` and Armijo backtracking on the feasible
projected segment. Every trial is projected before its callback, including when
the configured initial line-search step exceeds one. Fixed coordinates remain
fixed. Accepted objective/gradient pairs are cached, and every paired evaluation
consumes one objective-budget unit.

The reported stationarity is the unit-step gradient mapping
`||x-P(x-gradient)||_2`, also available as
`bounded::projected_gradient_norm`. Its zero set is precisely box first-order
stationarity: positive gradients at lower bounds, negative gradients at upper
bounds, and arbitrary gradients on fixed coordinates need not vanish. Away
from stationarity this mapping depends on coordinate units and box widths.
It must not be confused with the norm of an unconstrained gradient.

For step lengths at most one, convexity of the box ensures feasibility and
the projection inequality gives `gradient' d <= -||d||²`. With a smooth
objective and the usual bounded-level-set/Lipschitz-gradient assumptions,
backtracking supplies descent and first-order convergence. This implementation
has no quasi-Newton history, generalized Cauchy point, or reduced-space
minimization: it is **not L-BFGS-B**.

## SQP feasibility restoration

`constrained::sqp_with_restoration` accepts `SqpOptions` and returns
`SqpSolution { constrained, restoration }`. A failure returns
`SqpError { source, restoration, last_constrained }`, preserving the same event
history and a boxed full outer checkpoint, including equality/inequality
multipliers and the final counts/status. The checkpoint is absent when no
complete outer state was evaluated. Its source is the standard `Error` chain,
which also retains the compatibility `Solution` checkpoint.
Each `RestorationEvent` contains the
original QP error, outer iteration, initial/final infinity-norm violation,
accepted restoration iteration count, and optional failure.

Only an **infeasible linearized QP** triggers restoration. Callback failures,
QP iteration/budget exhaustion, or singular numerical systems do not silently
start another algorithm. Restoration minimizes the constraint-only merit

```text
theta(x) = 1/2 (||h(x)||² + ||min(g(x),0)||²),  g(x) >= 0.
```

It uses the equality rows and currently violated inequality rows, an augmented
damped QR least-squares direction, and backtracking on the actual nonlinear
merit. All inequalities are reevaluated at every trial, including previously
inactive ones. Once the merit decreases by the configured fraction (or
feasibility tolerance is reached), the method resets the SQP Hessian model and
multipliers and resumes the **same SQP**. Restoration acceptance ignores the
objective, but the objective/gradient are evaluated to produce valid reports.
An infeasible nonlinear start can therefore recover even when no step solves
its initial linearized constraints.

This is a merit-based local restoration phase, not a global feasibility
oracle, filter-SQP, elastic exact-penalty SQP, or a switch to an augmented
Lagrangian solver. A stationary infeasible restoration merit reports
`SubproblemStopped` with `StepStagnation`; this is **not** a proof that the
nonlinear feasible set is empty. Repeated restoration, numerical errors,
iteration limits, and insufficient merit decrease remain explicit outcomes.
Usual SQP local convergence assumptions include smooth derivatives and
constraint qualifications; restoration cannot remove degeneracy.

Outer objective trials obey `solver.solver.stopping.max_evaluations`.
The typed SQP interface caches accepted trial states and checkpoints them with
the updated multipliers before the next iteration. Thus a budget stop or
subsequent callback failure retains the full last accepted constrained state
without spending an extra evaluation to reconstruct it. Legacy SQP evaluation
scheduling is unchanged.
`max_subproblem_evaluations` is a separate **whole-run** synthetic budget
covering QP phase I/active-set models and restoration direction models.
`max_subproblem_iterations` limits each QP phase (phase I and the objective
active set separately), while their combined evaluations share the strict
whole-run synthetic budget;
`max_restoration_iterations` limits each restoration activation. All accepted
restoration work counts against the outer iteration schedule through the
activation, and every activation consumes an outer iteration. Exhaustion of
the objective budget returns `MaxEvaluations`; exhaustion of the synthetic
budget is an explicitly logged/reportable `SubproblemStopped` error.

## References

* J. J. Moré, “The Levenberg–Marquardt Algorithm: Implementation and Theory,”
  *Numerical Analysis*, Lecture Notes in Mathematics 630, 1978, pp. 105–116.
  DOI: [10.1007/BFb0067700](https://doi.org/10.1007/BFb0067700).
* K. Madsen, H. B. Nielsen, O. Tingleff, *Methods for Non-Linear Least Squares
  Problems*, 2nd ed., Technical University of Denmark, 2004, §3.2.
* D. P. Bertsekas, *Nonlinear Programming*, 2nd ed., Athena Scientific, 1999:
  gradient projection and constrained first-order conditions.
* J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
  2006, Chapters 16 and 18: quadratic programming, SQP, and restoration.
* R. H. Byrd, P. Lu, J. Nocedal, C. Zhu, “A Limited Memory Algorithm for Bound
  Constrained Optimization,” *SIAM Journal on Scientific Computing* 16(5),
  1995, pp. 1190–1208. DOI:
  [10.1137/0916069](https://doi.org/10.1137/0916069).
  This describes L-BFGS-B machinery deliberately **not** claimed here.
