# Solver selection and mathematical foundations

This describes implemented methods. None of the general nonconvex solvers
certifies a global nonlinear optimum.

| Problem | Recommended entry point | Main costs / limitations |
| --- | --- | --- |
| Smooth, gradient available, moderate/large dimension | `local::lbfgs` | O(m*n) history; unbounded domain |
| Smooth, Hessian-vector products available | `local::newton_cg`, `trust_region_cg` | O(n) work vectors; operator must be symmetric |
| Small dense smooth problem | `minimize` BFGS/modified Newton, dense trust region | O(n^2) storage; dense solves/eigenanalysis O(n^3) |
| Finite box, no derivatives, small dimension | `derivative_free::pattern_search`, `nelder_mead` | Local search; simplex storage O(n^2) |
| Finite box, multimodal, no derivatives | `global::differential_evolution`, `particle_swarm` | O(population*n); stochastic search |
| Finite box, gradient available | `bounded::projected_gradient` | Feasible trials, gradient-mapping stopping |
| Residual/Jacobian available | `levenberg_marquardt_with_options` | Scaled damped QR; nonzero residual can be optimal |
| Convex dense QP | `quadratic::minimize` | Active-set KKT solves; limited flat/unbounded cases |
| Smooth nonlinear constraints | `sqp_with_restoration`, PHR | Local restoration/penalty methods, no infeasibility certificate |

## Gradient and line-search methods

For `f:R^n->R`, a descent direction satisfies `g'd<0`. Armijo accepts
`f(x+alpha*d) <= f(x)+c1*alpha*g'd`, with contraction when rejected.
Strong Wolfe also requires
`|g(x+alpha*d)'d| <= c2*|g(x)'d|`, `0<c1<c2<1`.
The curvature condition is useful for positive quasi-Newton updates. Searches
have finite trial budgets and propagate callback failures; they do not take
unchecked steps after exhaustion.

Steepest descent uses `d=-g`; normalized gradient scales by `||g||`.
Fletcher-Reeves uses `d_k=-g_k+beta*d_(k-1)`,
`beta=||g_k||^2/||g_(k-1)||^2`, with periodic restarts and a descent safeguard.
Gradient/CG methods now allocate only vector-sized workspace.

Newton solves `H*d=-g`. Modified Newton shifts `H` until a positive-definite
factorization exists. These are local smooth methods; an indefinite Newton
direction may not decrease the objective and is safeguarded.

## Quasi-Newton and L-BFGS

For `s=x_(k+1)-x_k`, `y=g_(k+1)-g_k`, inverse updates approximate the secant
condition `B*y=s`. Inverse BFGS uses
`B_new=(I-rho*s*y')*B*(I-rho*y*s')+rho*s*s'`, `rho=1/(s'y)`.
DFP uses `B+s*s'/(s'y)-(B*y)*(B*y)'/(y'B*y)`.
SR1 uses `B+(s-B*y)*(s-B*y)'/((s-B*y)'y)`.
Small/unsafe curvature denominators are skipped. The implemented convex
Broyden family interpolates DFP (`phi=0`) and inverse BFGS (`phi=1`).

L-BFGS stores at most `m` correction pairs and applies the inverse approximation
by two-loop recursion, not by constructing an n-by-n matrix. Accepted
strong-Wolfe value/gradient results are reused. See [SCALABLE_LOCAL](SCALABLE_LOCAL.md)
for exact API, operator requirements, stopping and matrix-free limitations.

## Trust regions

The model is `q(d)=g'd+d'H*d/2` with `||d||<=Delta`.
Global subproblem optimality requires
`(H+lambda*I)d=-g`, `H+lambda*I` positive semidefinite,
`lambda>=0`, and `lambda*(||d||-Delta)=0`.

The dense spectral solver handles indefinite Hessians and the hard case when
the gradient is orthogonal to the minimum eigenspace. Steihaug truncates CG at
the trust boundary or negative curvature. It is an approximate subproblem
solver; with zero gradient it does not discover hidden negative eigenspaces.
Actual/predicted reduction controls acceptance and radius. Rejected steps are
not accepted-progress stagnation.

## Scalar and derivative-free local search

Golden-section and safeguarded parabolic/Brent interpolation assume a continuous
unimodal objective in the supplied scalar bracket. An interval tolerance does
not justify applying these methods as global multimodal solvers.

Bounded pattern search polls positive/negative coordinate directions and
reduces its normalized mesh after an unsuccessful poll.
Nelder-Mead maintains a simplex, orders objective values and uses reflection,
expansion, contraction and shrink steps with coefficients 1, 2, 1/2 and 1/2.
Both project trials to finite bounds and remove fixed coordinates from simplex
initialization. Projected simplices can collapse at a boundary. Mesh/simplex
collapse reports stagnation, not a certified stationary point.

## Least squares and bounded gradients

For residual vector `r` and row-oriented Jacobian `J`,
`f=||r||^2/2`, `g=J'r`. A small `g` can occur with nonzero residual.
Legacy LM uses damped normal equations. The typed adaptive method instead solves
the scaled augmented least-squares problem by rectangular QR, avoiding explicit
`J'J` in its step computation and adjusting damping from model agreement.
See [ROBUSTNESS](ROBUSTNESS.md) and the [certified NIST check](NIST.md).

For a box, the projected-gradient mapping is
`G(x)=x-project(x-g)`. Its norm is appropriate at an active bound even when
`||g||` is nonzero. Projected gradient is not L-BFGS-B; full generalized-Cauchy
and subspace machinery is not implemented.

## QP and nonlinear constraints

Canonical constraints are `h(x)=0`, `g(x)>=0`; Jacobians have one row per
constraint. The Lagrangian is `L=f-mu'h-lambda'g`, `lambda>=0`.
KKT checks include stationarity, feasibility and complementarity.

For QP `f=x'H*x/2+c'x`, an active-set step solves a KKT system for active rows,
adds blocking inequalities and removes rows with incompatible multipliers.
Phase I finds a feasible point when possible. Consistent singular systems have
a checked least-norm fallback; the solver does not certify general LP
unboundedness.

SQP solves quadratic models with linearized constraints. Infeasible
linearizations require an explicit restoration phase; the typed restoration
entry point records these events and retains full constrained checkpoints.
It does not silently substitute PHR for SQP.

PHR uses
`Phi=f-mu'h+rho*||h||^2/2+
(||max(0,lambda-rho*g)||^2-||lambda||^2)/(2*rho)`.
Consistent updates are `mu <- mu-rho*h`,
`lambda <- max(0,lambda-rho*g)`. The projected inequality term must be
differentiated, not treated as a fixed multiplier during the inner solve.
Exact equality methods require the Hessian of the canonical Lagrangian.

## Global methods, reports and comparisons

[GLOBAL](GLOBAL.md) gives DE/PSO update equations and their exact implemented
variants. Targets, budgets and cancellation are distinct from mathematical
optimality; a collapsed population is not a convergence proof.
Fallible callbacks must be finite at evaluated points. Panics are not caught.

Performance comparisons use identical feasible domains and objective targets,
not identical tolerance labels or iteration counts. Record failures and
evaluation costs; compare repeated-seed distributions, not just the best run.
See [BENCHMARKS](BENCHMARKS.md).

## References

- Nocedal and Wright, *Numerical Optimization*, second edition (2006):
  line searches, quasi-Newton, trust regions, QP/SQP and augmented Lagrangians.
- Nocedal (1980), *Updating quasi-Newton matrices with limited storage*;
  Liu and Nocedal (1989), *On the limited memory BFGS method for large scale optimization*.
- Steihaug (1983), *The conjugate gradient method and trust regions in large scale optimization*.
- Levenberg (1944), *A method for the solution of certain non-linear problems in least squares*;
  Marquardt (1963), *An algorithm for least-squares estimation of nonlinear parameters*.
- Nelder and Mead (1965), *A simplex method for function minimization*;
  Brent, *Algorithms for Minimization without Derivatives* (1973).
- Powell (1969), *A method for nonlinear constraints in minimization problems*;
  Hestenes (1969), *Multiplier and gradient methods*.
- [NLopt algorithm/reference citations](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/)
  and [COCO methodology](https://arxiv.org/abs/1603.08785).
