# nlopt4r improvement and release plan

Assessment date: 2026-11-25. The design below is the original plan; the status
checklist records what has actually been implemented and validated.
Preserve the pure-Rust default library and its borrowed callback interfaces.
Do not attempt to duplicate every NLopt algorithm before a first release.

## Implementation status

- [ ] M0 release gate: metadata/MSRV/dry-run/consumer checks passed; **blocked**
  on maintainer-approved license and source redistribution confirmation.
- [ ] M1 foundation and baseline:
  - [x] Finite/fixed bounds, scaling, projection and seeded sampling.
  - [x] Public checked evaluation context, strict budgets, target/cancellation,
    fallible observers and value-only recovery.
  - [x] External bounded pattern-search application using public utilities.
  - [x] Native comparisons isolated in unpublished tooling; nine tests preserved.
  - [ ] Compatibility local namespace, benchmark smoke runner and baseline.
- [ ] M2 global methods:
  - [x] Synchronous DE/rand/1/bin and inertia-weight global-best PSO.
  - [x] Budget, bounds/fixed-coordinate, seed, crossover/donor, synchronous PSO,
    observer/cancellation and error-recovery conformance tests.
  - [x] Typed reports, documented defaults, theory/references and runnable example.
  - [ ] Measured 30-seed benchmark report.
- [ ] M3 scalable local algorithms and measured complexity.
- [ ] M4 numerical robustness and bounded local optimization.
- [ ] M5 broad problem catalog, standardized adapters, theory, CI and coverage.

Unchecked milestones are not complete; optional future algorithms and external
licensed datasets are not implicitly completed by another solver.

## 1. Assessment against NLopt

The current package has dense gradient/quasi-Newton/Newton methods, scalar
searches, spectral trust regions, least squares, QP, SQP and augmented Lagrangian
methods. Shared stopping criteria, strict objective budgets, callback errors and
recoverable checkpoints are already implemented. There are 62 integration tests
(including nine native-reference tests) and one doctest.

Evidence: [public API](../src/lib.rs), [local solvers](../src/unconstrained.rs),
[accounting](../src/stopping.rs), [reference comparisons](../tests/nlopt/main.rs)
and [interface tests](../tests/interfaces.rs).

| Capability | Upstream NLopt | nlopt4r today / recommended direction |
| --- | --- | --- |
| Large smooth problems | L-BFGS, truncated Newton | Dense methods only; add limited-memory BFGS and Hessian-vector products |
| Local derivative-free | COBYLA, BOBYQA, Nelder-Mead, Sbplx, others | Scalar only; start with pattern search and Nelder-Mead, then model-based methods |
| Global / population | DIRECT, CRS, MLSL, ISRES, ESCH, others | None; add DE and PSO, then multistart and optional local polishing |
| Bounds | First-class bounds with method-specific support | No common multidimensional bounds API; introduce validated bounds |
| Nonlinear constraints | SLSQP, MMA/CCSA; AUGLAG composition | SQP/PHR exist; improve scaling and elastic restoration before adding MMA |
| Extension | Callback configuration and subsidiary solvers | Useful primitives exist, but accounting/validation/update helpers are mostly private |
| Benchmarking | Target-value comparison guidance | Current tests establish agreement, not comparative speed or robustness |

PSO and DE are additions motivated by this request, not algorithms to copy from
NLopt: its documented catalog does not expose standard PSO or DE solvers.
Reimplement mathematics independently; do not import upstream C/Fortran
implementations into this pure-Rust package.

## 2. API and module design

### Proposed destination, reached incrementally

```text
src/
  lib.rs                       small facade, common exports and quickstart
  core/                        evaluation, budgets, reports, errors, observers
  problem/                     bounds, constraints, residual/derivative adapters
  local/                       gradient, quasi_newton, newton, derivative_free
  global/                      differential_evolution, particle_swarm, multistart
  constrained/                 sqp, augmented_lagrangian
  least_squares/               lm, robust residual adapters
  quadratic/                   active_set, equality
  utils/                       documented, reusable numerical solver primitives
benches/                       microbenchmarks, pure-Rust only
tools/bench/                   unpublished experiment runner and problem registry
tools/nlopt-reference/         unpublished native reference/comparison crate
docs/                          theory, solver guide, extension and benchmark guide
```

Keep one publishable solver crate initially. Do not split every module into a
crate, introduce a plugin registry, or require a universal solver object.
Keep small function entry points, generic borrowed callbacks and trait-object
support. Add typed `LbfgsOptions`, `DeOptions`, `PsoOptions`, etc.; do not overload
the current local `Method` enum with incompatible global settings.

1. Extract internal implementation modules with compatibility reexports at old
   paths. Move behavior only after conformance tests exist.
2. Separate common budget/target/cancellation controls from gradient/KKT,
   interval and population diagnostics. Keep `StopOptions` as a compatibility
   facade for existing solvers; global solvers must not require a fictitious
   gradient tolerance. Introduce builders before any breaking struct changes.
3. Add validated `Bounds`: dimension checks, finite ordered bounds for population
   methods, fixed-coordinate support, containment, projection and scaling.
   Local methods may allow infinite bounds only when they actually support them.
4. Keep `Evaluator` as the callback compatibility boundary. Add value-only
   convenience constructors; never request gradients in DE/PSO. Add optional
   derivative-checking/finite-difference adapters with explicit step sizes and
   correct accounting of every objective call.
5. Add opt-in observer/cancellation hooks. Distinguish target reached,
   cancellation, budget exhaustion and stagnation. A known benchmark target
   reached is not a mathematical global-optimality certificate.
6. Global reports need best point/value, seed, generations, population diversity
   and evaluation counts. Use optional/typed optimality diagnostics rather than
   inventing a gradient or inserting NaN into normal reports.
7. Preserve constraint conventions `h=0`, `g>=0`, Jacobian rows and
   `L=f-mu'h-lambda'g`. Preserve strict budgets and fully evaluated checkpoints.
   Future typed recoveries should retain constrained multipliers as well.

Do not expose the current private generic `Run<Report>` implementation wholesale.
Offer a small documented evaluation context that external algorithms can use to
validate callbacks, charge budgets, store valid checkpoints and emit observations.
Support concrete buffers and explicit ownership; avoid mandatory allocation or
dynamic dispatch at every numerical operation.

### Utilities for adding algorithms

Stabilize only utilities exercised by at least two solvers:

- Checked value/gradient/residual evaluation into caller-owned buffers.
- Bounds handling, coordinate scaling and seeded sampling.
- Armijo and strong-Wolfe line searches with accepted-point caches and explicit
  trial limits; retain the existing standalone Armijo function.
- Guarded BFGS/SR1 updates, L-BFGS two-loop recursion and curvature-pair storage.
- Checked dense linear solves, QR/SVD least-squares fallbacks and truncated CG.
- Shared termination/checkpoint accounting and fallible observers.

Provide a complete external-crate extension example: implement bounded pattern
search using the public context and bounds utilities, register its adapter in the
benchmark runner, and reuse conformance tests. An optional small solver trait may
support that runner; existing function-based solvers must not depend on it.

For a 0.2 API change, publish a migration table and preserve old function/path
wrappers where practical. Do not claim source compatibility for changed public
struct literals or exhaustive enum matches.

## 3. Algorithm priorities

| Priority | Work | Prerequisites and measurable acceptance |
| --- | --- | --- |
| P0 | Bounds, evaluation context, benchmark baseline | External algorithm example; budgets/errors match current solvers |
| P1 | DE/rand/1/bin and inertia-weight PSO | Shared seeded sampling/bounds; reproducibility and bounded, derivative-free callbacks |
| P1 | Strong-Wolfe search and L-BFGS | O(m*n) history storage; curvature tests; Rosenbrock and ill-conditioned quadratics |
| P1 | Pattern search / Nelder-Mead | Derivative-free conformance; degenerate simplex and boundary regressions |
| P2 | Newton-CG / Steihaug trust region | Hessian-vector callbacks and preconditioning; negative-curvature handling |
| P2 | Projected gradient, then L-BFGS-B | Projected-gradient stopping; correct active bounds, not clamped unconstrained BFGS |
| P2 | Adaptive LM with QR-based steps | Avoid relying solely on normal equations; rank-deficient/scaled NIST problems |
| P2 | SQP restoration, scaling, merit/filter work | Infeasible linearizations; feasibility/optimality verified independently |
| P3 | CMA-ES, DIRECT, BOBYQA/COBYLA, MMA/CCSA | Separate mathematical design/review, problem demand and benchmark evidence |

Retain classical methods for teaching and regression; identify recommended
defaults in a solver-selection guide. Sparse solvers, AD integration, GPU,
mixed-integer and multiobjective methods are later projects, not first-release
requirements.

### DE first implementation

For target `x_i`, sample distinct donors `a,b,c`, all different from `i`:

```text
v = x_a + F * (x_b - x_c)
u_j = v_j if uniform(0,1) < CR or j == j_rand; otherwise x_i,j
next_i = u if f(u) <= f(x_i); otherwise x_i
```

Require population size >= 4, finite `0 < F <= 2` and `0 <= CR <= 1`.
`j_rand` guarantees at least one donor coordinate. Specify generational updates:
all donors read the previous generation, not an accidentally order-dependent mix.
Initially use deterministic midpoint repair between the violated bound and the
target coordinate, with a final containment check. Test fixed coordinates and
overflow in arithmetic explicitly.

Default initialization is seeded uniform sampling in finite bounds, with an
optional caller-supplied population validated before evaluations. Use contiguous
population storage and reuse candidate buffers. Return the best evaluated point
if the budget ends during initialization or a partial generation; do not discard
an evaluated improvement. Adaptive `F/CR` and additional strategies come later.

### PSO first implementation

Use explicitly documented inertia-weight global-best PSO:

```text
v_i = w*v_i + c1*r1*(pbest_i-x_i) + c2*r2*(gbest-x_i)
x_i = x_i + v_i
```

Draw independent coordinate-wise `r1,r2`; snapshot `gbest` for a synchronous
generation. Validate finite coefficients (`0 <= w < 1`, nonnegative `c1,c2`),
population size >= 2 and finite bounds. Treat these as a supported configuration
range, not a convergence theorem. Start with zero velocities; make any velocity
limit an explicit fraction of each coordinate's range. Project violated
coordinates to bounds and zero their velocity; keep fixed coordinates fixed.

Specify tie handling and when personal/global bests update. Retain best-ever
evaluated positions, including partial generations. Add local-best topology and
constriction variants only as separately documented strategies.

### Shared global-search contract

- Explicit seeded RNG (for example a pinned ChaCha implementation); never global
  mutable randomness. Record RNG identity/version. Do not promise bitwise
  cross-platform reproducibility from a seed alone.
- Finite bounds only in the first release. Equality/inequality constraints are
  not automatically supported by clamping or arbitrary penalties.
- Mandatory finite evaluation budget; optional target and stagnation controls.
  Population collapse is stagnation, not proof of convergence.
- Sequential execution first. Optional parallel batches later must reserve
  budget before launch, have reproducible result ordering, propagate callback
  errors and require `Sync` only in the parallel API.
- Optional local polishing must share the remaining budget and respect bounds;
  do not run today's unconstrained BFGS outside a bounded objective's domain.

## 4. Performance work: measure before changing

Code inspection suggests opportunities, not measured speedups:

| Evidence | Proposed change | Validation |
| --- | --- | --- |
| Local minimizer allocates an identity inverse matrix even for gradient/CG methods | Allocate method-specific workspace only | Gradient/CG scratch storage scales O(n), not O(n^2) |
| Accepted Armijo value is followed by combined reevaluation | Optional accepted-value/gradient cache contract | Fewer expensive calls without changing callback semantics or budgets |
| `Run::record` clones reports and point baselines repeatedly | Reuse storage; separate report construction from checkpoint update | Allocation counts and unchanged recovery/stagnation behavior |
| Trust solver reevaluates unchanged rejected iterates and uses eigendecomposition | Cache deterministic accepted state; offer iterative subproblems | Equal targets/KKT tests, lower calls and large-n storage |
| LM forms J'J and uses general LU | QR-based damped least squares, adaptive damping | Conditioning/rank-deficiency tests, not timing alone |
| QP reconstructs working KKT systems and may use SVD | Reuse assembly/storage; later factorization updates | Preserve residual checks and singular-system fallbacks |

Caching assumes deterministic callbacks. Make that contract explicit or opt-in;
never silently change behavior of stateful/noisy evaluations.

Measure time, attempted calls, allocation counts, peak workspace and achieved
accuracy separately. Include cheap analytic and deliberately expensive synthetic
callbacks. Set regression thresholds only after repeated baseline measurements;
do not invent a universal speedup percentage. L-BFGS memory must be O(m*n),
population memory O(population*n), and local gradient/CG memory O(n).

## 5. Problem catalog, tests and benchmark workflow

### Versioned catalog

Keep the solver crate small. Put broad experiment datasets and native adapters
in unpublished tooling crates; share small analytic fixtures with tests.
Each problem carries ID, version/source, dimension rules, domain/bounds, start
sets, optimum or clearly labeled best-known value, derivative capabilities,
constraint convention and licensing/provenance.

| Group | Initial content |
| --- | --- |
| Smooth/local | Sphere, rotated ellipsoid, Rosenbrock, Powell singular, Wood, Beale, Brown badly scaled, extended Freudenstein-Roth |
| Global | Ackley, Rastrigin, Griewank, Schwefel, Levy, Zakharov, Styblinski-Tang, Alpine; 2-D Himmelblau, Branin, six-hump camel, Goldstein-Price; Hartmann 3/6 |
| Least squares | Current fixtures, exponential/rational fits, NIST Misra/Chwirut/MGH/BoxBOD cases with certified values |
| QP/constrained | Existing MATLAB cases, generated SPD/PSD QPs with known KKT pairs, redundant constraints, simplex/portfolio and engineering-design examples |
| Modern applications | Stable logistic regression, regularized nonlinear fitting, robust residual models, small optimal-control discretizations |
| Standardized black-box | COCO/BBOB's 24 noiseless base functions and versioned instances; later licensed CEC suites and optional CUTEst adapters |

First implement at least 30 independently verified analytic problems across these
groups. Then integrate exact COCO/BBOB externally and/or independently implement
its specified transformations with reference-value validation. Until validated,
call variants "BBOB-inspired", not COCO-conformant. CEC/CUTEst datasets and native
reference code need their own license review; no downloads during unit tests.

Use dimensions 2, 5, 10, 20, 50 and 100 where defined. Smooth large-scale tests
also use 1,000 and 10,000 variables; do not run dense Newton/QP at those sizes by
default. Add shifts, rotations, variable/objective scaling, multiple starts and
controlled noise as separate identified instances. Do not mislabel classical
test functions as a modern benchmark suite merely by collecting their names.

### Three complementary test layers

1. Fast deterministic CI: preserve all existing tests; validate catalog formulas,
   derivatives by directional differences, analytic optima and KKT residuals.
   Cover budgets on initialization/trials/partial generations, invalid options,
   callback errors, accepted checkpoints, ties, fixed bounds and degeneration.
   Test DE distinct donors/forced crossover and PSO synchronous updates directly.
2. Property tests: seeded SPD QPs with constructed solutions, consistent redundant
   constraints, scaling/permutation invariants where applicable, monotone global
   best value and bound containment for every evaluated point.
3. Scheduled experiments: multi-start/multi-seed success rates, difficult noisy
   functions, high dimensions and native NLopt comparisons. Use independent
   analytic or certified checks; NLopt agreement alone is not an oracle.

Add coverage tooling (for example cargo-llvm-cov) only when implementing coverage
measurement. Publish branch/function gaps and prioritize numerical/error paths;
do not set a percentage target before a measured baseline.
Keep stochastic CI on fixed small cases/seeds; evaluate statistical robustness
with 30 seeds offline rather than flaky random "always converges" tests.

### Benchmark commands to implement

Proposed interface, not runnable commands today:

```sh
cargo bench --bench kernels
cargo run --release -p nlopt4r-bench -- --suite smoke --algorithms bfgs,lbfgs,de,pso
cargo run --release -p nlopt4r-bench -- --suite global --seeds 0..30 --budget-per-dimension 10000 --output results.jsonl
cargo run --release -p nlopt4r-reference -- --suite smooth --algorithms nlopt-lbfgs,nlopt4r-lbfgs
```

Use Criterion for kernels and a separate repeated-run harness for solver
experiments, not Criterion's adaptive timing loop as a global-search comparison.
Reset problem state and RNG per run. Provide filters, a smoke suite, machine-
readable JSONL/CSV, an offline summary tool and one registry entry per new
algorithm/problem; adding either must not require editing every benchmark.

Record package/commit, compiler, OS/CPU, options, seeds, instance identity,
termination reason, counters, elapsed time, feasibility and target-hit history.
Use common objective targets (`f_star + delta`), normalized objective gaps when
needed and identical feasible domains/budgets. Report targets such as 1e-2, 1e-5
and 1e-8, applicable KKT/feasibility tolerances, median/quantile time-to-target,
success rate and empirical cumulative distributions. Label unknown best values.

Expected running time in evaluations is total evaluations consumed by all runs,
including unsuccessful runs, divided by successful runs; report infinity when
there are none. Never hide failures or compare only successful runs.
Control adapter/callback overhead, count actual user evaluations on both sides,
record native versions and thread settings, and do not equate iteration counts
or different libraries' tolerance semantics.

## 6. Theory and user documentation

Add a solver-selection table: smoothness/derivatives, bounds/constraints, scale,
memory cost, deterministic/stochastic behavior and recommended alternatives.
Keep a short README; add runnable examples and API doc examples for borrowed
callbacks, constrained problems, DE/PSO, errors/restarts and custom algorithms.

Each algorithm chapter must give its mathematical problem, update equations,
assumptions, stopping meaning, complexity, safeguards, practical limitations,
implementation departures and primary references. Distinguish first-order KKT
stationarity from a local minimum and from certified global optimality.

Theory coverage:

- Armijo sufficient decrease and strong-Wolfe curvature; safeguarded Brent
  interpolation assumes a unimodal scalar bracket.
- Newton regularization, conjugate-gradient restarts and BFGS/DFP/SR1 secant
  updates; L-BFGS two-loop recursion and curvature-pair safeguards.
- Trust-region KKT system `(B+lambda*I)d=-g`, PSD condition, boundary
  complementarity and the spectral hard case; iterative negative curvature.
- Least squares `f=||r||^2/2`, `g=J'r`, damped steps and why small gradient does
  not imply zero residual.
- QP/SQP KKT systems, multiplier signs, working sets and restoration limits.
- PHR's projected inequality term and consistent multiplier/penalty updates.
  With the canonical signs, describe
  `Phi_rho = f - mu'h + rho*||h||^2/2 +
  (||max(0,lambda-rho*g)||^2-||lambda||^2)/(2*rho)`,
  then `mu <- mu-rho*h`, `lambda <- max(0,lambda-rho*g)`.
- DE mutation/crossover/selection and PSO memory/velocity dynamics as above;
  finite-budget search success does not establish a global convergence theorem.

Build examples and doctests in CI. Review references and provenance before
redistribution; cite papers, do not reproduce their copyrighted text or figures.

## 7. Delivery sequence and acceptance gates

| Milestone | Depends on | Deliverables / exit gate |
| --- | --- | --- |
| M0: release audit | Nothing | Package inventory, approved license/provenance, metadata, validated MSRV, dry-run and downstream build |
| M1: foundation and baseline | Current behavior tests | Compatibility modules, bounds/context, external algorithm example, smoke runner and measured baseline |
| M2: global methods | M1 | DE and PSO, focused/property tests, documented defaults and 30-seed benchmark report |
| M3: scalable local methods | M1 | Wolfe/L-BFGS and iterative Newton/trust methods; verified complexity and equal-target comparisons |
| M4: numerical robustness | M1; M3 where shared utilities apply | Adaptive QR-LM, bounded local methods, SQP restoration and scaled regressions |
| M5: broader validation/docs | Starts at M1; release requires relevant M2-M4 gates | Verified catalog, standardized adapters, theoretical chapters, migration and release checklist |

M2 and M3 can be independent after M1; do not make PSO wait for every local
algorithm. Documentation and targeted tests accompany every implementation
commit. Commit each coherent change; publish benchmark evidence before claiming
performance improvements. Push only when repository authentication is resolved.

## 8. Publication preparation (no upload authorized)

The current manifest retains the package name/version and pure-Rust runtime
dependency, and now supplies repository, documentation, keywords and categories.
Do not rename the repository remote or fabricate a license.

**Release blockers:** no license declaration/file and unconfirmed redistribution
rights for the MATLAB-derived work. The maintainer must select the license and
confirm those rights; local compilation cannot establish them.

The manifest now declares Rust 1.88 after a successful locked library check with
Rust 1.88.0. The locked nalgebra 0.34.2 declares Rust 1.87, while nlopt4r uses
let-chains (Rust 1.88+). Dependency MSRV alone is not package MSRV.
Rust 1.88.0 was installed for validation without changing the default toolchain.
Keep minimum-compiler testing in CI and review dependency updates against it.

Default library consumers and default repository tests now build pure Rust.
Native NLopt is isolated in `tools/nlopt-reference`; invoke
`cargo test -p nlopt4r-reference` for the preserved nine comparisons.
Keep an explicit native-reference CI job.

Safe validation commands from the package directory:

```sh
cargo test --locked
cargo +1.88.0 test --locked
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps
cargo package --locked --list
cargo publish --dry-run --locked --registry crates-io
```

Inspect the archive for source/examples/tests, unintended datasets and secrets;
check compressed size. Compile/test the extracted package, and compile a small
external path-dependent application against that extracted source to catch
repository-only assumptions. Native references must remain dev-only.
Do not read credential files, run login again or issue a real publish command.

A successful dry-run verifies local packaging/builds, not license rights,
registry name ownership/availability, token permissions or final server
acceptance. A license warning means not release-ready even if Cargo exits zero.
No release tag or actual upload should happen until these gates are satisfied.

### Validation performed in this planning pass

- `cargo publish --dry-run --locked --registry crates-io` succeeded, verified
  the extracted library and explicitly aborted upload. The initial inventory
  contained 26 files and no unrelated datasets/build output/credential files.
- All 62 integration tests and one doctest passed against the extracted source,
  including all nine native NLopt comparison tests.
- Formatting, all-target Clippy with warnings denied, warning-free API
  documentation and local Markdown link checks passed.
- An independent application built and ran against the extracted package using
  a borrowed buffered callback, public solver options and strict evaluation
  budget, and verified its analytic optimum.
- A locked, offline library check and all 63 tests passed on Rust 1.88.0;
  this is compiler compatibility evidence, not a proof that arbitrary future
  dependency versions will preserve the MSRV.

Repeat the dry-run on the final committed source whenever metadata or package
contents change. The missing-license warning is deliberately not suppressed.

## 9. References

- [NLopt algorithm catalog](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/):
  algorithm capabilities, primary citations and equal-target comparison advice.
- [NLopt reference](https://nlopt.readthedocs.io/en/latest/NLopt_Reference/):
  bounds, termination and subsidiary solver configuration.
- Kennedy and Eberhart (1995), *Particle swarm optimization*,
  [doi:10.1109/ICNN.1995.488968](https://doi.org/10.1109/ICNN.1995.488968).
- Storn and Price (1997), *Differential Evolution - A Simple and Efficient
  Heuristic for Global Optimization over Continuous Spaces*,
  [doi:10.1023/A:1008202821328](https://doi.org/10.1023/A:1008202821328).
- Nocedal (1980), *Updating quasi-Newton matrices with limited storage*, and
  Liu and Nocedal (1989), *On the limited memory BFGS method for large scale
  optimization*: cited in the NLopt L-BFGS section.
- Dembo and Steihaug (1983), *Truncated Newton algorithms for large-scale
  optimization*, [doi:10.1007/BF02592055](https://doi.org/10.1007/BF02592055).
- Nocedal and Wright (2006), *Numerical Optimization*, second edition:
  line searches, quasi-Newton/trust-region, QP/SQP and augmented Lagrangian theory.
- Hansen et al., *COCO: A Platform for Comparing Continuous Optimizers in a
  Black-Box Setting*, [arXiv:1603.08785](https://arxiv.org/abs/1603.08785),
  revised 2020: problem instances, targets and evaluation-based runtime.
- [NIST nonlinear regression reference datasets](https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml):
  certified fitting values across difficulty levels.
- [Cargo publishing guide](https://doc.rust-lang.org/cargo/reference/publishing.html)
  and [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).
