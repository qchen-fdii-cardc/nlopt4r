# Reproducible benchmark baseline

`tools/bench` is the unpublished `nlopt4r-bench` executable. Serde is a tooling
dependency, not a library dependency. Native NLopt comparisons live separately
in `tools/nlopt-reference`; they are not silently substituted for Rust solvers.
The parent workspace includes this crate and configures Criterion's
`benches/kernels.rs` target.

```sh
CARGO_BUILD_JOBS=2 cargo test -p nlopt4r-bench
CARGO_BUILD_JOBS=2 cargo run --release -p nlopt4r-bench --features extended -- \
  --suite smoke --algorithms bfgs,lbfgs,de,pso --seeds 0..30 \
  --budget-per-dimension 1000 --dimension 4 --output smoke.jsonl
cargo run -p nlopt4r-bench -- --summarize smoke.jsonl
CARGO_BUILD_JOBS=2 cargo bench --bench kernels
```

`0..30` means seeds 0 through 29. Outputs ending in `.jsonl` or `.csv` are
supported; existing files are never overwritten. CSV contains convenient
scalar columns and a quoted, complete `record_json` column, preserving histories
and metadata losslessly. Offline summarization accepts either format.
Unknown options, algorithms, malformed records, unavailable algorithms, write
errors and solver errors are explicit. All completed runs are retained, and
solver errors make the run command exit unsuccessfully. A budget stop or a
missed target is a benchmark failure, not an executable error.

The default `extended` feature enables the integrated `local::lbfgs`,
`global::differential_evolution`, `global::particle_swarm` and bounded
`derivative_free::{pattern_search, nelder_mead}` exports.
`--no-default-features` builds only BFGS; requesting any unavailable algorithm
is an error, not a skip. The explicit algorithm registry is
`bfgs,lbfgs,de,pso,pattern-search,nelder-mead,external-pattern`.
The default `external-pattern` feature uses the public `tools/extension`
library. It calls that independent example directly through
the same measured evaluator, rather than aliasing the built-in pattern search.
The external example is deterministic and does not take a seed: repeated
seed labels are timing repetitions of its same search, not stochastic trials.
Built-in pattern search and Nelder–Mead use the run's seeded initial point.
An opt-in BFGS/external-only build can use
`--no-default-features --features external-pattern`.

## Protocol and limitations

* `smoke`: sphere, Rosenbrock, Booth, Rastrigin, linear fit, coupled QP.
  `smooth`: analytic, non-global, unconstrained catalog entries.
  `global`: the multimodal/nonsmooth group below.
* Variable dimensions are 1..128 subject to each formula's minimum; fixed-size
  problems retain their declared dimensions regardless of `--dimension`.
* Every run constructs a new measured evaluator, starts a new timer, resets all
  accounting/history, initializes local iterates from a run-local SplitMix64
  stream, and supplies the seed to the population algorithm. No cross-run
  objective cache, optimizer state or RNG state is reused.
* The objective-call budget is `budget_per_dimension * actual_dimension`.
  BFGS/L-BFGS analytic gradients use exact forward chain-rule arithmetic
  (not finite differences); each gradient request is recorded separately.
  The catalog uses one scalar forward pass per coordinate; its derivative
  computation overhead is included in time and is not an allocation-optimized
  application evaluator. Criterion uses a separate buffered sphere evaluator.
  Ackley (nondifferentiable at zero), Alpine and Schwefel-222 use explicit
  central differences for local solvers. Every base/probe call is charged
  to the same strict budget; exhausted partial derivatives return an error
  checkpoint rather than receiving free evaluations.
* Bound-feasible best-ever objective improvement histories record objective
  evaluation number and elapsed wall time. Derivative probes are evaluations
  too. Targets require both box feasibility and constraint violation <=1e-8.
  Histories are compressed improvements, not accepted-iterate trajectories.
* Target = proved reference value + `1e-6 * (1 + abs(reference value))`.
  Success is a recorded feasible target hit, not a solver's convergence flag.
  Local solvers can continue after a hit; the first hit determines ERT.
* BFGS/L-BFGS are unbounded solvers. They may evaluate outside the declared
  domain, but those evaluations count and cannot certify a benchmark hit.
  Population solvers enforce boxes. General constrained entries are catalog
  verification fixtures, **excluded from these unconstrained run suites**:
  no penalty surrogate or implicit constraint handling is claimed.
* ERT = sum(first-hit evaluations for successes + **actual consumed
  evaluations for failures**) / number of successes. Time ERT uses the
  corresponding hit/full-run times. Zero successes gives null (infinite ERT).
  Early-stopped failures retain their actual cost, not an invented full budget.
  Summaries separate different budgets, targets and complete recorded
  environments (CPU, OS, architecture, rustc, revision, build profile and
  package version). Environment metadata accompanies each summary group.
  Matching metadata is not a claim of timing fairness: affinity, frequency,
  background load and compiler flags are not controlled, and `unavailable`
  metadata does not identify a machine uniquely.
* Records contain seed, dimensions, budget, derivative mode, termination/error,
  best point/value, history and OS/architecture/CPU/rustc/profile/revision.
  Missing optional CPU/revision metadata is explicitly `unavailable`.
  Wall times are observations, not portable performance claims.

## Public catalog interface for other runners

All catalog types are under `nlopt4r_bench::catalog`:

```rust,ignore
pub static CATALOG: &[Problem];
pub struct Problem {
    pub id: &'static str,
    pub group: Group,
    pub fixed_dimension: Option<usize>,
    pub min_dimension: usize,
    pub bounds: (f64, f64),
    pub analytic: bool,
}
impl Problem {
    pub fn dimension(&self, requested: usize) -> Result<usize, String>;
    pub fn reference(&self, n: usize) -> (Vec<f64>, f64);
    pub fn value(&self, x: &[f64]) -> f64;
    pub fn gradient(&self, x: &[f64]) -> Vec<f64>;
    pub fn violation(&self, x: &[f64]) -> f64;
}
```

`Group` variants are `Smooth`, `Global`, `LeastSquares`, `Constrained`, `Qp`,
`Application`. Call `dimension` before constructing points or references; the
value/gradient methods assume a valid dimensional input. Scalar bounds repeat
for each coordinate. `analytic=false` requires explicitly counted numerical
derivatives for local solver comparisons; do not call the chain-rule helper at
nondifferentiable points and label that a valid analytic derivative.

## Verified objective catalog (35 distinct objectives)

Indices start at 1; `n` is the actual dimension. Unless noted, the reference
is x=0, f=0. Formula descriptions are the primary specification. The classical
definitions follow the [SFU optimization test-function collection](https://www.sfu.ca/~ssurjano/optimization.html)
and Powell's singular least-squares function (M. J. D. Powell, *An iterative
method for finding stationary values of a function of several variables*,
Computer Journal 5, 1962, pp. 147–151). Constructed objectives below are
explicitly identified, with algebraic proofs rather than external optimum
labels. No numerical search result is labeled a known optimum.

| Objective / group | Dimensions; finite box | Formula and independently justified reference |
|---|---|---|
| sphere / smooth | n>=1; [-5,5] | sum x_i² >=0 |
| ellipsoid / smooth | n>=2; [-5,5] | sum 10^(6(i-1)/(n-1)) x_i² >=0 |
| Rosenbrock / smooth | n>=2; [-3,3] | sum [100(x_(i+1)-x_i²)²+(x_i-1)²]; x=1 gives zero |
| Zakharov / smooth | n>=1; [-5,5] | sum x_i² + z²+z⁴, z=sum i x_i/2; nonnegative |
| Dixon–Price / smooth | n>=2; [-3,3] | (x_1-1)²+sum_(i=2)^n i(2x_i²-x_(i-1))²; x_i=2^(-(1-2^(1-i))) makes every residual zero |
| quartic / smooth | n>=1; [-3,3] | sum i x_i⁴; deterministic, no noise |
| different powers / smooth | n>=1; [-2,2] | sum abs(x_i)^(i+1) >=0 |
| cosh bowl / smooth, constructed | n>=1; [-3,3] | sum(cosh x_i-1)>=0 |
| logcosh / smooth, constructed | n>=1; [-3,3] | sum log(cosh x_i)>=0 |
| Trid / smooth | n>=2; [-10000,10000] | sum(x_i-1)²-sum x_i x_(i+1); positive-definite tridiagonal Hessian, unique stationary x_i=i(n+1-i), value -n(n+4)(n-1)/6 |
| Booth / smooth | 2; [-10,10] | (x+2y-7)²+(2x+y-5)²; (1,3), zero |
| Matyas / smooth | 2; [-10,10] | .26(x²+y²)-.48xy = .01(x+y)²+.25(x-y)² |
| Beale / LS | 2; [-4.5,4.5] | sum_(k=1)^3 (c_k-x+xy^k)², c=(1.5,2.25,2.625); (3,.5), zero |
| Himmelblau / global | 2; [-6,6] | (x²+y-11)²+(x+y²-7)²; (3,2), zero; no uniqueness claim |
| three-hump camel / smooth | 2; [-5,5] | 2x²-1.05x⁴+x⁶/6+xy+y² = (y+x/2)²+x²(1.75-1.05x²+x⁴/6); quadratic in x² has negative discriminant, hence nonnegative |
| Powell singular / LS | 4; [-4,5] | (x1+10x2)²+5(x3-x4)²+(x2-2x3)⁴+10(x1-x4)⁴ >=0 |
| Rastrigin / global | n>=1; [-5.12,5.12] | sum[x_i²+10(1-cos(2pi x_i))]>=0 |
| Ackley / global | n>=1; [-5,5] | -20exp(-.2 sqrt(sum x_i²/n))-exp(sum cos(2pi x_i)/n)+20+e >=0 since exponentials <=1 and <=e |
| Griewank / global | n>=1; [-100,100] | sum x_i²/4000-prod cos(x_i/sqrt(i))+1 >=0 |
| Levy / global | n>=2; [-10,10] | w_i=1+(x_i-1)/4; sin²(pi w1)+sum_(i<n)(w_i-1)²[1+10sin²(pi w_i+1)]+(w_n-1)²[1+sin²(2pi w_n)]; x=1, zero |
| Easom / global | 2; [-10,10] | -cos x cos y exp(-(x-pi)²-(y-pi)²)>=-1; (pi,pi) gives -1 |
| Qing / global | n>=1; [-12,12] | sum(x_i²-i)²; x_i=sqrt(i), zero |
| Brown / smooth | n>=2; [-2,2] | sum[(x_i²)^(x_(i+1)²+1)+(x_(i+1)²)^(x_i²+1)]>=0, continuous zero extension |
| Alpine / global | n>=1; [-10,10] | sum abs(x_i sin x_i+.1x_i)>=0 |
| Schwefel-222 / global | n>=1; [-10,10] | sum abs x_i+prod abs x_i>=0 |
| linear fit / LS, constructed | 2; [-5,5] | sum_(t=0,.25,...,1)[a+bt-(1+2t)]²; (1,2), zero |
| exponential fit / LS, constructed | 2; [-2,2] | sum_(same t)[a exp(bt)-exp(.5t)]²; (1,.5), zero |
| logistic fit / LS, constructed | 2; [-3,3] | t=-1,-.5,...,1; sum[sigmoid(a+bt)-sigmoid(.5+t)]²; (.5,1), zero |
| Gaussian fit / LS, constructed | 2; [.1,3] | same t; sum[a exp(-t²/(2b²))-exp(-t²/(2*1.5²))]²; (1,1.5), zero |
| ridge regression / application, constructed | n>=2; [-3,3] | zero-response design with adjacent rows (1,2): sum(x_i+2x_(i+1))²+.1 sum x_i²; ridge makes minimizer unique |
| Poisson likelihood / application, constructed | n>=1; [-3,3] | sum i(exp x_i-x_i-1); exposure/count=i, log-rate x_i; exp(t)>=1+t proves zero optimum |
| spring chain / application, constructed | n>=2; [-3,3] | x1²+sum i(x_(i+1)-x_i)²; pinned chain, positive stiffnesses; unique zero equilibrium |
| coupled QP / QP, constructed | n>=2; [-3,3] | sum x_i²+sum(x_i+x_(i+1))²; SPD quadratic, unique zero optimum |
| simplex QP / constrained, constructed | n>=2; [0,1] | sum(x_i-1/n)², equality sum x_i=1; uniform point feasible, zero lower bound |
| disk linear / constrained, constructed | 2; [-1,1] | -x-y, x²+y²<=1; Cauchy–Schwarz lower bound -sqrt(2), attained at (1/sqrt(2),1/sqrt(2)) |

Tests check every independently specified reference value and feasible point,
catalog uniqueness and domain/dimension restrictions, independent nonoptimal
values, directional finite differences against exact chain-rule derivatives
(at differentiable points for nonsmooth objectives), counted fallback
derivative accuracy and strict budgets,
output round trips and failure-inclusive ERT. Reference identity tests are
not claims that an algorithm will find the reference.

## Baseline status

Measured results are recorded separately; the harness is not itself evidence of
a speedup. Constrained solver comparisons, statistical confidence intervals and
controlled hardware repeat measurements remain follow-up work.

## Native equal-target L-BFGS experiment

```sh
cargo run --release --locked -p nlopt4r-reference -- \
  --suite smooth --algorithms nlopt-lbfgs,nlopt4r-lbfgs \
  --seeds 0..30 --dimension 2 --budget-per-dimension 2000 \
  --output target/native-30-seed.jsonl
```

This separate runner uses sphere and Rosenbrock with identical ChaCha8-seeded
initial points, an **unbounded domain for both solvers**, history size 10,
an objective target of reference + 1e-6 and the same objective budget.
Native NLopt stops at the target; Rust L-BFGS uses gradient tolerance 1e-8
and may continue. Compare **first target-hit costs**, not total solve times.
Analytic derivative work is included in time; objective and gradient requests
are counted separately. C callbacks record errors and return NaN because their
ABI cannot return a Rust error; any recorded solver error makes the command fail.

Its JSONL schema `nlopt4r-native-v1` is intentionally separate from the main
harness and is not accepted by `--summarize`. Records include first-hit and
full-run costs, termination/errors, compiler, revision, dirty-tree status,
profile and algorithm versions. Existing output files are never overwritten.
Do not pool these unbounded experiments with the finite-box benchmark groups
or interpret their uncontrolled wall times as portable speedups.
