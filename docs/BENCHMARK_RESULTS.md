# Measured first-release baseline

Algorithm source: `5a245cada6c8a60cf4059a4437004b674c1ee39b`, with a clean
working tree at experiment launch and completion. Release builds used Rust
1.99.1 on Linux/x86_64, Intel Core i7-12700H. CPU affinity, frequency,
background load and compiler flags were not controlled; no portable wall-time
speedup is claimed.

All groups use seeds 0..29 and the default algorithm settings at that revision.
DE: population 40, F=0.8, CR=0.9. PSO: population 40, inertia=0.7,
cognitive/social=1.5, velocity limit=0.2 of coordinate width.
Population generation limit is 1000; the objective budget can stop earlier.
See [the protocol](BENCHMARKS.md) for targets, starts, derivative accounting,
feasibility and native-versus-Rust stopping differences.

## Reproduce and inspect

```sh
cargo run --release --locked -p nlopt4r-bench -- \
  --suite smoke --algorithms bfgs,lbfgs,de,pso --seeds 0..30 \
  --dimension 2 --budget-per-dimension 2000 --output target/reproduce-smoke.jsonl
cargo run --release --locked -p nlopt4r-bench -- \
  --suite global --algorithms de,pso --seeds 0..30 \
  --dimension 4 --budget-per-dimension 2000 --output target/reproduce-global.jsonl
cargo run --release --locked -p nlopt4r-bench -- \
  --summarize target/reproduce-global.jsonl
cargo run --release --locked -p nlopt4r-reference -- \
  --suite smooth --algorithms nlopt-lbfgs,nlopt4r-lbfgs --seeds 0..30 \
  --dimension 2 --budget-per-dimension 2000 --output target/reproduce-native.jsonl
```

Repeat the native command with dimension 20 and a different output filename.
That experiment records five Rust line-search errors and therefore exits
unsuccessfully after writing all 120 records; this is intentional error
reporting, not a missing dataset.

Compressed original JSONL records, including histories where available:
[smoke](../tools/bench/results/final-smoke.jsonl.gz),
[global](../tools/bench/results/final-global.jsonl.gz),
[native 2-D](../tools/bench/results/final-native.jsonl.gz),
[native 20-D](../tools/bench/results/final-native-20.jsonl.gz).
Decompress with `gzip -dc`; the main harness summarizes its decompressed files.
The native schema remains separate. Results reside in unpublished tooling and
are excluded from the solver package.

## Fixed-target results

Each cell is **target hits / 30; ERT in objective evaluations**. ERT includes
actual failed-run costs, not just successful runs; infinity means no target hits.
Targets are reference + 1e-6*(1+abs(reference)) for the finite-box harness.
All recorded objective counts stay within their budgets. Target misses are
not necessarily solver errors.

### Smoke: dimension 2, budget 4000

720 records; no solver errors.

| Problem | BFGS | L-BFGS | DE | PSO |
|---|---|---|---|---|
| Sphere | 30/30; 3.00 | 30/30; 3.00 | 30/30; 1420.67 | 30/30; 1343.23 |
| Rosenbrock | 30/30; 62.87 | 30/30; 39.30 | 30/30; 2232.73 | 24/30; 3879.67 |
| Booth | 30/30; 13.07 | 30/30; 10.53 | 30/30; 1744.53 | 30/30; 1773.03 |
| Rastrigin | 0/30; infinity | 0/30; infinity | 29/30; 3534.10 | 30/30; 2513.23 |
| Linear fit | 30/30; 9.00 | 30/30; 10.20 | 30/30; 1517.37 | 30/30; 1448.27 |
| Coupled QP | 30/30; 11.70 | 30/30; 7.27 | 30/30; 1320.60 | 30/30; 1173.60 |

### Global: requested dimension 4

540 records; no solver errors. Fixed-size cases retain dimension 2.

| Problem | Dimension / budget | DE | PSO |
|---|---|---|---|
| Ackley | 4 / 8000 | 2/30; 119806.50 | 30/30; 5067.03 |
| Alpine | 4 / 8000 | 0/30; infinity | 30/30; 4891.83 |
| Easom | 2 / 4000 | 30/30; 1722.60 | 30/30; 1499.43 |
| Griewank | 4 / 8000 | 0/30; infinity | 0/30; infinity |
| Himmelblau | 2 / 4000 | 30/30; 2977.20 | 30/30; 1942.23 |
| Levy | 4 / 8000 | 30/30; 4402.57 | 30/30; 2351.37 |
| Qing | 4 / 8000 | 0/30; infinity | 30/30; 3096.20 |
| Rastrigin | 4 / 8000 | 0/30; infinity | 7/30; 31449.29 |
| Schwefel-222 | 4 / 8000 | 1/30; 239707.00 | 30/30; 5193.03 |

ERT can exceed an individual run's budget when many failures are charged.
These results support documenting parameter sensitivity and further adaptive
strategies/restarts; they do not justify a claim of reliable global convergence.

### Native L-BFGS: identical unbounded domains

240 records. Both solvers use history size 10 and identical ChaCha8 starts;
target is reference + 1e-6. Native NLopt uses the crates.io wrapper 0.8.1.
Rust uses gradient tolerance 1e-8 and may continue after a target hit.

| Problem | Dimension / budget | Native NLopt | nlopt4r |
|---|---|---|---|
| Sphere | 2 / 4000 | 30/30; 3.00 | 30/30; 3.00 |
| Rosenbrock | 2 / 4000 | 30/30; 37.83 | 30/30; 43.30 |
| Sphere | 20 / 40000 | 30/30; 3.00 | 30/30; 3.00 |
| Rosenbrock | 20 / 40000 | 25/30; 166.52 | 25/30; 195.80 |

The 20-D Rust Rosenbrock runs with seeds 1, 5, 9, 15 and 16 report explicit
strong-Wolfe search exhaustion; none hits the global target. Native also misses
five targets, but reports no solver errors. These failures are retained in ERT.
Near-stationary search termination warrants follow-up; no unchecked-step or
relaxed-success workaround was added. Optional stagnation controls remain
available for applications that want a non-converged plateau report.
The data do **not** show Rust L-BFGS beating native NLopt in objective-call cost.

## Validation and measured coverage

- Stable and Rust 1.88.0 workspace tests pass: 90 solver integration tests,
  three solver unit tests, one doctest, ten native comparisons and 14 tooling
  tests (118 total).
- Formatting, strict all-target Clippy and warning-free workspace docs pass.
- All 13 Criterion kernel smoke checks pass; offline mode avoids metadata
  network stalls.
- Allocation regressions enforce vector/limited-memory workspace at dimensions
  512 and 2048; matrix-free and ill-conditioned solver tests also pass.
- The extracted package passes all 94 pure-Rust tests. An independent application
  builds against it and verifies buffered local and seeded global APIs.
- Publication dry-run succeeds and explicitly aborts upload. The pre-report
  archive contains 55 files, about 98.2 KiB compressed, with native/tooling
  directories excluded. Repeat packaging after documentation updates.
- Licensing and MATLAB-source redistribution approval remain release blockers.
  No license was guessed and nothing was uploaded.

`cargo llvm-cov -p nlopt4r --json --output-path target/coverage-full.json`
measured **90.05% lines** (3331/3699), **93.91% functions** (262/279) and
88.34% regions (5546/6278), excluding doctests and native tooling.
Seventeen functions and 732 regions remain unexecuted; utility solve paths,
callback adapters, validation/error handling and Wolfe safeguards are remaining
coverage areas. Branch coverage was **not instrumented**, so no branch-coverage
percentage is claimed. These measurements are a baseline, not a proof of
numerical correctness or a newly invented coverage threshold.

Exact COCO/BBOB/CEC/CUTEst integration, additional certified datasets, full
L-BFGS-B, preconditioning and further adaptive strategies remain explicitly
unchecked extension items in [the roadmap](ROADMAP.md).

## Paired shift/rotation experiment

Source: `360e609`, clean working tree. This follow-up uses the same machine and
compiler as above, instance seed **42**, optimizer seeds **0..29**, dimension 2,
budget **4000**, and `de,pso,external-pattern` across all six smoke problems.
Four independent on/off combinations produce **2160 records** with zero solver
errors; target misses remain failures in ERT. The external pattern example is
deterministic: its 30 seed labels repeat the same midpoint start.

Each cell is **target hits / 30; ERT in evaluations**, including failed-run costs:

| Problem / algorithm | Both off | Shift only | Rotation only | Shift + rotation |
|---|---|---|---|---|
| Sphere / external pattern | 30/30; 1.00 | 30/30; 96.00 | 30/30; 1.00 | 30/30; 96.00 |
| Sphere / DE | 30/30; 1420.67 | 30/30; 1350.47 | 30/30; 1420.67 | 30/30; 1350.47 |
| Coupled QP / external pattern | 30/30; 1.00 | 30/30; 85.00 | 30/30; 1.00 | 30/30; 93.00 |
| Rastrigin / external pattern | 30/30; 1.00 | 0/30; infinity | 30/30; 1.00 | 0/30; infinity |
| Rastrigin / DE | 29/30; 3534.10 | 30/30; 3345.03 | 25/30; 4150.28 | 24/30; 4462.96 |
| Rastrigin / PSO | 30/30; 2513.23 | 29/30; 2585.52 | 28/30; 2889.46 | 25/30; 3562.88 |

This removes a real center-start advantage: the external example no longer gets
an automatic one-call target hit on shifted sphere/coupled QP/Rastrigin.
Orientation matters on Rastrigin, while sphere is rotation-invariant.
Translation does **not** make every solver/problem harder: shifted sphere DE
uses fewer calls here. One instance seed is illustrative, not statistical
evidence across independent landscapes, and no controlled timing claim is made.

All **360** untransformed DE/PSO outcomes exactly match the original retained
baseline in objective counts, best values/points, first target-hit counts and
termination. Across all four modes, every algorithm and optimizer seed receives
the same transformation for a given problem. The shift vector is identical
between shift-only and combined runs; the matrix is identical between
rotation-only and combined runs.

An additional **120 native comparison records** verify both native/Rust L-BFGS
receive the exact same transformations as the main harness, with no solver errors
and all objective counts within budget. The native experiment remains unbounded
and uses its own shared native-runner initialization protocol; do not pool it
with finite-box harness results.

Complete replayable evidence:
[paired JSONL](../tools/bench/results/affine-paired.jsonl.gz) and
[native JSONL](../tools/bench/results/affine-native.jsonl.gz).
Decompress the paired file and pass it to `--summarize`: its 72 summary groups
keep all problems/algorithms/transformations separate. See
[commands, formulas and domain caveats](BENCHMARKS.md#optional-recorded-shifts-and-rotations).

Validation includes 20 benchmark tests on stable and Rust 1.88, 18 tests with
default tooling features disabled, ten native correctness comparisons,
gradient-chain-rule and counted-difference checks, proper orthogonal rotations
through dimension 128, historical-schema reading, exact coefficient round trips
and rejection of corrupt transformation metadata.

## Paired observation-noise experiment

Source: `7fa4b1d`, clean working tree. The main harness ran DE and PSO on the
six smoke problems, dimension 2, optimizer seeds **0..29**, budget **4000**,
shift + rotation enabled with instance seed **42**, and noise seed **17**.
Seven configurations (noise off; Gaussian, uniform and Laplace at standard
deviations **1e-6** and **0.01**) produced **2520 records**, **84 summary groups**,
and **zero solver errors**. Misses remain failures; zero errors is not zero misses.

Each cell below is **noise-free target visits / successful returned recommendations**,
out of 30 runs. A dash means post-run recommendation scoring is disabled for
noiseless runs, not zero successes.

| Distribution | Standard deviation | Sphere / PSO | Rastrigin / DE |
|---|---:|---:|---:|
| None | 0 | 30 / - | 24 / - |
| Gaussian | 1e-6 | 30 / 30 | 24 / 22 |
| Uniform | 1e-6 | 30 / 30 | 23 / 21 |
| Laplace | 1e-6 | 30 / 29 | 24 / 23 |
| Gaussian | 0.01 | 17 / 0 | 7 / 0 |
| Uniform | 0.01 | 29 / 1 | 5 / 0 |
| Laplace | 0.01 | 14 / 0 | 5 / 0 |

The distinction matters: under Gaussian sigma 0.01, sphere PSO visits the target
in 17/30 runs but returns a target-quality point in **none**. A favorable noisy
incumbent is not necessarily a good solution. Across all six problems and both
algorithms at sigma 0.01, Gaussian yields 118 visited hits and 0 recommendation
hits out of 360; uniform yields 208 and 5; Laplace yields 80 and 0.
These are observations on one transformed instance seed, not universal rankings
of noise models or claims of noise-robust optimization.

Failure-inclusive visitation ERT for sphere PSO is 1361.03 evaluations without
noise, 5280.29 with Gaussian sigma 0.01, 2130.76 with uniform sigma 0.01, and
7109.79 with Laplace sigma 0.01. ERT can exceed the per-run budget because failed
runs contribute cost. Noisy runs disable objective-target stopping and use the
full 4000-call budget here; noiseless runs keep their original stopping policy.
First-hit costs remain comparable as visitation metrics, not certification that
the noisy solver selected the optimum. No wall-time speedup is claimed.

All **360 noiseless outcomes** exactly reproduce the previous shifted/rotated
baseline: objective/gradient counts, best points/values, first-hit counts,
termination, transformation coefficients, and complete evaluation/value histories.
For every noisy run, draw count equals objective count, objective count stays
within budget, and the single post-run truth score is separately counted.
Transform coefficients remain identical across every noise model/level.
Summary successes, recommendation successes and ERT were independently recomputed;
transformed sphere recommendation scores were independently checked as well.

Replayable evidence: [paired noise JSONL](../tools/bench/results/noise-paired.jsonl.gz).
Decompress and use `--summarize`; historical schema-1/2 records remain supported.
See [noise options, sampling definitions and gradient caveats](BENCHMARKS.md#optional-reproducible-evaluation-noise).
The native comparison runner remains noiseless.

Validation: full workspace regression suite (including native comparisons and
solver doctests), **28 benchmark tests** on stable and Rust 1.88, **25** with
default features disabled, warning-free Clippy for both feature configurations,
and warning-free benchmark rustdoc. Focused tests cover seeded distribution
moments/scaling, repeated evaluations, strict noisy-probe budgets, exact
mean-gradient semantics, false-target prevention, actual recommendation quality,
overflow/error records, serialization, corrupt metadata and separated summaries.
