use crate::{
    catalog::{CATALOG, Group, Problem},
    instance::Transformation,
};
use nlopt4r::{Error, Evaluator, Method, Options, Vector};
use serde::{Deserialize, Serialize};
use std::{cell::RefCell, time::Instant};

pub const ALGORITHMS: &[&str] = &[
    "bfgs",
    "lbfgs",
    "de",
    "pso",
    "pattern-search",
    "nelder-mead",
    "external-pattern",
];

pub fn algorithm_available(name: &str) -> bool {
    name == "bfgs"
        || cfg!(feature = "extended")
            && matches!(
                name,
                "lbfgs" | "de" | "pso" | "pattern-search" | "nelder-mead"
            )
        || cfg!(feature = "external-pattern") && name == "external-pattern"
}

#[derive(Debug, Clone)]
pub struct Config {
    pub suite: String,
    pub algorithms: Vec<String>,
    pub seeds: Vec<u64>,
    pub budget_per_dimension: usize,
    pub dimension: usize,
    pub output: String,
    pub shift: bool,
    pub rotate: bool,
    pub instance_seed: u64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            suite: "smoke".into(),
            algorithms: vec!["bfgs".into()],
            seeds: vec![0],
            budget_per_dimension: 1000,
            dimension: 4,
            output: "benchmark.jsonl".into(),
            shift: false,
            rotate: false,
            instance_seed: 0,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct History {
    pub evaluation: usize,
    pub seconds: f64,
    pub best_value: f64,
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Environment {
    pub os: String,
    pub architecture: String,
    pub cpu: String,
    pub rustc: String,
    pub package_version: String,
    pub profile: String,
    pub revision: String,
}
impl Environment {
    pub fn collect() -> Result<Self, String> {
        let output = std::process::Command::new("rustc")
            .arg("--version")
            .output()
            .map_err(|e| e.to_string())?;
        if !output.status.success() {
            return Err("rustc --version failed".into());
        }
        let cpu = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("model name"))
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "unavailable".into());
        let revision = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
            .unwrap_or_else(|| "unavailable".into());
        Ok(Self {
            os: std::env::consts::OS.into(),
            architecture: std::env::consts::ARCH.into(),
            cpu,
            rustc: String::from_utf8_lossy(&output.stdout).trim().into(),
            package_version: env!("CARGO_PKG_VERSION").into(),
            profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
            .into(),
            revision,
        })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunRecord {
    pub schema_version: u32,
    pub problem: String,
    pub group: String,
    pub algorithm: String,
    pub seed: u64,
    pub dimension: usize,
    pub budget: usize,
    pub evaluations: usize,
    pub gradient_calls: usize,
    pub derivative_mode: String,
    pub reference_value: f64,
    pub target_value: f64,
    pub best_value: Option<f64>,
    pub best_x: Option<Vec<f64>>,
    pub hit_evaluation: Option<usize>,
    pub hit_seconds: Option<f64>,
    pub seconds: f64,
    pub termination: String,
    pub error: Option<String>,
    pub history: Vec<History>,
    pub environment: Environment,
    /// Absent in legacy schema-1 records and untransformed schema-2 records.
    #[serde(default)]
    pub transformation: Option<Transformation>,
}
struct State {
    evaluations: usize,
    gradients: usize,
    best: Option<(f64, Vec<f64>)>,
    history: Vec<History>,
    hit: Option<(usize, f64)>,
}
struct Measured<'a> {
    problem: &'a Problem,
    transformation: Option<&'a Transformation>,
    budget: usize,
    target: f64,
    start: Instant,
    state: RefCell<State>,
}
impl Measured<'_> {
    #[cfg(any(feature = "extended", feature = "external-pattern"))]
    fn checked_termination(
        &self,
        reason: nlopt4r::core::SearchTermination,
        evaluations: nlopt4r::EvaluationCounts,
    ) -> Result<String, Error> {
        if evaluations.objective != self.state.borrow().evaluations {
            return Err(Error::InvalidInput(
                "benchmark adapter objective counts disagree with measured callbacks",
            ));
        }
        Ok(format!("{reason:?}"))
    }
    fn value(&self, x: &[f64]) -> Result<f64, Error> {
        let mut s = self.state.borrow_mut();
        if s.evaluations >= self.budget {
            return Err(Error::EvaluationBudgetExceeded);
        }
        s.evaluations += 1;
        let transformed = self.transformation.map(|t| t.to_base(self.problem, x));
        let base_x = transformed.as_deref().unwrap_or(x);
        let value = self.problem.value(base_x);
        if !value.is_finite() {
            return Err(Error::NonFinite("benchmark objective"));
        }
        let seconds = self.start.elapsed().as_secs_f64();
        let feasible = x
            .iter()
            .all(|&v| v >= self.problem.bounds.0 && v <= self.problem.bounds.1)
            && self.problem.violation(base_x) <= 1e-8;
        if feasible && s.best.as_ref().is_none_or(|(v, _)| value < *v) {
            s.best = Some((value, x.to_vec()));
            let evaluation = s.evaluations;
            s.history.push(History {
                evaluation,
                seconds,
                best_value: value,
            });
        }
        if feasible && value <= self.target && s.hit.is_none() {
            s.hit = Some((s.evaluations, seconds));
        }
        Ok(value)
    }
}
impl Evaluator for Measured<'_> {
    fn evaluate(&self, x: &[f64], gradient: Option<&mut [f64]>) -> Result<f64, Error> {
        let value = self.value(x)?;
        if let Some(g) = gradient {
            self.state.borrow_mut().gradients += 1;
            if self.problem.analytic {
                let base_gradient = match self.transformation {
                    Some(t) => self.problem.gradient(&t.to_base(self.problem, x)),
                    None => self.problem.gradient(x),
                };
                let gradient = match self.transformation {
                    Some(t) => t.gradient(&base_gradient),
                    None => base_gradient,
                };
                g.copy_from_slice(&gradient);
            } else {
                let mut probe = x.to_vec();
                for i in 0..x.len() {
                    let h = f64::EPSILON.cbrt() * (1.0 + x[i].abs());
                    probe[i] = x[i] + h;
                    let plus = self.value(&probe)?;
                    probe[i] = x[i] - h;
                    let minus = self.value(&probe)?;
                    g[i] = (plus - minus) / (2.0 * h);
                    probe[i] = x[i];
                }
            }
            if !g.iter().all(|v| v.is_finite()) {
                return Err(Error::NonFinite("benchmark gradient"));
            }
        }
        Ok(value)
    }
}
pub fn selected(config: &Config) -> Result<Vec<&'static Problem>, String> {
    match config.suite.as_str() {
        "smoke" => Ok(CATALOG
            .iter()
            .filter(|p| {
                [
                    "sphere",
                    "rosenbrock",
                    "booth",
                    "rastrigin",
                    "linear-fit",
                    "coupled-qp",
                ]
                .contains(&p.id)
            })
            .collect()),
        "smooth" => Ok(CATALOG
            .iter()
            .filter(|p| p.analytic && p.group != Group::Constrained && p.group != Group::Global)
            .collect()),
        "global" => Ok(CATALOG
            .iter()
            .filter(|p| p.group == Group::Global)
            .collect()),
        _ => Err(format!("unknown suite {}", config.suite)),
    }
}
// A deterministic run-local SplitMix64 stream, independent of solver RNG state.
fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}
pub fn run(
    p: &Problem,
    algorithm: &str,
    seed: u64,
    config: &Config,
    environment: &Environment,
) -> Result<RunRecord, String> {
    if !ALGORITHMS.contains(&algorithm) {
        return Err(format!("unknown algorithm {algorithm}"));
    }
    let n = p.dimension(config.dimension)?;
    let budget = config
        .budget_per_dimension
        .checked_mul(n)
        .filter(|b| *b > 0)
        .ok_or("invalid or overflowing budget")?;
    let (_, reference) = p.reference(n);
    let target = reference + 1e-6 * (1.0 + reference.abs());
    let transformation =
        Transformation::new(p, n, config.shift, config.rotate, config.instance_seed)?;
    let measured = Measured {
        problem: p,
        transformation: transformation.as_ref(),
        budget,
        target,
        start: Instant::now(),
        state: RefCell::new(State {
            evaluations: 0,
            gradients: 0,
            best: None,
            history: Vec::new(),
            hit: None,
        }),
    };
    let mut rng = seed;
    let x0 = Vector::from_iterator(
        n,
        (0..n).map(|_| p.bounds.0 + (p.bounds.1 - p.bounds.0) * uniform(&mut rng)),
    );
    let mut options = Options::default();
    options.stopping.max_evaluations = Some(budget);
    options.stopping.max_iterations = budget;
    let outcome: Result<String, Error> = match algorithm {
        "bfgs" => nlopt4r::minimize(&measured, &x0, Method::Bfgs, &options)
            .map(|s| format!("{:?}", s.termination_reason)),
        #[cfg(feature = "external-pattern")]
        "external-pattern" => {
            let bounds = nlopt4r::Bounds::new(vec![p.bounds.0; n], vec![p.bounds.1; n])
                .map_err(|e| e.to_string())?;
            nlopt4r_extension_example::pattern_search(
                &measured,
                &bounds,
                &nlopt4r::core::BudgetOptions {
                    max_evaluations: budget,
                    target_value: Some(target),
                },
            )
            .and_then(|s| measured.checked_termination(s.termination_reason, s.evaluations))
        }
        #[cfg(feature = "extended")]
        "pattern-search" | "nelder-mead" => {
            let bounds = nlopt4r::Bounds::new(vec![p.bounds.0; n], vec![p.bounds.1; n])
                .map_err(|e| e.to_string())?;
            let options = nlopt4r::derivative_free::Options {
                budget: nlopt4r::core::BudgetOptions {
                    max_evaluations: budget,
                    target_value: Some(target),
                },
                max_iterations: budget,
                ..Default::default()
            };
            if algorithm == "pattern-search" {
                nlopt4r::derivative_free::pattern_search(&measured, &x0, &bounds, &options)
                    .and_then(|s| measured.checked_termination(s.termination_reason, s.evaluations))
            } else {
                nlopt4r::derivative_free::nelder_mead(&measured, &x0, &bounds, &options)
                    .and_then(|s| measured.checked_termination(s.termination_reason, s.evaluations))
            }
        }
        #[cfg(feature = "extended")]
        "lbfgs" => nlopt4r::local::lbfgs(
            &measured,
            &x0,
            &nlopt4r::local::LbfgsOptions {
                stopping: options.stopping.clone(),
                ..Default::default()
            },
        )
        .map(|s| format!("{:?}", s.termination_reason)),
        #[cfg(feature = "extended")]
        "de" | "pso" => {
            let bounds = nlopt4r::Bounds::new(vec![p.bounds.0; n], vec![p.bounds.1; n])
                .map_err(|e| e.to_string())?;
            let budget = nlopt4r::core::BudgetOptions {
                max_evaluations: budget,
                target_value: Some(target),
            };
            if algorithm == "de" {
                let max_generations = budget.max_evaluations;
                nlopt4r::global::differential_evolution(
                    &measured,
                    &bounds,
                    &nlopt4r::global::DeOptions {
                        budget,
                        seed,
                        max_generations,
                        ..Default::default()
                    },
                )
                .map(|s| format!("{:?}", s.termination_reason))
            } else {
                let max_generations = budget.max_evaluations;
                nlopt4r::global::particle_swarm(
                    &measured,
                    &bounds,
                    &nlopt4r::global::PsoOptions {
                        budget,
                        seed,
                        max_generations,
                        ..Default::default()
                    },
                )
                .map(|s| format!("{:?}", s.termination_reason))
            }
        }
        _ => {
            return Err(format!(
                "{algorithm} unavailable: enable its extended or external-pattern build feature"
            ));
        }
    };
    let seconds = measured.start.elapsed().as_secs_f64();
    let s = measured.state.into_inner();
    let (termination, error) = match outcome {
        Ok(reason) => (reason, None),
        Err(e) if matches!(e.root_cause(), Error::EvaluationBudgetExceeded) => {
            ("MaxEvaluations".into(), None)
        }
        Err(e) => ("Failed".into(), Some(e.to_string())),
    };
    Ok(RunRecord {
        schema_version: 2,
        problem: p.id.into(),
        group: format!("{:?}", p.group),
        algorithm: algorithm.into(),
        seed,
        dimension: n,
        budget,
        evaluations: s.evaluations,
        gradient_calls: s.gradients,
        derivative_mode: if s.gradients == 0 {
            "value-only"
        } else if p.analytic {
            "analytic-forward-chain-rule"
        } else {
            "counted-central-difference"
        }
        .into(),
        reference_value: reference,
        target_value: target,
        best_value: s.best.as_ref().map(|b| b.0),
        best_x: s.best.map(|b| b.1),
        hit_evaluation: s.hit.map(|h| h.0),
        hit_seconds: s.hit.map(|h| h.1),
        seconds,
        termination,
        error,
        history: s.history,
        environment: environment.clone(),
        transformation,
    })
}

pub fn parse(args: impl IntoIterator<Item = String>) -> Result<(Config, Option<String>), String> {
    let mut config = Config::default();
    let mut summarize = None;
    let mut args = args.into_iter();
    while let Some(key) = args.next() {
        if ![
            "--suite",
            "--algorithms",
            "--seeds",
            "--budget-per-dimension",
            "--dimension",
            "--output",
            "--summarize",
            "--shift",
            "--rotate",
            "--instance-seed",
        ]
        .contains(&key.as_str())
        {
            return Err(format!("unknown option {key}"));
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key.as_str() {
            "--suite" => config.suite = value,
            "--algorithms" => {
                config.algorithms = value.split(',').map(str::to_owned).collect();
                for a in &config.algorithms {
                    if !ALGORITHMS.contains(&a.as_str()) {
                        return Err(format!("unknown algorithm {a}"));
                    }
                }
                let unique = config
                    .algorithms
                    .iter()
                    .collect::<std::collections::HashSet<_>>();
                if unique.len() != config.algorithms.len() {
                    return Err("duplicate algorithms".into());
                }
            }
            "--seeds" => {
                if let Some((lo, hi)) = value.split_once("..") {
                    let lo = lo.parse::<u64>().map_err(|_| "invalid seed start")?;
                    let hi = hi.parse::<u64>().map_err(|_| "invalid seed end")?;
                    if lo >= hi || hi - lo > 10000 {
                        return Err("seed range must be nonempty and at most 10000 runs".into());
                    }
                    config.seeds = (lo..hi).collect();
                } else {
                    config.seeds = vec![value.parse().map_err(|_| "invalid seed")?];
                }
            }
            "--budget-per-dimension" => {
                config.budget_per_dimension = value.parse().map_err(|_| "invalid budget")?
            }
            "--dimension" => config.dimension = value.parse().map_err(|_| "invalid dimension")?,
            "--output" => config.output = value,
            "--summarize" => summarize = Some(value),
            "--shift" | "--rotate" => {
                let enabled = match value.as_str() {
                    "on" => true,
                    "off" => false,
                    _ => return Err(format!("{key} must be on or off")),
                };
                if key == "--shift" {
                    config.shift = enabled;
                } else {
                    config.rotate = enabled;
                }
            }
            "--instance-seed" => {
                config.instance_seed = value.parse().map_err(|_| "invalid instance seed")?
            }
            _ => unreachable!(),
        }
    }
    if config.budget_per_dimension == 0 || config.dimension == 0 || config.dimension > 128 {
        return Err("budget must be positive; dimension must be 1..=128".into());
    }
    selected(&config)?;
    if !config.output.ends_with(".jsonl") && !config.output.ends_with(".csv") {
        return Err("output must end in .jsonl or .csv".into());
    }
    Ok((config, summarize))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn environment() -> Environment {
        Environment {
            os: "test".into(),
            architecture: "test".into(),
            cpu: "test".into(),
            rustc: "test".into(),
            package_version: "test".into(),
            profile: "test".into(),
            revision: "test".into(),
        }
    }
    #[test]
    fn strict_cli_errors_and_seed_range() {
        for args in [
            vec!["--bogus", "x"],
            vec!["--algorithms", "bgfs"],
            vec!["--seeds", "30..0"],
            vec!["--budget-per-dimension", "0"],
            vec!["--dimension", "129"],
            vec!["--algorithms", "bfgs,bfgs"],
            vec!["--suite"],
            vec!["--shift", "yes"],
            vec!["--rotate", "true"],
            vec!["--instance-seed", "-1"],
        ] {
            assert!(parse(args.into_iter().map(str::to_owned)).is_err());
        }
        let (c, _) = parse(["--seeds", "0..30"].into_iter().map(str::to_owned)).unwrap();
        assert_eq!(c.seeds, (0..30).collect::<Vec<_>>());
        let (c, _) = parse(
            ["--shift", "on", "--rotate", "off", "--instance-seed", "17"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap();
        assert!(c.shift && !c.rotate && c.instance_seed == 17);
    }
    #[test]
    fn counted_derivative_budget_includes_partial_probes() {
        let p = CATALOG.iter().find(|p| p.id == "alpine").unwrap();
        let measured = Measured {
            problem: p,
            transformation: None,
            budget: 4,
            target: 1e-6,
            start: Instant::now(),
            state: RefCell::new(State {
                evaluations: 0,
                gradients: 0,
                best: None,
                history: Vec::new(),
                hit: None,
            }),
        };
        let mut gradient = [0.0; 2];
        assert!(matches!(
            measured.evaluate(&[1.0, 2.0], Some(&mut gradient)),
            Err(Error::EvaluationBudgetExceeded)
        ));
        assert_eq!(measured.state.borrow().evaluations, 4);
        assert_eq!(measured.state.borrow().gradients, 1);
        assert!(matches!(
            measured.evaluate(&[1.0, 2.0], None),
            Err(Error::EvaluationBudgetExceeded)
        ));
        assert_eq!(measured.state.borrow().evaluations, 4);
    }
    #[test]
    fn counted_fallback_derivatives_match_smooth_point_chain_rule() {
        for p in CATALOG.iter().filter(|p| !p.analytic) {
            let x = [0.31, 0.48, 0.65, 0.82];
            let measured = Measured {
                problem: p,
                transformation: None,
                budget: 20,
                target: 1e-6,
                start: Instant::now(),
                state: RefCell::new(State {
                    evaluations: 0,
                    gradients: 0,
                    best: None,
                    history: Vec::new(),
                    hit: None,
                }),
            };
            let mut gradient = [0.0; 4];
            measured.evaluate(&x, Some(&mut gradient)).unwrap();
            assert_eq!(measured.state.borrow().evaluations, 9);
            for (actual, expected) in gradient.iter().zip(p.gradient(&x)) {
                assert!(
                    (actual - expected).abs() < 1e-7 * (1.0 + expected.abs()),
                    "{}",
                    p.id
                );
            }
        }
    }
    #[test]
    fn state_and_seed_are_reset_between_runs() {
        let config = Config {
            budget_per_dimension: 20,
            ..Default::default()
        };
        let a = run(&CATALOG[0], "bfgs", 4, &config, &environment()).unwrap();
        let b = run(&CATALOG[0], "bfgs", 4, &config, &environment()).unwrap();
        assert_eq!(a.evaluations, b.evaluations);
        assert_eq!(a.best_x, b.best_x);
        assert_eq!(a.hit_evaluation, b.hit_evaluation);
        assert_eq!(
            a.history
                .iter()
                .map(|h| (h.evaluation, h.best_value))
                .collect::<Vec<_>>(),
            b.history
                .iter()
                .map(|h| (h.evaluation, h.best_value))
                .collect::<Vec<_>>()
        );
    }
    #[cfg(feature = "external-pattern")]
    #[test]
    fn external_pattern_is_deterministic_without_solver_seed() {
        let config = Config {
            budget_per_dimension: 100,
            ..Default::default()
        };
        let problem = CATALOG.iter().find(|p| p.id == "booth").unwrap();
        let a = run(problem, "external-pattern", 0, &config, &environment()).unwrap();
        let b = run(problem, "external-pattern", 9, &config, &environment()).unwrap();
        assert!(a.error.is_none());
        assert!(b.error.is_none());
        assert_eq!(a.best_x, b.best_x);
        assert_eq!(a.evaluations, b.evaluations);
        assert_eq!(a.hit_evaluation, b.hit_evaluation);
        assert_eq!(a.derivative_mode, "value-only");
        assert_eq!(
            a.history
                .iter()
                .map(|h| (h.evaluation, h.best_value))
                .collect::<Vec<_>>(),
            b.history
                .iter()
                .map(|h| (h.evaluation, h.best_value))
                .collect::<Vec<_>>()
        );
    }
    #[cfg(feature = "extended")]
    #[test]
    fn registry_algorithms_respect_small_budget() {
        let config = Config {
            budget_per_dimension: 3,
            ..Default::default()
        };
        for algorithm in ALGORITHMS
            .iter()
            .copied()
            .filter(|name| algorithm_available(name))
        {
            let record = run(&CATALOG[0], algorithm, 0, &config, &environment()).unwrap();
            assert!(record.evaluations <= record.budget, "{algorithm}");
            assert!(record.error.is_none(), "{algorithm}: {:?}", record.error);
        }
    }

    #[test]
    fn transformed_instances_are_shared_across_algorithms_and_optimizer_seeds() {
        let problem = CATALOG.iter().find(|p| p.id == "booth").unwrap();
        for (shift, rotate) in [(false, false), (true, false), (false, true), (true, true)] {
            let config = Config {
                shift,
                rotate,
                instance_seed: 123,
                dimension: 2,
                budget_per_dimension: 2,
                ..Default::default()
            };
            let a = run(problem, "bfgs", 4, &config, &environment()).unwrap();
            assert_eq!(a.transformation.is_some(), shift || rotate);
            for algorithm in ALGORITHMS.iter().filter(|name| algorithm_available(name)) {
                let b = run(problem, algorithm, 9, &config, &environment()).unwrap();
                assert_eq!(a.transformation, b.transformation);
                assert!(b.evaluations <= b.budget, "{algorithm}");
                assert!(b.error.is_none(), "{algorithm}: {:?}", b.error);
            }
        }
    }

    #[test]
    fn transformed_analytic_and_counted_gradients_use_solver_coordinates() {
        for id in ["ellipsoid", "alpine"] {
            let p = CATALOG.iter().find(|p| p.id == id).unwrap();
            let t = Transformation::new(p, 3, true, true, 7).unwrap().unwrap();
            let measured = Measured {
                problem: p,
                transformation: Some(&t),
                budget: 20,
                target: 1e-6,
                start: Instant::now(),
                state: RefCell::new(State {
                    evaluations: 0,
                    gradients: 0,
                    best: None,
                    history: Vec::new(),
                    hit: None,
                }),
            };
            let x = [0.31, 0.48, 0.65];
            let mut gradient = [0.0; 3];
            measured.evaluate(&x, Some(&mut gradient)).unwrap();
            assert_eq!(
                measured.state.borrow().evaluations,
                if p.analytic { 1 } else { 7 }
            );
            for i in 0..3 {
                let mut plus = x;
                let mut minus = x;
                plus[i] += 1e-6;
                minus[i] -= 1e-6;
                let expected =
                    (p.value(&t.to_base(p, &plus)) - p.value(&t.to_base(p, &minus))) / 2e-6;
                assert!(
                    (gradient[i] - expected).abs() < 1e-5 * (1.0 + expected.abs()),
                    "{id}: {} versus {expected}",
                    gradient[i]
                );
            }
            // Targets must be evaluated in transformed coordinates as well.
            measured.evaluate(&t.reference_point(p), None).unwrap();
            assert!(measured.state.borrow().hit.is_some());
        }
    }
}
