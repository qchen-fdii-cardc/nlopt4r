use nlopt::{Algorithm, Nlopt, Target};
use nlopt4r::utils::{ChaCha8Rng, SeedableRng};
use nlopt4r::{Bounds, Error, Evaluator, Vector};
use nlopt4r_bench::catalog::{CATALOG, Problem};
use nlopt4r_bench::instance::Transformation;
use serde_json::json;
use std::{
    cell::{Cell, RefCell},
    fs::OpenOptions,
    io::Write,
    time::Instant,
};

struct Measured<'a> {
    problem: &'a Problem,
    transformation: Option<&'a Transformation>,
    target: f64,
    started: Instant,
    calls: Cell<usize>,
    gradients: Cell<usize>,
    hit: Cell<Option<(usize, f64)>>,
    callback_error: RefCell<Option<String>>,
}
impl Evaluator for Measured<'_> {
    fn evaluate(&self, x: &[f64], gradient: Option<&mut [f64]>) -> Result<f64, Error> {
        self.calls.set(self.calls.get() + 1);
        let base = self.transformation.map(|t| t.to_base(self.problem, x));
        let base_x = base.as_deref().unwrap_or(x);
        let value = self.problem.value(base_x);
        if !value.is_finite() {
            return Err(Error::NonFinite("reference objective"));
        }
        if let Some(g) = gradient {
            self.gradients.set(self.gradients.get() + 1);
            let computed = self.problem.gradient(base_x);
            let computed = match self.transformation {
                Some(t) => t.gradient(&computed),
                None => computed,
            };
            nlopt4r::utils::validate_slice(&computed, x.len(), "reference gradient")?;
            g.copy_from_slice(&computed);
        }
        if value <= self.target && self.hit.get().is_none() {
            self.hit.set(Some((
                self.calls.get(),
                self.started.elapsed().as_secs_f64(),
            )));
        }
        Ok(value)
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dimension = 2usize;
    let mut budget_per_dimension = 2000usize;
    let mut seeds = 0u64..30;
    let mut algorithms = vec!["nlopt-lbfgs".to_string(), "nlopt4r-lbfgs".to_string()];
    let mut output = None;
    let mut shift = false;
    let mut rotate = false;
    let mut instance_seed = 0u64;
    let mut args = std::env::args().skip(1);
    while let Some(option) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {option}"))?;
        match option.as_str() {
            "--suite" if value == "smooth" => {}
            "--dimension" => dimension = value.parse()?,
            "--budget-per-dimension" => budget_per_dimension = value.parse()?,
            "--seeds" => {
                let (lo, hi) = value.split_once("..").ok_or("seeds must use lo..hi")?;
                seeds = lo.parse()?..hi.parse()?;
            }
            "--algorithms" => algorithms = value.split(',').map(str::to_string).collect(),
            "--output" => output = Some(value),
            "--shift" | "--rotate" => {
                let enabled = match value.as_str() {
                    "on" => true,
                    "off" => false,
                    _ => return Err(format!("{option} must be on or off").into()),
                };
                if option == "--shift" {
                    shift = enabled;
                } else {
                    rotate = enabled;
                }
            }
            "--instance-seed" => instance_seed = value.parse()?,
            _ => return Err(format!("unknown option or suite: {option} {value}").into()),
        }
    }
    if dimension < 2
        || seeds.is_empty()
        || budget_per_dimension == 0
        || algorithms.is_empty()
        || algorithms
            .iter()
            .any(|a| a != "nlopt-lbfgs" && a != "nlopt4r-lbfgs")
    {
        return Err("invalid dimensions, seed range, budget or algorithm list".into());
    }
    let limit = dimension
        .checked_mul(budget_per_dimension)
        .ok_or("budget overflow")?;
    let native_limit = u32::try_from(i32::try_from(limit)?)?;
    let output = output.ok_or("--output FILE.jsonl is required")?;
    let compiler = std::process::Command::new("rustc")
        .arg("--version")
        .output()?;
    if !compiler.status.success() {
        return Err("rustc metadata command failed".into());
    }
    let compiler = String::from_utf8(compiler.stdout)?.trim().to_string();
    let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let revision = std::process::Command::new("git")
        .arg("-C")
        .arg(&repository)
        .args(["rev-parse", "HEAD"])
        .output()?;
    if !revision.status.success() {
        return Err("git metadata command failed".into());
    }
    let revision = String::from_utf8(revision.stdout)?.trim().to_string();
    let status = std::process::Command::new("git")
        .arg("-C")
        .arg(&repository)
        .args(["status", "--porcelain"])
        .output()?;
    if !status.status.success() {
        return Err("git status metadata command failed".into());
    }
    let dirty = !status.stdout.is_empty();
    let mut writer = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let mut errors = 0;
    for id in ["sphere", "rosenbrock"] {
        let problem = CATALOG
            .iter()
            .find(|p| p.id == id)
            .ok_or("missing reference problem")?;
        let n = problem.dimension(dimension)?;
        let bounds = Bounds::uniform(n, problem.bounds.0, problem.bounds.1)?;
        let target = problem.reference(n).1 + 1e-6;
        let transformation = Transformation::new(problem, n, shift, rotate, instance_seed)?;
        for seed in seeds.clone() {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut initial = vec![0.0; n];
            bounds.sample(&mut rng, &mut initial)?;
            for algorithm in &algorithms {
                let measured = Measured {
                    problem,
                    transformation: transformation.as_ref(),
                    target,
                    started: Instant::now(),
                    calls: Cell::new(0),
                    gradients: Cell::new(0),
                    hit: Cell::new(None),
                    callback_error: RefCell::new(None),
                };
                let (termination, error) = if algorithm == "nlopt-lbfgs" {
                    let mut opt = Nlopt::new(
                        Algorithm::Lbfgs,
                        n,
                        |x: &[f64], g: Option<&mut [f64]>, p: &mut &Measured<'_>| match p
                            .evaluate(x, g)
                        {
                            Ok(value) => value,
                            Err(e) => {
                                *p.callback_error.borrow_mut() = Some(e.to_string());
                                f64::NAN
                            }
                        },
                        Target::Minimize,
                        &measured,
                    );
                    opt.set_maxeval(native_limit)
                        .map_err(|e| format!("NLopt maxeval: {e:?}"))?;
                    opt.set_stopval(target)
                        .map_err(|e| format!("NLopt target: {e:?}"))?;
                    opt.set_vector_storage(Some(10))
                        .map_err(|e| format!("NLopt vector storage: {e:?}"))?;
                    let mut x = initial.clone();
                    match opt.optimize(&mut x) {
                        Ok((status, _)) => (
                            format!("{status:?}"),
                            measured.callback_error.borrow().clone(),
                        ),
                        Err((status, _)) => (
                            format!("{status:?}"),
                            Some(
                                measured
                                    .callback_error
                                    .borrow()
                                    .clone()
                                    .unwrap_or_else(|| format!("NLopt failed: {status:?}")),
                            ),
                        ),
                    }
                } else {
                    let options = nlopt4r::local::LbfgsOptions {
                        stopping: nlopt4r::StopOptions {
                            max_evaluations: Some(limit),
                            gradient_tolerance: 1e-8,
                            ..Default::default()
                        },
                        ..Default::default()
                    };
                    match nlopt4r::local::lbfgs(
                        &measured,
                        &Vector::from_column_slice(&initial),
                        &options,
                    ) {
                        Ok(r) => (format!("{:?}", r.termination_reason), None),
                        Err(e) => ("Failed".to_string(), Some(e.to_string())),
                    }
                };
                if error.is_some() {
                    errors += 1;
                }
                let hit = measured.hit.get();
                let row = json!({"schema":"nlopt4r-native-v2","problem":id,"algorithm":algorithm,
                    "transformation":transformation,
                    "dimension":n,"seed":seed,"budget":limit,"target":target,
                    "domain":"unbounded; identical seeded box initialization",
                    "objective_evaluations":measured.calls.get(),"gradient_evaluations":measured.gradients.get(),
                    "target_hit_evaluations":hit.map(|h|h.0),"target_hit_seconds":hit.map(|h|h.1),
                    "elapsed_seconds":measured.started.elapsed().as_secs_f64(),
                    "termination_reason":termination,"error":error,
                    "nlopt_crate_version":"0.8.1","nlopt4r_version":env!("CARGO_PKG_VERSION"),
                    "os":std::env::consts::OS,"architecture":std::env::consts::ARCH,
                    "rustc":compiler,"revision":revision,"working_tree_dirty":dirty,
                    "history_size":10,"profile":if cfg!(debug_assertions){"debug"}else{"release"}});
                writeln!(writer, "{row}")?;
            }
        }
    }
    if errors > 0 {
        return Err(format!("{errors} native/reference solver errors recorded").into());
    }
    Ok(())
}
