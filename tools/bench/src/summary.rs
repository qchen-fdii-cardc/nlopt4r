use crate::{
    catalog::CATALOG,
    instance::Transformation,
    runner::{ALGORITHMS, Environment, RunRecord},
};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    io::{BufRead, Write},
};

#[derive(Debug, Serialize)]
pub struct Summary {
    pub problem: String,
    pub algorithm: String,
    pub dimension: usize,
    pub budget: usize,
    pub target_value: f64,
    pub environment: Environment,
    pub transformation: Option<Transformation>,
    pub runs: usize,
    pub successes: usize,
    pub errors: usize,
    pub success_rate: f64,
    /// Null if no successes: ERT is then infinite, not zero.
    pub ert_evaluations: Option<f64>,
    pub ert_seconds: Option<f64>,
}
pub fn summarize(records: &[RunRecord]) -> Vec<Summary> {
    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    struct Key {
        problem: String,
        algorithm: String,
        dimension: usize,
        budget: usize,
        target_bits: u64,
        environment: Environment,
        transformation: Option<InstanceKey>,
    }
    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    struct InstanceKey {
        version: u32,
        seed: u64,
        shifted: bool,
        rotated: bool,
        shift_bits: Vec<u64>,
        rotation_bits: Vec<Vec<u64>>,
    }
    let mut groups: BTreeMap<Key, Vec<&RunRecord>> = BTreeMap::new();
    for r in records {
        groups
            .entry(Key {
                problem: r.problem.clone(),
                algorithm: r.algorithm.clone(),
                dimension: r.dimension,
                budget: r.budget,
                target_bits: r.target_value.to_bits(),
                environment: r.environment.clone(),
                transformation: r.transformation.as_ref().map(|t| InstanceKey {
                    version: t.version,
                    seed: t.seed,
                    shifted: t.shift_enabled,
                    rotated: t.rotation_enabled,
                    shift_bits: t.shift.iter().map(|v| v.to_bits()).collect(),
                    rotation_bits: t
                        .rotation
                        .iter()
                        .map(|row| row.iter().map(|v| v.to_bits()).collect())
                        .collect(),
                }),
            })
            .or_default()
            .push(r);
    }
    groups
        .into_iter()
        .map(
            |(
                Key {
                    problem,
                    algorithm,
                    dimension,
                    budget,
                    target_bits,
                    environment,
                    transformation: _,
                },
                runs,
            )| {
                let successes = runs.iter().filter(|r| r.hit_evaluation.is_some()).count();
                let evaluations = runs
                    .iter()
                    .map(|r| r.hit_evaluation.unwrap_or(r.evaluations) as f64)
                    .sum::<f64>();
                let seconds = runs
                    .iter()
                    .map(|r| r.hit_seconds.unwrap_or(r.seconds))
                    .sum::<f64>();
                Summary {
                    problem,
                    algorithm,
                    dimension,
                    budget,
                    target_value: f64::from_bits(target_bits),
                    environment,
                    transformation: runs[0].transformation.clone(),
                    runs: runs.len(),
                    successes,
                    errors: runs.iter().filter(|r| r.error.is_some()).count(),
                    success_rate: successes as f64 / runs.len() as f64,
                    ert_evaluations: if successes > 0 {
                        Some(evaluations / successes as f64)
                    } else {
                        None
                    },
                    ert_seconds: if successes > 0 {
                        Some(seconds / successes as f64)
                    } else {
                        None
                    },
                }
            },
        )
        .collect()
}
pub const CSV_HEADER: &str =
    "problem,algorithm,seed,dimension,evaluations,hit_evaluation,seconds,record_json";
pub fn write_record(out: &mut impl Write, r: &RunRecord, csv: bool) -> Result<(), String> {
    let json = serde_json::to_string(r).map_err(|e| e.to_string())?;
    if csv {
        writeln!(
            out,
            "{},{},{},{},{},{},{},\"{}\"",
            r.problem,
            r.algorithm,
            r.seed,
            r.dimension,
            r.evaluations,
            r.hit_evaluation.map(|n| n.to_string()).unwrap_or_default(),
            r.seconds,
            json.replace('"', "\"\"")
        )
    } else {
        writeln!(out, "{json}")
    }
    .map_err(|e| e.to_string())
}
pub fn read_records(input: impl BufRead, csv: bool) -> Result<Vec<RunRecord>, String> {
    let mut result = Vec::new();
    for (i, line) in input.lines().enumerate() {
        let line = line.map_err(|e| format!("line {}: {e}", i + 1))?;
        if csv && i == 0 {
            if line != CSV_HEADER {
                return Err("unexpected CSV schema".into());
            }
            continue;
        }
        if line.trim().is_empty() {
            return Err(format!("empty record at line {}", i + 1));
        }
        let fields = if csv {
            Some(line.splitn(8, ',').collect::<Vec<_>>())
        } else {
            None
        };
        let json = if let Some(fields) = &fields {
            if fields.len() != 8 {
                return Err(format!("missing CSV columns at line {}", i + 1));
            }
            let field = fields[7];
            if !field.starts_with('"') || !field.ends_with('"') {
                return Err(format!("malformed CSV line {}", i + 1));
            }
            field[1..field.len() - 1].replace("\"\"", "\"")
        } else {
            line.clone()
        };
        let r: RunRecord =
            serde_json::from_str(&json).map_err(|e| format!("line {}: {e}", i + 1))?;
        if let Some(fields) = fields {
            let hit = if fields[5].is_empty() {
                None
            } else {
                Some(fields[5].parse::<usize>().map_err(|_| "invalid CSV hit")?)
            };
            if fields[0] != r.problem
                || fields[1] != r.algorithm
                || fields[2].parse::<u64>().map_err(|_| "invalid CSV seed")? != r.seed
                || fields[3]
                    .parse::<usize>()
                    .map_err(|_| "invalid CSV dimension")?
                    != r.dimension
                || fields[4]
                    .parse::<usize>()
                    .map_err(|_| "invalid CSV evaluations")?
                    != r.evaluations
                || hit != r.hit_evaluation
                || fields[6]
                    .parse::<f64>()
                    .map_err(|_| "invalid CSV seconds")?
                    != r.seconds
            {
                return Err(format!("inconsistent CSV columns at line {}", i + 1));
            }
        }
        validate(&r).map_err(|e| format!("line {}: {e}", i + 1))?;
        result.push(r);
    }
    if result.is_empty() {
        return Err("no benchmark records".into());
    }
    Ok(result)
}
fn validate(r: &RunRecord) -> Result<(), String> {
    if !ALGORITHMS.contains(&r.algorithm.as_str()) {
        return Err(format!("unknown algorithm {}", r.algorithm));
    }
    let problem = CATALOG
        .iter()
        .find(|p| p.id == r.problem)
        .ok_or_else(|| format!("unknown problem {}", r.problem))?;
    if problem.dimension(r.dimension)? != r.dimension {
        return Err("invalid fixed dimension".into());
    }
    if let Some(t) = &r.transformation {
        t.validate(problem, r.dimension)?;
    }
    if !matches!(r.schema_version, 1 | 2)
        || (r.schema_version == 1 && r.transformation.is_some())
        || r.budget == 0
        || r.evaluations > r.budget
        || !r.seconds.is_finite()
        || r.seconds < 0.0
        || !r.target_value.is_finite()
        || r.hit_evaluation.is_some() != r.hit_seconds.is_some()
        || r.hit_evaluation
            .is_some_and(|v| v == 0 || v > r.evaluations)
        || r.hit_seconds
            .is_some_and(|v| !v.is_finite() || v < 0.0 || v > r.seconds)
        || r.best_x
            .as_ref()
            .is_some_and(|x| x.len() != r.dimension || x.iter().any(|v| !v.is_finite()))
        || r.best_x.is_some() != r.best_value.is_some()
        || r.best_value.is_some_and(|v| !v.is_finite())
        || r.hit_evaluation.is_some() && r.best_value.is_none_or(|v| v > r.target_value)
    {
        return Err("invalid benchmark accounting".into());
    }
    let mut previous = None;
    for h in &r.history {
        if h.evaluation == 0
            || h.evaluation > r.evaluations
            || !h.seconds.is_finite()
            || h.seconds < 0.0
            || h.seconds > r.seconds
            || !h.best_value.is_finite()
            || previous
                .is_some_and(|(e, t, v)| h.evaluation <= e || h.seconds < t || h.best_value >= v)
        {
            return Err("invalid target history".into());
        }
        previous = Some((h.evaluation, h.seconds, h.best_value));
    }
    if r.history.last().map(|h| h.best_value) != r.best_value {
        return Err("history and best value disagree".into());
    }
    if let (Some(hit), Some(time)) = (r.hit_evaluation, r.hit_seconds)
        && !r
            .history
            .iter()
            .any(|h| h.evaluation == hit && h.seconds == time && h.best_value <= r.target_value)
    {
        return Err("target hit missing from evaluation history".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        catalog::CATALOG,
        runner::{Config, Environment, run},
    };
    fn record() -> RunRecord {
        let env = Environment {
            os: "test".into(),
            architecture: "test".into(),
            cpu: "test".into(),
            rustc: "test".into(),
            package_version: "test".into(),
            profile: "test".into(),
            revision: "test".into(),
        };
        run(&CATALOG[0], "bfgs", 0, &Config::default(), &env).unwrap()
    }
    #[test]
    fn failures_charge_ert_and_zero_success_is_infinite() {
        let mut hit = record();
        hit.evaluations = 20;
        hit.hit_evaluation = Some(10);
        hit.hit_seconds = Some(0.1);
        hit.seconds = 0.2;
        let mut fail = hit.clone();
        fail.evaluations = 100;
        fail.hit_evaluation = None;
        fail.hit_seconds = None;
        fail.seconds = 1.0;
        fail.error = Some("failure".into());
        let s = summarize(&[hit, fail.clone()]);
        assert_eq!(s[0].success_rate, 0.5);
        assert_eq!(s[0].ert_evaluations, Some(110.0));
        assert!((s[0].ert_seconds.unwrap() - 1.1).abs() < 1e-12);
        assert_eq!(s[0].errors, 1);
        assert_eq!(summarize(&[fail])[0].ert_evaluations, None);
    }
    #[test]
    fn formats_roundtrip_and_reject_corruption() {
        let r = record();
        for csv in [false, true] {
            let mut out = Vec::new();
            if csv {
                writeln!(out, "{CSV_HEADER}").unwrap();
            }
            write_record(&mut out, &r, csv).unwrap();
            let parsed = read_records(std::io::Cursor::new(out), csv).unwrap();
            assert_eq!(parsed[0].evaluations, r.evaluations);
            assert_eq!(parsed[0].history.len(), r.history.len());
        }
        assert!(read_records(std::io::Cursor::new(b"{bad}\n"), false).is_err());
    }
    #[test]
    fn transformed_records_roundtrip_separate_instances_and_reject_corruption() {
        let mut a = record();
        a.transformation = Transformation::new(&CATALOG[0], a.dimension, true, true, 42).unwrap();
        for csv in [false, true] {
            let mut out = Vec::new();
            if csv {
                writeln!(out, "{CSV_HEADER}").unwrap();
            }
            write_record(&mut out, &a, csv).unwrap();
            let parsed = read_records(std::io::Cursor::new(out), csv).unwrap();
            assert_eq!(parsed[0].transformation, a.transformation);
        }
        let baseline = record();
        let mut other_seed = a.clone();
        other_seed.transformation =
            Transformation::new(&CATALOG[0], a.dimension, true, true, 43).unwrap();
        let mut shift_only = a.clone();
        shift_only.transformation =
            Transformation::new(&CATALOG[0], a.dimension, true, false, 42).unwrap();
        let mut other_matrix = a.clone();
        // Flip two columns: another proper rotation with the same seed label.
        for row in &mut other_matrix.transformation.as_mut().unwrap().rotation {
            row[0] = -row[0];
            row[1] = -row[1];
        }
        validate(&other_matrix).unwrap();
        assert_eq!(
            summarize(&[baseline, a.clone(), other_seed, shift_only, other_matrix]).len(),
            5
        );
        a.transformation.as_mut().unwrap().rotation[0][0] += 0.1;
        let bad = format!("{}\n", serde_json::to_string(&a).unwrap());
        assert!(read_records(std::io::Cursor::new(bad), false).is_err());

        let mut legacy = serde_json::to_value(record()).unwrap();
        legacy["schema_version"] = 1.into();
        legacy.as_object_mut().unwrap().remove("transformation");
        let parsed = read_records(std::io::Cursor::new(format!("{legacy}\n")), false).unwrap();
        assert!(parsed[0].transformation.is_none());
    }
    #[test]
    fn summaries_do_not_mix_budgets_or_targets() {
        let a = record();
        let mut b = a.clone();
        b.budget += 1;
        let mut c = a.clone();
        c.target_value += 1.0;
        assert_eq!(summarize(&[a, b, c]).len(), 3);
    }
    #[test]
    fn failed_runs_in_different_environments_are_not_pooled() {
        let hit = record();
        assert!(hit.hit_evaluation.is_some());
        for field in 0..7 {
            let mut failure = hit.clone();
            failure.hit_evaluation = None;
            failure.hit_seconds = None;
            failure.error = Some("solver failure".into());
            let env = &mut failure.environment;
            let value = match field {
                0 => &mut env.cpu,
                1 => &mut env.os,
                2 => &mut env.architecture,
                3 => &mut env.rustc,
                4 => &mut env.revision,
                5 => &mut env.profile,
                _ => &mut env.package_version,
            };
            *value = "different".into();
            let summaries = summarize(&[hit.clone(), failure]);
            assert_eq!(summaries.len(), 2);
            assert!(summaries.iter().all(|s| s.runs == 1));
            let failed = summaries.iter().find(|s| s.errors == 1).unwrap();
            assert_eq!(failed.successes, 0);
            assert_eq!(failed.ert_evaluations, None);
            assert_eq!(failed.ert_seconds, None);
            let successful = summaries.iter().find(|s| s.successes == 1).unwrap();
            assert_eq!(successful.success_rate, 1.0);
            assert_eq!(
                successful.ert_evaluations,
                hit.hit_evaluation.map(|v| v as f64)
            );
        }
    }
}
