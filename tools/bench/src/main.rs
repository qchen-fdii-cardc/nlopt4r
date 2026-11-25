use nlopt4r_bench::{
    runner::{self, Environment},
    summary,
};
use std::io::{BufReader, BufWriter, Write};

fn execute() -> Result<(), String> {
    let (config, summarize) = runner::parse(std::env::args().skip(1))?;
    if let Some(path) = summarize {
        let csv = path.ends_with(".csv");
        let input = std::fs::File::open(&path).map_err(|e| format!("{path}: {e}"))?;
        let records = summary::read_records(BufReader::new(input), csv)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&summary::summarize(&records))
                .map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    let environment = Environment::collect()?;
    let problems = runner::selected(&config)?;
    for problem in &problems {
        let dimension = problem.dimension(config.dimension)?;
        config
            .budget_per_dimension
            .checked_mul(dimension)
            .ok_or("budget overflow")?;
    }
    for algorithm in &config.algorithms {
        if !runner::algorithm_available(algorithm) {
            return Err(format!(
                "{algorithm} unavailable: enable its extended or external-pattern build feature"
            ));
        }
    }
    let csv = config.output.ends_with(".csv");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&config.output)
        .map_err(|e| {
            format!(
                "{}: {e} (existing outputs are never overwritten)",
                config.output
            )
        })?;
    let mut out = BufWriter::new(file);
    if csv {
        writeln!(out, "{}", summary::CSV_HEADER).map_err(|e| e.to_string())?;
    }
    let mut errors = 0usize;
    let mut runs = 0usize;
    for problem in problems {
        for algorithm in &config.algorithms {
            for &seed in &config.seeds {
                let record = runner::run(problem, algorithm, seed, &config, &environment)?;
                if let Some(error) = &record.error {
                    errors += 1;
                    eprintln!("{} {algorithm} seed {seed}: {error}", problem.id);
                }
                summary::write_record(&mut out, &record, csv)?;
                out.flush().map_err(|e| e.to_string())?;
                runs += 1;
            }
        }
    }
    eprintln!(
        "wrote {runs} runs to {} ({errors} solver errors)",
        config.output
    );
    out.flush().map_err(|e| e.to_string())?;
    out.get_ref().sync_all().map_err(|e| e.to_string())?;
    if errors > 0 {
        return Err(format!(
            "{errors} solver errors; all completed runs retained"
        ));
    }
    Ok(())
}
fn main() {
    if let Err(error) = execute() {
        eprintln!("benchmark error: {error}");
        std::process::exit(1);
    }
}
