use nlopt::{Matrix, Method, Options, Vector, minimize};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem = (
        |x: &Vector| 100.0 * (x[0] * x[0] - x[1]).powi(2) + (x[0] - 1.0).powi(2),
        |x: &Vector| {
            Vector::from_vec(vec![
                400.0 * x[0] * (x[0] * x[0] - x[1]) + 2.0 * (x[0] - 1.0),
                -200.0 * (x[0] * x[0] - x[1]),
            ])
        },
        |x: &Vector| {
            Matrix::from_row_slice(
                2,
                2,
                &[
                    1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0,
                    -400.0 * x[0],
                    -400.0 * x[0],
                    200.0,
                ],
            )
        },
    );
    let result = minimize(
        &problem,
        &Vector::from_vec(vec![-1.2, 1.0]),
        Method::Bfgs,
        &Options::default(),
    )?;
    if !result.converged() {
        return Err("iteration budget exhausted".into());
    }
    println!(
        "x = {}, f = {:.12e}, iterations = {}",
        result.x.transpose(),
        result.value,
        result.iterations
    );
    Ok(())
}
