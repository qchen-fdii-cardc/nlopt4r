use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use nlopt4r::{Evaluator, Function, Matrix, Method, Options, Vector};

fn kernels(c: &mut Criterion) {
    let evaluator = Function::new(|x: &[f64], gradient: Option<&mut [f64]>| {
        if let Some(g) = gradient {
            for (out, &v) in g.iter_mut().zip(x) {
                *out = 2.0 * v;
            }
        }
        Ok(x.iter().map(|v| v * v).sum())
    });
    let mut group = c.benchmark_group("buffered_evaluation");
    for n in [8, 128, 1024] {
        let x = vec![0.5; n];
        let mut gradient = vec![0.0; n];
        group.bench_with_input(BenchmarkId::new("value", n), &n, |b, _| {
            b.iter(|| evaluator.evaluate(black_box(&x), None).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("value_gradient", n), &n, |b, _| {
            b.iter(|| {
                evaluator
                    .evaluate(black_box(&x), Some(black_box(&mut gradient)))
                    .unwrap()
            })
        });
    }
    group.finish();
    let mut group = c.benchmark_group("dense_linear_algebra");
    for n in [8, 128] {
        let x = Vector::from_element(n, 0.5);
        let y = Vector::from_element(n, 0.25);
        let matrix = Matrix::from_fn(n, n, |i, j| if i == j { 2.0 } else { 0.01 });
        group.bench_with_input(BenchmarkId::new("dot", n), &n, |b, _| {
            b.iter(|| black_box(&x).dot(black_box(&y)))
        });
        group.bench_with_input(BenchmarkId::new("matrix_vector", n), &n, |b, _| {
            b.iter(|| black_box(&matrix) * black_box(&x))
        });
        group.bench_with_input(BenchmarkId::new("outer_product", n), &n, |b, _| {
            b.iter(|| black_box(&x) * black_box(&y).transpose())
        });
    }
    group.finish();
    let mut options = Options::default();
    options.stopping.max_evaluations = Some(100);
    c.bench_function("bfgs_sphere_8", |b| {
        let initial = Vector::from_element(8, 0.5);
        b.iter(|| {
            nlopt4r::minimize(&evaluator, black_box(&initial), Method::Bfgs, &options).unwrap()
        })
    });
}
criterion_group!(benches, kernels);
criterion_main!(benches);
