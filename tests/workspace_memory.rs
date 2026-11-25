use nlopt4r::{Method, Options, Vector, minimize};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct TrackingAllocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static LARGEST: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) }
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            LARGEST.fetch_max(new_size, Ordering::Relaxed);
        }
        unsafe { System.realloc(p, layout, new_size) }
    }
}
#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[test]
fn gradient_and_limited_memory_methods_do_not_allocate_dense_hessians() {
    for n in [512, 2048] {
        let p = (|x: &Vector| 0.5 * x.norm_squared(), |x: &Vector| x.clone());
        let x = Vector::from_element(n, 1.0);
        for method in [Method::SteepestDescent, Method::FletcherReeves] {
            LARGEST.store(0, Ordering::SeqCst);
            TRACK.store(true, Ordering::SeqCst);
            let result = minimize(&p, &x, method, &Options::default());
            TRACK.store(false, Ordering::SeqCst);
            assert!(result.unwrap().converged());
            assert!(
                LARGEST.load(Ordering::SeqCst) <= 8 * n * 8,
                "a dense workspace was allocated"
            );
        }
        let weights = Vector::from_fn(n, |i, _| 1.0 + (i % 31) as f64);
        let p = (
            |x: &Vector| 0.5 * x.component_mul(&weights).dot(x),
            |x: &Vector| x.component_mul(&weights),
        );
        LARGEST.store(0, Ordering::SeqCst);
        TRACK.store(true, Ordering::SeqCst);
        let result = nlopt4r::local::lbfgs(
            &p,
            &x,
            &nlopt4r::local::LbfgsOptions {
                history_size: 7,
                ..Default::default()
            },
        );
        TRACK.store(false, Ordering::SeqCst);
        assert!(result.unwrap().converged());
        assert!(
            LARGEST.load(Ordering::SeqCst) <= 8 * n * 8,
            "L-BFGS allocated a dense matrix"
        );
    }
}
