//! Minimal parallel helpers over std threads.

use std::sync::atomic::{AtomicUsize, Ordering};

static THREADS: AtomicUsize = AtomicUsize::new(0);

/// Number of worker threads (set from the command line; defaults to all available cores).
pub fn threads() -> usize {
    match THREADS.load(Ordering::Relaxed) {
        0 => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        n => n,
    }
}

pub fn set_threads(n: usize) {
    THREADS.store(n, Ordering::Relaxed);
}

/// Run `f` on chunks of `0..n` of size `chunk` across all threads.
pub fn for_chunks<F>(n: usize, chunk: usize, f: F)
where
    F: Fn(std::ops::Range<usize>) + Sync,
{
    let next = AtomicUsize::new(0);
    let t = threads().min(n.div_ceil(chunk)).max(1);
    std::thread::scope(|s| {
        for _ in 0..t {
            s.spawn(|| {
                loop {
                    let start = next.fetch_add(chunk, Ordering::Relaxed);
                    if start >= n {
                        break;
                    }
                    f(start..(start + chunk).min(n));
                }
            });
        }
    });
}
