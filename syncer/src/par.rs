//! Bit-exact multi-threaded execution of the merge hot path.
//!
//! Every parallel kernel built on these helpers produces exactly the bits of
//! its serial counterpart, for every thread count and every scheduling:
//!
//! * Element-wise passes (weighted accumulation, rescaling, the outer
//!   optimizer step, finiteness checks) are split into disjoint chunks. Each
//!   element still sees the same operations in the same order (for example
//!   the learners of an average are added in the same sequence), so the
//!   split is invisible in the result.
//! * Floating-point reductions (L2 norms, dot products) are never split.
//!   Each one runs as a single sequential loop on one thread, exactly as in
//!   the serial code; parallelism comes from running independent
//!   reductions (different tensors, different learners, the separate norms
//!   of HeLoCo) at the same time.
//!
//! Work is pulled from a shared queue, so which thread runs a task depends
//! on timing, but tasks write disjoint outputs and never share an
//! accumulator, so that never affects the values.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use anyhow::{bail, Result};

/// Element-wise chunk length (f32 values).
pub const CHUNK: usize = 1 << 16;

/// Environment variable read when `--merge-threads` is not given.
pub const THREADS_ENV: &str = "YETO_SYNCER_MERGE_THREADS";

/// Upper bound of the automatic thread count. The merge is memory-bandwidth
/// bound, so more threads than this rarely help and only steal cores from
/// the network runtime.
const AUTO_MAX_THREADS: usize = 32;

static DEFAULT_THREADS: AtomicUsize = AtomicUsize::new(1);

/// Thread count new `GlobalState`s start with (1 unless the binary set it).
pub fn default_threads() -> usize {
    DEFAULT_THREADS.load(Ordering::Relaxed).max(1)
}

pub fn set_default_threads(threads: usize) {
    DEFAULT_THREADS.store(threads.max(1), Ordering::Relaxed);
}

/// Automatic thread count: the available cores, capped.
pub fn auto_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, AUTO_MAX_THREADS)
}

/// Resolve the merge thread count from the CLI value, else the environment.
/// Absent or 0 selects `auto_threads`.
pub fn resolve_threads(cli: Option<usize>, env: Option<&str>) -> Result<usize> {
    let requested = match (cli, env.map(str::trim).filter(|s| !s.is_empty())) {
        (Some(n), _) => n,
        (None, Some(text)) => match text.parse::<usize>() {
            Ok(n) => n,
            Err(_) => bail!("{THREADS_ENV} must be a non-negative integer, got {text:?}"),
        },
        (None, None) => 0,
    };
    Ok(if requested == 0 { auto_threads() } else { requested })
}

/// Run `f` on every task using up to `threads` threads. With one thread (or
/// one task) the tasks run in order on the caller's thread.
pub fn run_tasks<T: Send>(threads: usize, tasks: Vec<T>, f: impl Fn(T) + Sync) {
    let threads = threads.min(tasks.len());
    if threads <= 1 {
        tasks.into_iter().for_each(f);
        return;
    }
    let queue = Mutex::new(tasks.into_iter());
    let (queue, f) = (&queue, &f);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(move || loop {
                // Release the lock before running the task.
                let next = queue.lock().unwrap_or_else(|e| e.into_inner()).next();
                match next {
                    Some(task) => f(task),
                    None => break,
                }
            });
        }
    });
}

/// Split `buf` into consecutive mutable slices of the given lengths.
pub fn split_lengths<'a>(mut buf: &'a mut [f32], lengths: &[usize]) -> Vec<&'a mut [f32]> {
    let mut out = Vec::with_capacity(lengths.len());
    for &n in lengths {
        let (head, tail) = std::mem::take(&mut buf).split_at_mut(n);
        out.push(head);
        buf = tail;
    }
    out
}

/// Whether any value is NaN or infinite (same answer as the serial
/// `iter().any(|v| !v.is_finite())`).
pub fn any_non_finite(threads: usize, values: &[f32]) -> bool {
    if threads <= 1 {
        return values.iter().any(|v| !v.is_finite());
    }
    let found = AtomicBool::new(false);
    run_tasks(threads, values.chunks(CHUNK).collect(), |chunk| {
        if !found.load(Ordering::Relaxed) && chunk.iter().any(|v| !v.is_finite()) {
            found.store(true, Ordering::Relaxed);
        }
    });
    found.into_inner()
}

/// `dst.copy_from_slice(src)` in parallel chunks.
pub fn copy(threads: usize, dst: &mut [f32], src: &[f32]) {
    let tasks: Vec<_> = dst.chunks_mut(CHUNK).zip(src.chunks(CHUNK)).collect();
    run_tasks(threads, tasks, |(d, s)| d.copy_from_slice(s));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_prefers_cli_then_env_then_auto() {
        assert_eq!(resolve_threads(Some(3), Some("7")).unwrap(), 3);
        assert_eq!(resolve_threads(None, Some("7")).unwrap(), 7);
        assert_eq!(resolve_threads(None, Some(" ")).unwrap(), auto_threads());
        assert_eq!(resolve_threads(Some(0), None).unwrap(), auto_threads());
        assert_eq!(resolve_threads(None, Some("0")).unwrap(), auto_threads());
        assert!(resolve_threads(None, Some("x")).is_err());
    }

    #[test]
    fn run_tasks_visits_every_task_once() {
        for threads in [1, 2, 7] {
            let hits: Vec<AtomicUsize> = (0..100).map(|_| AtomicUsize::new(0)).collect();
            run_tasks(threads, (0..100).collect(), |i: usize| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
        }
    }

    #[test]
    fn split_and_non_finite() {
        let mut buf = vec![0.0f32; 10];
        let parts = split_lengths(&mut buf, &[3, 0, 7]);
        assert_eq!(parts.iter().map(|p| p.len()).collect::<Vec<_>>(), [3, 0, 7]);
        let mut values = vec![1.0f32; 3 * CHUNK + 5];
        assert!(!any_non_finite(4, &values));
        values[2 * CHUNK + 1] = f32::NAN;
        assert!(any_non_finite(4, &values));
        values[2 * CHUNK + 1] = f32::NEG_INFINITY;
        assert!(any_non_finite(4, &values));
    }
}
