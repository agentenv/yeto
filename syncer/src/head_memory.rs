//! Head memory of the strict full-parameter syncer (S27 P4): a byte model
//! for real model sizes and a scaled-down measurement on CPU.
//!
//! Test-only.  The model counts the large buffers the head holds; it leaves
//! out sockets, frame buffers and allocator slack.  The measurement replays
//! the same buffer life cycle on `GlobalState`/`FragmentFold` with a
//! per-thread heap meter, so its numbers are exact heap bytes of this
//! process, not RSS.

use crate::fold::FragmentFold;
use crate::protocol::DTYPE_F32;
use crate::state::{FragmentInfo, GlobalState, Layout, MERGE_RDA};

/// Per-thread heap meter installed as the test binary's global allocator.
/// Each test runs on its own thread, so concurrent tests do not mix.
pub mod meter {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    pub struct Meter;

    thread_local! {
        static LIVE: Cell<isize> = const { Cell::new(0) };
        static PEAK: Cell<isize> = const { Cell::new(0) };
    }

    fn note(delta: isize) {
        let _ = LIVE.try_with(|live| {
            let now = live.get() + delta;
            live.set(now);
            let _ = PEAK.try_with(|peak| peak.set(peak.get().max(now)));
        });
    }

    unsafe impl GlobalAlloc for Meter {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { System.alloc(layout) };
            if !ptr.is_null() {
                note(layout.size() as isize);
            }
            ptr
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { System.alloc_zeroed(layout) };
            if !ptr.is_null() {
                note(layout.size() as isize);
            }
            ptr
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) };
            note(-(layout.size() as isize));
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let out = unsafe { System.realloc(ptr, layout, new_size) };
            if !out.is_null() {
                note(new_size as isize - layout.size() as isize);
            }
            out
        }
    }

    /// Live heap bytes allocated (net) on this thread.
    pub fn live() -> isize {
        LIVE.with(Cell::get)
    }

    /// Reset the peak to the current live bytes; returns that baseline.
    pub fn reset_peak() -> isize {
        let now = live();
        PEAK.with(|peak| peak.set(now));
        now
    }

    /// Peak live bytes above `baseline` since the last `reset_peak`.
    pub fn peak_above(baseline: isize) -> isize {
        PEAK.with(Cell::get) - baseline
    }
}

#[global_allocator]
static METER: meter::Meter = meter::Meter;

const F32: f64 = 4.0;

/// Shape of one strict full-parameter session.
#[derive(Clone, Copy)]
struct Session {
    params: f64,
    /// Largest fragment: at most the fragment bound (a larger tensor is
    /// split into dimension-0 row slices across several fragments).
    max_fragment: f64,
    islands: f64,
    /// Fragment rounds in flight (`--pipeline`).
    pipeline: f64,
    /// RDA folds a second accumulator.
    rda: bool,
    /// bf16 downlink keeps one f32 anchor per fragment.
    downlink_anchor: bool,
}

/// Peak head bytes, buffered (S26) and streaming (S27 P4).
struct Estimate {
    resident: f64,
    rounds_buffered: f64,
    rounds_fold_in_order: f64,
    rounds_fold_worst: f64,
    async_checkpoint_buffered: f64,
    async_checkpoint_snapshot: f64,
    resume_buffered: f64,
    resume_streamed: f64,
}

impl Session {
    fn estimate(&self) -> Estimate {
        let n = self.params * F32;
        let f = self.max_fragment * F32;
        let resident = 2.0 * n + if self.downlink_anchor { n } else { 0.0 };
        let acc = if self.rda { 2.0 } else { 1.0 };
        Estimate {
            resident,
            // M decoded pushes + the reduced delta per in-flight round.
            rounds_buffered: self.pipeline * (self.islands + 1.0) * f,
            // Accumulators plus the push being folded.
            rounds_fold_in_order: self.pipeline * (acc + 1.0) * f,
            // Lowest id last: M-1 held pushes + the incoming one become the
            // accumulator, plus the RDA direction sum.
            rounds_fold_worst: self.pipeline * (self.islands + acc - 1.0) * f,
            // encode_checkpoint: params + momentum as one byte vector.
            async_checkpoint_buffered: 2.0 * n,
            // Shared snapshot; commits during the write clone their
            // fragment's params + momentum.
            async_checkpoint_snapshot: self.pipeline * 2.0 * f,
            // Whole file in memory next to the restored state.
            resume_buffered: 2.0 * n,
            resume_streamed: 0.0,
        }
    }
}

fn gib(bytes: f64) -> f64 {
    bytes / (1u64 << 30) as f64
}

/// Qwen3-class 27B dense model: 27e9 parameters; the largest tensor is
/// assumed to be a 152064 x 5120 embedding (shape not read from a config).
#[test]
fn head_memory_model_for_27b_with_two_and_four_islands() {
    println!(
        "islands | resident | rounds S26 | rounds P4 in-order/worst | \
         async ckpt S26/P4 | resume S26/P4 | peak S26 | peak P4 (worst)  [GiB]"
    );
    let mut peaks = Vec::new();
    for islands in [2.0, 4.0] {
        let session = Session {
            params: 27e9,
            max_fragment: 152_064.0 * 5120.0,
            islands,
            pipeline: 2.0,
            rda: true,
            downlink_anchor: true,
        };
        let e = session.estimate();
        let before = e.resident + e.rounds_buffered + e.async_checkpoint_buffered;
        let after = e.resident + e.rounds_fold_worst + e.async_checkpoint_snapshot;
        println!(
            "{islands} | {:.1} | {:.1} | {:.1}/{:.1} | {:.1}/{:.1} | {:.1}/{:.1} | {:.1} | {:.1}",
            gib(e.resident),
            gib(e.rounds_buffered),
            gib(e.rounds_fold_in_order),
            gib(e.rounds_fold_worst),
            gib(e.async_checkpoint_buffered),
            gib(e.async_checkpoint_snapshot),
            gib(e.resume_buffered),
            gib(e.resume_streamed),
            gib(before),
            gib(after),
        );
        assert!(after < before);
        peaks.push((before, after));
    }
    // The f32 resident state (params, momentum, downlink anchor) dominates
    // after P4 and is not reduced by it.
    let (_, after_four) = peaks[1];
    assert!(gib(after_four) > 300.0);
}

fn values(seed: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) ^ seed) as f32 * 1e-9 - 2.0)
        .collect()
}

fn scaled_state(fragments: usize, fragment_numel: usize) -> GlobalState {
    let layout = Layout {
        fragments: (0..fragments)
            .map(|_| FragmentInfo {
                merge_mode: MERGE_RDA,
                tensor_numels: vec![fragment_numel as u64],
                tensor_shapes: None,
            })
            .collect(),
    };
    let mut st = GlobalState::new(layout, 0.7, 0.9, DTYPE_F32).unwrap();
    for fid in 0..fragments {
        st.init_fragment(fid, values(fid as u32, fragment_numel)).unwrap();
    }
    st
}

/// Peak bytes above the resident state for one round, buffered.
fn buffered_round_peak(st: &GlobalState, islands: u32, numel: usize) -> isize {
    let base = meter::reset_peak();
    let pushes: Vec<Vec<f32>> = (0..islands).map(|m| values(m + 100, numel)).collect();
    let slices: Vec<&[f32]> = pushes.iter().map(Vec::as_slice).collect();
    let weights = vec![1.0; slices.len()];
    let prepared = st.prepare_merge(0, 0, &slices, &weights).unwrap();
    drop(prepared);
    drop(pushes);
    meter::peak_above(base)
}

/// Peak bytes above the resident state for one round, streaming, with the
/// pushes arriving in `arrival` order (learner ids).
fn folded_round_peak(st: &GlobalState, arrival: &[u32], numel: usize) -> isize {
    let base = meter::reset_peak();
    let roster: Vec<u32> = (0..arrival.len() as u32).collect();
    let mut fold = FragmentFold::for_equal_full_roster(&st.layout.fragments[0], &roster).unwrap();
    let mut held: Vec<(u32, Vec<f32>)> = Vec::new();
    for &id in arrival {
        held.push((id, values(id + 100, numel)));
        while let Some(next) = fold.next_learner() {
            let Some(at) = held.iter().position(|(held_id, _)| *held_id == next) else {
                break;
            };
            let (_, gradient) = held.swap_remove(at);
            fold.fold_next(next, gradient).unwrap();
        }
    }
    let prepared = st.prepare_reduced_merge(0, 0, fold.finish().unwrap()).unwrap();
    drop(prepared);
    meter::peak_above(base)
}

/// Measured on a 4M-parameter state (4 fragments x 1M, f32, RDA): the
/// streaming fold, snapshot checkpoint and streamed resume against the
/// buffered S26 paths.  Values are heap bytes in units of one fragment.
#[test]
fn head_memory_measured_scaled_down() {
    const FRAGMENTS: usize = 4;
    const NUMEL: usize = 1 << 20;
    let frag_bytes = (NUMEL * 4) as f64;
    let units = |bytes: isize| bytes as f64 / frag_bytes;

    let before_state = meter::live();
    let mut st = scaled_state(FRAGMENTS, NUMEL);
    let resident = meter::live() - before_state;
    println!("resident params+momentum: {:.2} fragments", units(resident));
    assert!((units(resident) - 2.0 * FRAGMENTS as f64).abs() < 0.01);

    for islands in [2u32, 4] {
        let ascending: Vec<u32> = (0..islands).collect();
        let descending: Vec<u32> = ascending.iter().rev().copied().collect();
        let buffered = units(buffered_round_peak(&st, islands, NUMEL));
        let in_order = units(folded_round_peak(&st, &ascending, NUMEL));
        let worst = units(folded_round_peak(&st, &descending, NUMEL));
        println!(
            "{islands} islands, one round: buffered {buffered:.2}, \
             fold in-order {in_order:.2}, fold worst {worst:.2} fragments"
        );
        // RDA: in order = 2 accumulators + 1 incoming push.
        assert!(in_order <= 3.01, "{in_order}");
        assert!(worst <= f64::from(islands) + 1.01, "{worst}");
        assert!(buffered >= f64::from(islands) + 1.0, "{buffered}");
    }

    let dir = std::env::temp_dir().join(format!("yeto-p4-headmem-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("state.ckpt");

    let base = meter::reset_peak();
    let bytes = st.encode_checkpoint().unwrap();
    drop(bytes);
    let encoded = units(meter::peak_above(base));

    // Snapshot, then commit one fragment while the "write" is pending: only
    // that fragment's params and momentum are cloned.
    let base = meter::reset_peak();
    let snapshot = st.checkpoint_snapshot().unwrap();
    let delta = st
        .prepare_merge(0, 0, &[&values(7, NUMEL)], &[1.0])
        .unwrap()
        .finish_inline()
        .unwrap();
    st.commit_merge(delta).unwrap();
    snapshot.save(&path).unwrap();
    drop(snapshot);
    let snapshot_peak = units(meter::peak_above(base));
    println!(
        "async checkpoint: encoded bytes {encoded:.2}, \
         snapshot + 1 commit during write {snapshot_peak:.2} fragments"
    );
    assert!(encoded >= 2.0 * FRAGMENTS as f64);
    // 2 cloned buffers + the commit's own delta, or the 1 MiB write buffer.
    assert!(snapshot_peak <= 3.3, "{snapshot_peak}");

    let mut resumed = scaled_state(FRAGMENTS, NUMEL);
    let base = meter::reset_peak();
    resumed.load_checkpoint(&path).unwrap();
    let load_peak = units(meter::peak_above(base));
    let file_len = std::fs::metadata(&path).unwrap().len() as f64 / frag_bytes;
    println!(
        "resume: streamed load {load_peak:.2} fragments \
         (S26 also held the {file_len:.2}-fragment file)"
    );
    // The restored copy (2 per fragment) is built before the old one drops;
    // the rest is the 1 MiB read buffer and one 64 KiB decode block.
    let read_buffers = (1 << 20) as f64 / frag_bytes + 0.05;
    assert!(load_peak <= 2.0 * FRAGMENTS as f64 + read_buffers, "{load_peak}");
    std::fs::remove_dir_all(&dir).ok();
}

/// Qwen3.8-27B at TP2: each rank owns a 124160 x 5120 vocabulary shard
/// (2.54 GB f32), above the 2 GiB fragment bound, so the layout splits it
/// into two row slices.  Every fragment is then at most the bound, and the
/// per-round head terms scale with the bound instead of the largest tensor.
#[test]
fn head_memory_model_for_27b_tp2_with_split_vocabulary() {
    const FRAGMENT_BOUND_SCALARS: f64 = (2u64 << 30) as f64 / F32;
    let vocabulary_shard = 124_160.0 * 5120.0;
    assert!(vocabulary_shard > FRAGMENT_BOUND_SCALARS);
    let slices = (vocabulary_shard / FRAGMENT_BOUND_SCALARS).ceil();
    assert_eq!(slices, 2.0);
    println!("islands | resident | rounds fold worst | async ckpt snapshot | peak  [GiB]");
    for islands in [2.0, 4.0] {
        let split = Session {
            params: 27e9,
            max_fragment: FRAGMENT_BOUND_SCALARS,
            islands,
            pipeline: 2.0,
            rda: true,
            downlink_anchor: true,
        };
        let whole = Session { max_fragment: 152_064.0 * 5120.0, ..split };
        let (e, w) = (split.estimate(), whole.estimate());
        let peak = e.resident + e.rounds_fold_worst + e.async_checkpoint_snapshot;
        println!(
            "{islands} | {:.1} | {:.1} | {:.1} | {:.1}",
            gib(e.resident),
            gib(e.rounds_fold_worst),
            gib(e.async_checkpoint_snapshot),
            gib(peak),
        );
        // Splitting leaves the resident state alone and bounds every
        // per-round term by the fragment bound.
        assert_eq!(e.resident, w.resident);
        assert!(e.rounds_fold_worst < w.rounds_fold_worst);
        assert!(gib(e.rounds_fold_worst) <= 2.0 * (islands + 1.0) * 2.0 + 1e-9);
    }
}

/// S27 P4b: per-process head bytes for 27B with the streaming weighted fold
/// (any quorum, token weights) and the head split into N syncer shards.
///
/// A shard keeps params, momentum and the bf16-downlink anchor of its own
/// fragments only (3·n/N) and runs its own pipeline of rounds; a streaming
/// round holds its accumulators (2 for RDA) plus the push being folded,
/// whatever the quorum, where the buffered merge holds every responder's
/// push plus the reduced delta.  Checkpoint and resume stream as in P4.
#[test]
fn head_memory_model_for_27b_sharded_streaming_rounds() {
    println!(
        "islands | shards | resident/shard | rounds buffered | rounds streaming | \
         peak/shard buffered | peak/shard streaming | total streaming  [GiB]"
    );
    let mut table = Vec::new();
    for islands in [2.0, 4.0] {
        for shards in [1.0, 2.0, 4.0] {
            let session = Session {
                params: 27e9,
                max_fragment: 152_064.0 * 5120.0,
                islands,
                pipeline: 2.0,
                rda: true,
                downlink_anchor: true,
            };
            let e = session.estimate();
            let f = session.max_fragment * F32;
            let resident = e.resident / shards;
            // Buffered weighted round: every push + the reduced delta.
            let rounds_buffered = e.rounds_buffered;
            // Streaming: avg + direction accumulators + the incoming push.
            let rounds_streaming = session.pipeline * 3.0 * f;
            let peak_buffered = resident + rounds_buffered + e.async_checkpoint_snapshot;
            let peak_streaming = resident + rounds_streaming + e.async_checkpoint_snapshot;
            println!(
                "{islands} | {shards} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1}",
                gib(resident),
                gib(rounds_buffered),
                gib(rounds_streaming),
                gib(peak_buffered),
                gib(peak_streaming),
                gib(peak_streaming * shards),
            );
            assert!(peak_streaming <= peak_buffered);
            table.push((islands, shards, peak_streaming));
        }
    }
    // Sharding divides the dominant resident term: with 4 islands, 4 shards
    // bring a 27B head from ~331 to ~104 GiB per process.
    let peak = |m: f64, n: f64| table.iter().find(|(a, b, _)| *a == m && *b == n).unwrap().2;
    assert!(gib(peak(4.0, 4.0)) < 110.0, "{}", gib(peak(4.0, 4.0)));
    assert!(peak(4.0, 4.0) < peak(4.0, 1.0) / 3.0);
}

/// Measured: a shard allocates params and momentum for its own fragments
/// only, and its checkpoint stores the others empty.
#[test]
fn sharded_state_holds_only_owned_fragments_measured() {
    const FRAGMENTS: usize = 6;
    const NUMEL: usize = 1 << 18;
    let frag_bytes = (NUMEL * 4) as f64;
    let layout = |_: ()| Layout {
        fragments: (0..FRAGMENTS)
            .map(|_| FragmentInfo {
                merge_mode: MERGE_RDA,
                tensor_numels: vec![NUMEL as u64],
                tensor_shapes: None,
            })
            .collect(),
    };
    let dir = std::env::temp_dir().join(format!("yeto-p4b-shard-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for shards in [1u32, 2, 3] {
        let mut total = 0.0;
        for index in 0..shards {
            let plan = crate::shard::ShardPlan::new(index, shards).unwrap();
            let before = meter::live();
            let mut st = GlobalState::new(layout(()), 0.7, 0.9, DTYPE_F32).unwrap();
            st.set_shard(plan).unwrap();
            for fid in st.owned_fragments() {
                st.init_fragment(fid, values(fid as u32, NUMEL)).unwrap();
            }
            assert!(st.all_initialized());
            let foreign = (0..FRAGMENTS).find(|fid| !st.owns_fragment(*fid));
            if let Some(fid) = foreign {
                assert!(st.init_fragment(fid, values(0, NUMEL)).is_err());
            }
            let resident = (meter::live() - before) as f64 / frag_bytes;
            let owned = st.owned_fragments().len() as f64;
            assert!((resident - 2.0 * owned).abs() < 0.05, "{resident} vs {owned}");
            total += resident;

            let path = dir.join(format!("s{shards}-{index}.ckpt"));
            st.save_checkpoint(&path).unwrap();
            let file = std::fs::metadata(&path).unwrap().len() as f64 / frag_bytes;
            assert!((file - 2.0 * owned).abs() < 0.01, "{file}");
            let mut resumed = GlobalState::new(layout(()), 0.7, 0.9, DTYPE_F32).unwrap();
            resumed.set_shard(plan).unwrap();
            resumed.load_checkpoint(&path).unwrap();
            assert!(resumed.all_initialized());
            assert_eq!(resumed.owned_fragments(), st.owned_fragments());
            if shards > 1 {
                // Another shard's checkpoint is refused.
                let other = crate::shard::ShardPlan::new((index + 1) % shards, shards).unwrap();
                let mut wrong = GlobalState::new(layout(()), 0.7, 0.9, DTYPE_F32).unwrap();
                wrong.set_shard(other).unwrap();
                assert!(wrong.load_checkpoint(&path).is_err());
            }
        }
        println!("{shards} shards: total resident {total:.2} fragments (2 per fragment)");
        assert!((total - 2.0 * FRAGMENTS as f64).abs() < 0.1);
    }
    std::fs::remove_dir_all(&dir).ok();
}
