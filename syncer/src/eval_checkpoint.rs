//! Evaluation checkpoints (`--checkpoint-at-steps`): the global state at
//! chosen steps, written while the scheduler keeps merging.
//!
//! A step `t` is a global step: one fragment merge, the `step=` of the
//! "outer step" log line and the unit of `--total-steps`.  When step `t` of
//! the list has committed, the scheduler takes the consistent cut at `t`
//! (`GlobalState::checkpoint_snapshot`, no model copy) and a blocking task
//! writes it to `<checkpoint-path>.step-<t>`: the bytes a regular checkpoint
//! written at `t` holds.  Rounds keep merging and broadcasting meanwhile; a
//! commit copies a fragment only while the writer still holds it (the writer
//! releases each fragment once written, see `CheckpointSnapshot::persist`).
//!
//! * At most one write runs at a time.  A step that comes due while the
//!   previous write runs waits for it; the scheduler stalls for that time.
//! * The terminal cut waits for the running write before the final
//!   checkpoint is written, so the two never share the disk.
//! * A failed write is logged and training goes on: these files are outputs,
//!   not recovery state.  `--resume` reads only `--checkpoint-path`; a
//!   resumed run skips the listed steps its cut has already passed.
//!
//! Log lines (fixed format; head scripts key on them):
//!
//! ```text
//! eval checkpoint started step=<t> path=<p>
//! eval checkpoint written step=<t> path=<p> seconds=<s> bytes=<n> copied_fragments=<k>
//! eval checkpoint failed step=<t> path=<p> error=<e>
//! ```
//!
//! `<p>` is `--checkpoint-path` with `.step-<t>` appended.  The writer
//! renames the finished, fsynced `<p>.tmp` to `<p>`, so `<p>` is complete
//! once it exists and durable once "written" is logged.  A run that resumes
//! before `t` writes step `t` again and replaces `<p>` atomically.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use tracing::{error, info};

use crate::state::{CheckpointSnapshot, GlobalState};

/// `<checkpoint>.step-<step>`.
pub fn eval_checkpoint_path(checkpoint: &Path, step: u64) -> PathBuf {
    let mut name = checkpoint.as_os_str().to_owned();
    name.push(format!(".step-{step}"));
    PathBuf::from(name)
}

/// `<path>.tmp`: the writer's temporary, next to the file it becomes.
fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".tmp");
    PathBuf::from(name)
}

/// Listed steps are positive, strictly increasing and below `total_steps`:
/// the final checkpoint at `--checkpoint-path` already holds the last step.
pub fn validate_steps(steps: &[u64], total_steps: u64) -> Result<()> {
    if steps.first() == Some(&0) || steps.windows(2).any(|pair| pair[0] >= pair[1]) {
        bail!("--checkpoint-at-steps must be positive and strictly increasing, got {steps:?}");
    }
    if let Some(&last) = steps.last() {
        if last >= total_steps {
            bail!(
                "--checkpoint-at-steps {last} must be below --total-steps {total_steps}: \
                 the final checkpoint at --checkpoint-path holds the last step"
            );
        }
    }
    Ok(())
}

/// Test hook run on the writer thread before a write starts.
#[cfg(test)]
pub type BeforeWrite = std::sync::Arc<dyn Fn(u64) + Send + Sync>;

/// The scheduler's evaluation checkpoint writer; inert without listed steps.
pub struct EvalCheckpoints {
    /// Listed steps not reached yet, ascending, with their files.
    due: VecDeque<(u64, PathBuf)>,
    /// The write in flight.
    running: Option<RunningWrite>,
    #[cfg(test)]
    before_write: Option<BeforeWrite>,
}

struct RunningWrite {
    step: u64,
    task: tokio::task::JoinHandle<()>,
}

impl EvalCheckpoints {
    /// The writer for the (validated) `steps`, next to `checkpoint`, for a
    /// scheduler whose cut is at `global_step`.  Steps the cut has already
    /// passed (a resumed run) are skipped.
    pub fn new(steps: &[u64], checkpoint: Option<&Path>, global_step: u64) -> Result<Self> {
        let mut due = VecDeque::with_capacity(steps.len());
        if !steps.is_empty() {
            let checkpoint = checkpoint.context("--checkpoint-at-steps requires --checkpoint-path")?;
            for &step in steps {
                let path = eval_checkpoint_path(checkpoint, step);
                if step > global_step {
                    due.push_back((step, path));
                    continue;
                }
                info!(
                    step,
                    resumed_step = global_step,
                    path = %path.display(),
                    exists = path.exists(),
                    "eval checkpoint skipped: the resumed cut is past it"
                );
            }
        }
        Ok(Self {
            due,
            running: None,
            #[cfg(test)]
            before_write: None,
        })
    }

    #[cfg(test)]
    pub fn with_before_write(mut self, hook: BeforeWrite) -> Self {
        self.before_write = Some(hook);
        self
    }

    /// Call after every commit, with `st` at the cut just committed.  At a
    /// listed step: wait for a running write, then start this step's.
    pub async fn after_commit(&mut self, st: &GlobalState) -> Result<()> {
        let step = st.global_step;
        if self.due.front().map(|(due, _)| *due) != Some(step) {
            return Ok(());
        }
        let (_, path) = self.due.pop_front().context("eval checkpoint step disappeared")?;
        self.join_running(Some(step)).await?;
        let snapshot = st.checkpoint_snapshot()?;
        info!(step, path = %path.display(), "eval checkpoint started");
        #[cfg(test)]
        let before_write = self.before_write.clone();
        let task = tokio::task::spawn_blocking(move || {
            #[cfg(test)]
            if let Some(hook) = before_write {
                hook(step);
            }
            write(step, snapshot, &path);
        });
        self.running = Some(RunningWrite { step, task });
        Ok(())
    }

    /// Wait for the running write.  The terminal cut (or a learner-budget
    /// cutoff) calls this before it writes its checkpoint.
    pub async fn finish(&mut self) -> Result<()> {
        self.join_running(None).await
    }

    /// Wait for the running write.  `next` is the listed step that needs the
    /// writer, or None for the final checkpoint.
    async fn join_running(&mut self, next: Option<u64>) -> Result<()> {
        let Some(RunningWrite { step: previous, task }) = self.running.take() else {
            return Ok(());
        };
        let panicked = || format!("eval checkpoint writer for step {previous} panicked");
        if task.is_finished() {
            return task.await.with_context(panicked);
        }
        match next {
            Some(step) => info!(step, previous, "eval checkpoint waiting for the previous write"),
            None => info!(step = previous, "final checkpoint waiting for eval checkpoint"),
        }
        let started = Instant::now();
        task.await.with_context(panicked)?;
        let seconds = format!("{:.1}", started.elapsed().as_secs_f64());
        match next {
            Some(step) => info!(
                step,
                previous,
                seconds = %seconds,
                "eval checkpoint waited for the previous write"
            ),
            None => info!(
                step = previous,
                seconds = %seconds,
                "final checkpoint waited for eval checkpoint"
            ),
        }
        Ok(())
    }

    /// Whether a write is in flight and has not finished.
    #[cfg(test)]
    pub fn is_writing(&self) -> bool {
        self.running.as_ref().is_some_and(|running| !running.task.is_finished())
    }
}

/// Persist `snapshot` at `path` through `<path>.tmp` and log the outcome.
fn write(step: u64, snapshot: CheckpointSnapshot, path: &Path) {
    let started = Instant::now();
    let tmp = temporary_path(path);
    match snapshot.persist(path, &tmp) {
        Ok(written) => info!(
            step,
            path = %path.display(),
            seconds = %format!("{:.1}", started.elapsed().as_secs_f64()),
            bytes = written.bytes,
            copied_fragments = written.copied_fragments,
            "eval checkpoint written"
        ),
        Err(failure) => {
            // A partial model-sized file must not stay on the head disk.
            let _ = std::fs::remove_file(&tmp);
            error!(
                step,
                path = %path.display(),
                error = %format!("{failure:#}"),
                "eval checkpoint failed"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_files_sit_next_to_the_checkpoint_apart_from_its_temporary() {
        let checkpoint = Path::new("/out/yeto-state.ckpt");
        let path = eval_checkpoint_path(checkpoint, 75);
        assert_eq!(path, Path::new("/out/yeto-state.ckpt.step-75"));
        assert_eq!(temporary_path(&path), Path::new("/out/yeto-state.ckpt.step-75.tmp"));
        // The regular checkpoint writes through `<path without extension>.tmp`.
        assert_ne!(temporary_path(&path), checkpoint.with_extension("tmp"));
        assert_eq!(eval_checkpoint_path(Path::new("state"), 3), Path::new("state.step-3"));
    }
}
