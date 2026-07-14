//! Bounded background publication of exact stock pseudo-gradient vectors.
//!
//! This is intentionally narrower than learner capture-v2: the syncer already
//! owns the merged f32 direction, so the producer transfers that one vector to
//! a FIFO worker.  The worker writes content-bound vectors, a predecessor-
//! chained JSONL tape, and an atomic final manifest.  Any queue or I/O failure
//! is fatal to the opt-in capture run.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::thread::{self, JoinHandle};

const QUEUE_CAPACITY: usize = 8;
const ZERO_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Clone, Debug)]
pub(crate) struct StockVectorTapeConfig {
    pub root: PathBuf,
    pub capture_session_uuid: String,
    pub expected_layout_sha256: Option<String>,
    pub run_config_sha256: String,
    pub expected_records: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct StockVectorTapeRow {
    pub commit_seq: u64,
    pub step: u64,
    pub fragment: usize,
    pub fragment_version_before: u64,
    pub fragment_version_after: u64,
    pub responders: Vec<(u32, u64)>,
    pub vector: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StockVectorTapeCompletion {
    pub records: u64,
    pub vector_bytes: u64,
    pub tape_sha256: String,
    pub ledger_head: String,
    pub manifest_path: PathBuf,
    pub manifest_sha256: String,
}

pub(crate) fn write_initial_state_manifest(
    path: &Path,
    capture_enabled: bool,
    layout_sha256: &str,
    initial_state_sha256: &str,
) -> Result<String> {
    for (name, digest) in [
        ("layout_sha256", layout_sha256),
        ("initial_state_sha256", initial_state_sha256),
    ] {
        validate_sha256(name, digest)?;
    }
    if path.exists() || PathBuf::from(format!("{}.sha256", path.display())).exists() {
        bail!("stock-shadow initial-state receipt path is not fresh");
    }
    let parent = path
        .parent()
        .context("initial-state receipt has no parent")?;
    if !parent.is_dir() {
        bail!("initial-state receipt parent does not exist");
    }
    let manifest = json!({
        "schema": "cplg_stock_shadow_initial_state_v1",
        "capture_enabled": capture_enabled,
        "layout_sha256": layout_sha256,
        "initial_state_sha256": initial_state_sha256,
    });
    let mut raw = serde_json::to_vec_pretty(&manifest)?;
    raw.push(b'\n');
    write_atomic(path, &raw)?;
    let digest = sha256_file(path)?;
    let basename = path
        .file_name()
        .context("initial-state receipt has no basename")?
        .to_string_lossy();
    write_atomic(
        &PathBuf::from(format!("{}.sha256", path.display())),
        format!("{digest}  {basename}\n").as_bytes(),
    )?;
    Ok(digest)
}

pub(crate) fn write_completion_manifest(
    path: &Path,
    capture_enabled: bool,
    layout_sha256: &str,
    initial_state_sha256: &str,
    interval_duration_ns: u64,
    commits: u64,
) -> Result<String> {
    for (name, digest) in [
        ("layout_sha256", layout_sha256),
        ("initial_state_sha256", initial_state_sha256),
    ] {
        validate_sha256(name, digest)?;
    }
    if interval_duration_ns == 0 || commits == 0 {
        bail!("stock-shadow completion requires positive duration and commits");
    }
    if path.exists() || PathBuf::from(format!("{}.sha256", path.display())).exists() {
        bail!("stock-shadow completion receipt path is not fresh");
    }
    let parent = path.parent().context("completion receipt has no parent")?;
    if !parent.is_dir() {
        bail!("completion receipt parent does not exist");
    }
    let manifest = json!({
        "schema": "cplg_stock_shadow_completion_v1",
        "capture_enabled": capture_enabled,
        "layout_sha256": layout_sha256,
        "initial_state_sha256": initial_state_sha256,
        "interval_scope": "post_global_initialization_to_durable_vector_writer_close",
        "interval_start_monotonic_ns": 0,
        "interval_end_monotonic_ns": interval_duration_ns,
        "commits": commits,
    });
    let mut raw = serde_json::to_vec_pretty(&manifest)?;
    raw.push(b'\n');
    write_atomic(path, &raw)?;
    let digest = sha256_file(path)?;
    let basename = path
        .file_name()
        .context("completion receipt has no basename")?
        .to_string_lossy();
    write_atomic(
        &PathBuf::from(format!("{}.sha256", path.display())),
        format!("{digest}  {basename}\n").as_bytes(),
    )?;
    Ok(digest)
}

enum Message {
    Row(StockVectorTapeRow),
    Finish {
        accepted_items: u64,
        accepted_bytes: u64,
        complete: bool,
    },
}

pub(crate) struct StockVectorTapeWriter {
    sender: Option<SyncSender<Message>>,
    worker: Option<JoinHandle<Result<StockVectorTapeCompletion>>>,
    accepted_items: u64,
    accepted_bytes: u64,
}

impl StockVectorTapeWriter {
    pub(crate) fn create(
        config: StockVectorTapeConfig,
        layout_sha256: String,
        initial_state_sha256: String,
    ) -> Result<Self> {
        validate_config(&config, &layout_sha256, &initial_state_sha256)?;
        std::fs::create_dir(&config.root).with_context(|| {
            format!(
                "create fresh stock-vector capture root {}",
                config.root.display()
            )
        })?;
        std::fs::create_dir(config.root.join("vectors"))?;
        let (sender, receiver) = sync_channel(QUEUE_CAPACITY);
        let worker = thread::Builder::new()
            .name("cplg-stock-vector-writer".to_owned())
            .spawn(move || worker_main(config, layout_sha256, initial_state_sha256, receiver))
            .context("spawn stock-vector writer")?;
        Ok(Self {
            sender: Some(sender),
            worker: Some(worker),
            accepted_items: 0,
            accepted_bytes: 0,
        })
    }

    pub(crate) fn submit(&mut self, row: StockVectorTapeRow) -> Result<()> {
        if row.vector.is_empty() || row.vector.iter().any(|value| !value.is_finite()) {
            bail!("stock-vector capture requires a nonempty finite vector");
        }
        let bytes = u64::try_from(row.vector.len())?
            .checked_mul(4)
            .context("stock-vector byte count overflow")?;
        self.sender
            .as_ref()
            .context("stock-vector writer is closed")?
            .send(Message::Row(row))
            .map_err(|_| anyhow::anyhow!("stock-vector writer stopped before accepting row"))?;
        self.accepted_items = self
            .accepted_items
            .checked_add(1)
            .context("accepted item count overflow")?;
        self.accepted_bytes = self
            .accepted_bytes
            .checked_add(bytes)
            .context("accepted byte count overflow")?;
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<StockVectorTapeCompletion> {
        self.send_finish(true)?;
        self.join_worker()
    }

    fn send_finish(&mut self, complete: bool) -> Result<()> {
        let sender = self
            .sender
            .take()
            .context("stock-vector writer is closed")?;
        sender
            .send(Message::Finish {
                accepted_items: self.accepted_items,
                accepted_bytes: self.accepted_bytes,
                complete,
            })
            .map_err(|_| anyhow::anyhow!("stock-vector writer stopped before finalization"))
    }

    fn join_worker(&mut self) -> Result<StockVectorTapeCompletion> {
        self.worker
            .take()
            .context("stock-vector writer has no worker")?
            .join()
            .map_err(|_| anyhow::anyhow!("stock-vector writer thread panicked"))?
    }
}

impl Drop for StockVectorTapeWriter {
    fn drop(&mut self) {
        if self.sender.is_some() {
            let _ = self.send_finish(false);
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn validate_config(
    config: &StockVectorTapeConfig,
    layout_sha256: &str,
    initial_state_sha256: &str,
) -> Result<()> {
    if config.capture_session_uuid.trim().is_empty() {
        bail!("stock-vector capture session must not be empty");
    }
    for (name, digest) in [
        ("layout_sha256", layout_sha256),
        ("initial_state_sha256", initial_state_sha256),
        ("run_config_sha256", config.run_config_sha256.as_str()),
    ] {
        validate_sha256(name, digest)?;
    }
    if config.expected_records == 0 {
        bail!("stock-vector expected record count must be positive");
    }
    if let Some(expected) = &config.expected_layout_sha256 {
        if expected != layout_sha256 {
            bail!("actual syncer layout SHA-256 does not match frozen expectation");
        }
    }
    Ok(())
}

fn validate_sha256(name: &str, digest: &str) -> Result<()> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{name} must be lowercase hexadecimal SHA-256");
    }
    Ok(())
}

fn worker_main(
    config: StockVectorTapeConfig,
    layout_sha256: String,
    initial_state_sha256: String,
    receiver: std::sync::mpsc::Receiver<Message>,
) -> Result<StockVectorTapeCompletion> {
    let tape_path = config.root.join("stock_tape.jsonl");
    let tape_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tape_path)?;
    let mut tape = BufWriter::new(tape_file);
    let mut ledger_head = ZERO_SHA256.to_owned();
    let mut records = 0u64;
    let mut vector_bytes = 0u64;
    let mut fragment_counts: BTreeMap<usize, u64> = BTreeMap::new();
    let mut finish = None;

    while let Ok(message) = receiver.recv() {
        match message {
            Message::Row(row) => {
                let expected_commit = records + 1;
                if row.commit_seq != expected_commit {
                    bail!(
                        "stock-vector commit sequence expected {expected_commit}, got {}",
                        row.commit_seq
                    );
                }
                if row.fragment_version_after <= row.fragment_version_before {
                    bail!("stock-vector fragment version did not advance");
                }
                if row.responders.is_empty()
                    || row.responders.windows(2).any(|pair| pair[0].0 >= pair[1].0)
                {
                    bail!("stock-vector responder IDs must be nonempty and sorted");
                }
                let relative = format!(
                    "vectors/commit-{:08}-fragment-{:04}.f32le",
                    row.commit_seq, row.fragment
                );
                let (vector_sha256, bytes) =
                    write_vector_atomic(&config.root, &relative, &row.vector)?;
                let mut object = BTreeMap::<String, Value>::new();
                object.insert(
                    "capture_session_uuid".to_owned(),
                    json!(config.capture_session_uuid),
                );
                object.insert("commit_seq".to_owned(), json!(row.commit_seq));
                object.insert("fragment".to_owned(), json!(row.fragment));
                object.insert(
                    "fragment_version_after".to_owned(),
                    json!(row.fragment_version_after),
                );
                object.insert(
                    "fragment_version_before".to_owned(),
                    json!(row.fragment_version_before),
                );
                object.insert("layout_sha256".to_owned(), json!(layout_sha256));
                object.insert("ledger_prev_sha256".to_owned(), json!(ledger_head));
                object.insert("merge_rule".to_owned(), json!("production_weighted_rda"));
                object.insert("numel".to_owned(), json!(row.vector.len()));
                object.insert(
                    "responders".to_owned(),
                    Value::Array(
                        row.responders
                            .iter()
                            .map(|(id, weight_bits)| {
                                json!({
                                    "id": id,
                                    "weight_f64_bits": format!("{weight_bits:016x}"),
                                })
                            })
                            .collect(),
                    ),
                );
                object.insert(
                    "run_config_sha256".to_owned(),
                    json!(config.run_config_sha256),
                );
                object.insert("schema".to_owned(), json!("cplg_stock_vector_row_v1"));
                object.insert("step".to_owned(), json!(row.step));
                object.insert("stock_f32le".to_owned(), json!(relative));
                object.insert("stock_f32le_sha256".to_owned(), json!(vector_sha256));
                object.insert("wire_dtype".to_owned(), json!("f32_le"));
                let payload = serde_json::to_vec(&object)?;
                let mut hasher = Sha256::new();
                hasher.update(&payload);
                hasher.update(b"\n");
                ledger_head = format!("{:x}", hasher.finalize());
                object.insert("ledger_sha256".to_owned(), json!(ledger_head));
                serde_json::to_writer(&mut tape, &object)?;
                tape.write_all(b"\n")?;
                tape.flush()?;
                records += 1;
                vector_bytes = vector_bytes
                    .checked_add(bytes)
                    .context("stock-vector byte total overflow")?;
                *fragment_counts.entry(row.fragment).or_default() += 1;
            }
            Message::Finish {
                accepted_items,
                accepted_bytes,
                complete,
            } => {
                finish = Some((accepted_items, accepted_bytes, complete));
                break;
            }
        }
    }
    let (accepted_items, accepted_bytes, requested_complete) =
        finish.unwrap_or((records, vector_bytes, false));
    tape.flush()?;
    tape.get_ref().sync_all()?;
    let tape_sha256 = sha256_file(&tape_path)?;
    let complete = requested_complete
        && accepted_items == records
        && accepted_bytes == vector_bytes
        && records == config.expected_records;
    let writer = json!({
        "state": if complete { "closed" } else { "incomplete" },
        "accepted_items": accepted_items,
        "completed_items": records,
        "accepted_bytes": accepted_bytes,
        "completed_bytes": vector_bytes,
        "dropped_items": 0,
        "dropped_bytes": 0,
        "abandoned_items": accepted_items.saturating_sub(records),
        "abandoned_bytes": accepted_bytes.saturating_sub(vector_bytes),
        "pending_items": 0,
        "pending_bytes": 0,
        "error": Value::Null,
    });
    let manifest = json!({
        "schema": "cplg_stock_vector_tape_manifest_v1",
        "status": if complete { "COMPLETE" } else { "INCOMPLETE" },
        "capture_session_uuid": config.capture_session_uuid,
        "layout_sha256": layout_sha256,
        "initial_state_sha256": initial_state_sha256,
        "run_config_sha256": config.run_config_sha256,
        "expected_records": config.expected_records,
        "records": records,
        "vector_bytes": vector_bytes,
        "fragment_counts": fragment_counts,
        "stock_tape": "stock_tape.jsonl",
        "stock_tape_sha256": tape_sha256,
        "ledger_head": ledger_head,
        "writer": writer,
    });
    let manifest_path = config.root.join("stock_tape.manifest.json");
    let manifest_raw = serde_json::to_vec_pretty(&manifest)?;
    write_atomic(&manifest_path, &[manifest_raw.as_slice(), b"\n"].concat())?;
    let manifest_sha256 = sha256_file(&manifest_path)?;
    let checksum = format!(
        "{}  {}\n",
        manifest_sha256,
        manifest_path
            .file_name()
            .context("manifest has no basename")?
            .to_string_lossy()
    );
    write_atomic(
        &PathBuf::from(format!("{}.sha256", manifest_path.display())),
        checksum.as_bytes(),
    )?;
    sync_directory(&config.root)?;
    if !complete {
        bail!("stock-vector capture finalized incomplete");
    }
    Ok(StockVectorTapeCompletion {
        records,
        vector_bytes,
        tape_sha256,
        ledger_head,
        manifest_path,
        manifest_sha256,
    })
}

fn write_vector_atomic(root: &Path, relative: &str, vector: &[f32]) -> Result<(String, u64)> {
    let target = root.join(relative);
    let temporary = target.with_extension("f32le.tmp");
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let mut writer = BufWriter::new(file);
    let mut hasher = Sha256::new();
    for value in vector {
        let bytes = value.to_le_bytes();
        writer.write_all(&bytes)?;
        hasher.update(bytes);
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    std::fs::rename(&temporary, &target)?;
    sync_directory(target.parent().context("vector has no parent")?)?;
    Ok((
        format!("{:x}", hasher.finalize()),
        u64::try_from(vector.len())? * 4,
    ))
}

fn write_atomic(path: &Path, raw: &[u8]) -> Result<()> {
    let temporary = path.with_extension("tmp");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(raw)?;
    file.sync_all()?;
    std::fs::rename(&temporary, path)?;
    sync_directory(path.parent().context("atomic path has no parent")?)
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "yeto-stock-vector-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    fn config(root: PathBuf, expected_records: u64) -> StockVectorTapeConfig {
        StockVectorTapeConfig {
            root,
            capture_session_uuid: "123e4567-e89b-12d3-a456-426614174000".to_owned(),
            expected_layout_sha256: Some("1".repeat(64)),
            run_config_sha256: "2".repeat(64),
            expected_records,
        }
    }

    fn make_writer(config: StockVectorTapeConfig) -> Result<StockVectorTapeWriter> {
        StockVectorTapeWriter::create(config, "1".repeat(64), "3".repeat(64))
    }

    #[test]
    fn writes_exact_vectors_chain_and_completion_manifest() {
        let root = test_root("complete");
        std::fs::remove_dir_all(&root).ok();
        let mut writer = make_writer(config(root.clone(), 2)).unwrap();
        for commit_seq in 1..=2 {
            writer
                .submit(StockVectorTapeRow {
                    commit_seq,
                    step: commit_seq,
                    fragment: (commit_seq - 1) as usize,
                    fragment_version_before: 0,
                    fragment_version_after: commit_seq,
                    responders: vec![(0, 1.0f64.to_bits())],
                    vector: vec![commit_seq as f32, -0.0],
                })
                .unwrap();
        }
        let completion = writer.finish().unwrap();
        assert_eq!(completion.records, 2);
        assert_eq!(completion.vector_bytes, 16);
        assert_eq!(completion.ledger_head.len(), 64);
        assert_eq!(completion.manifest_sha256.len(), 64);
        let lines: Vec<Value> = std::fs::read_to_string(root.join("stock_tape.jsonl"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines[0]["ledger_prev_sha256"], ZERO_SHA256);
        assert_eq!(lines[1]["ledger_prev_sha256"], lines[0]["ledger_sha256"]);
        assert_eq!(
            std::fs::read(root.join("vectors/commit-00000001-fragment-0000.f32le")).unwrap(),
            [1.0f32.to_le_bytes(), (-0.0f32).to_le_bytes()].concat()
        );
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(root.join("stock_tape.manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["status"], "COMPLETE");
        assert_eq!(manifest["initial_state_sha256"], "3".repeat(64));
        assert_eq!(manifest["writer"]["accepted_items"], 2);
        assert_eq!(manifest["writer"]["completed_bytes"], 16);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn rejects_reused_root_and_nonfinite_vector() {
        let root = test_root("strict");
        std::fs::remove_dir_all(&root).ok();
        let mut writer = make_writer(config(root.clone(), 1)).unwrap();
        assert!(make_writer(config(root.clone(), 1)).is_err());
        assert!(writer
            .submit(StockVectorTapeRow {
                commit_seq: 1,
                step: 1,
                fragment: 0,
                fragment_version_before: 0,
                fragment_version_after: 1,
                responders: vec![(0, 1.0f64.to_bits())],
                vector: vec![f32::NAN],
            })
            .is_err());
        drop(writer);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn initial_state_receipt_is_atomic_checksummed_and_fresh() {
        let root = test_root("initial-state");
        std::fs::remove_dir_all(&root).ok();
        std::fs::create_dir(&root).unwrap();
        let path = root.join("initial_state.json");
        let digest =
            write_initial_state_manifest(&path, false, &"1".repeat(64), &"3".repeat(64)).unwrap();
        let value: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(value["capture_enabled"], false);
        assert_eq!(value["initial_state_sha256"], "3".repeat(64));
        assert_eq!(
            std::fs::read_to_string(format!("{}.sha256", path.display())).unwrap(),
            format!("{digest}  initial_state.json\n")
        );
        assert!(
            write_initial_state_manifest(&path, false, &"1".repeat(64), &"3".repeat(64),).is_err()
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn completion_receipt_freezes_unrounded_interval() {
        let root = test_root("completion");
        std::fs::remove_dir_all(&root).ok();
        std::fs::create_dir(&root).unwrap();
        let path = root.join("completion.json");
        let digest = write_completion_manifest(
            &path,
            true,
            &"1".repeat(64),
            &"3".repeat(64),
            123_456_789,
            32,
        )
        .unwrap();
        let value: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(value["interval_start_monotonic_ns"], 0);
        assert_eq!(value["interval_end_monotonic_ns"], 123_456_789);
        assert_eq!(value["commits"], 32);
        assert_eq!(digest.len(), 64);
        std::fs::remove_dir_all(root).ok();
    }
}
