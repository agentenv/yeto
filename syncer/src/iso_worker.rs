//! Iso spectrum-flattening backends.
//!
//! The scalar backend is the small-matrix reference implementation in
//! `merge.rs`. Production matrices use a bounded pool of persistent
//! Python/Torch workers: one process per configured device and one complete
//! row-major f32 matrix per job.

use std::collections::VecDeque;
use std::fmt;
use std::io::{BufReader, Read, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::str::FromStr;
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{anyhow, bail, Context, Result};
use tokio::sync::{oneshot, Notify, OwnedSemaphorePermit, Semaphore};

use crate::merge;

const MAGIC: &[u8; 8] = b"YETOISO1";
const VERSION: u32 = 1;
const REQUEST_FLATTEN: u32 = 1;
const RESPONSE_OK: u32 = 0;
const HEADER_LEN: usize = 48;
const MAX_ERROR_BYTES: u64 = 64 * 1024;
const LEGACY_QUEUE_CAPACITY: usize = 1;
// A shell argument cannot contain NUL, so this cannot collide with a legacy
// `--iso-worker-device` value. It lets the typed pooled constructor traverse
// the existing three-field config/GlobalState API without changing main.rs.
const POOL_SPEC_PREFIX: &str = "\0YETO_ISO_POOL_V1\0";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum IsoBackendKind {
    Scalar = 0,
    TorchSvd = 1,
}

impl IsoBackendKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::TorchSvd => "torch-svd",
        }
    }

    pub fn from_checkpoint(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Scalar),
            1 => Ok(Self::TorchSvd),
            other => bail!("checkpoint has unknown iso backend id {other}"),
        }
    }
}

impl fmt::Display for IsoBackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for IsoBackendKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "scalar" => Ok(Self::Scalar),
            "torch-svd" => Ok(Self::TorchSvd),
            other => bail!("--iso-backend must be 'scalar' or 'torch-svd', got {other:?}"),
        }
    }
}

/// Backend selection shared with the existing CLI.
///
/// `device` remains as the source-compatible single-worker setting. New
/// callers use [`IsoBackend::start_pool`] (or [`Self::start_pool`]) to pass
/// an explicit ordered device vector and bounded queue capacity.
#[derive(Clone, Debug)]
pub struct IsoBackendConfig {
    pub kind: IsoBackendKind,
    pub python: PathBuf,
    pub device: String,
}

impl Default for IsoBackendConfig {
    fn default() -> Self {
        Self {
            kind: IsoBackendKind::Scalar,
            python: PathBuf::from("python3"),
            device: "cuda:0".to_owned(),
        }
    }
}

impl IsoBackendConfig {
    /// Attach typed pool settings while retaining the source-compatible
    /// three-field config consumed by `GlobalState`.
    pub fn with_pool(mut self, devices: Vec<String>, queue_capacity: usize) -> Result<Self> {
        self.device = encode_pool_spec(&devices, queue_capacity)?;
        Ok(self)
    }

    pub fn start_pool(&self, devices: Vec<String>, queue_capacity: usize) -> Result<IsoBackend> {
        IsoBackend::start_pool(self, devices, queue_capacity)
    }

    fn pool_options(&self) -> Result<(Vec<String>, usize)> {
        match decode_pool_spec(&self.device)? {
            Some(options) => Ok(options),
            None => Ok((vec![self.device.clone()], LEGACY_QUEUE_CAPACITY)),
        }
    }
}

#[derive(Clone)]
pub enum IsoBackend {
    Scalar,
    TorchSvd(TorchIsoPool),
}

impl IsoBackend {
    /// Source-compatible legacy entry point: one process on `config.device`.
    pub fn start(config: &IsoBackendConfig) -> Result<Self> {
        Self::start_with_factory(config, Arc::new(ProcessWorkerFactory))
    }

    fn start_with_factory(
        config: &IsoBackendConfig,
        factory: Arc<dyn WorkerFactory>,
    ) -> Result<Self> {
        let (devices, queue_capacity) = config.pool_options()?;
        match config.kind {
            IsoBackendKind::Scalar => Ok(Self::Scalar),
            IsoBackendKind::TorchSvd => Ok(Self::TorchSvd(TorchIsoPool::start_with_factory(
                config.python.clone(),
                devices,
                queue_capacity,
                factory,
            )?)),
        }
    }

    /// Start exactly one independent persistent worker per device. Device
    /// order is retained as worker identity; it is never sorted or deduped.
    pub fn start_pool(
        config: &IsoBackendConfig,
        devices: Vec<String>,
        queue_capacity: usize,
    ) -> Result<Self> {
        match config.kind {
            IsoBackendKind::Scalar => Ok(Self::Scalar),
            IsoBackendKind::TorchSvd => Ok(Self::TorchSvd(TorchIsoPool::start(
                config.python.clone(),
                devices,
                queue_capacity,
            )?)),
        }
    }

    pub fn kind(&self) -> IsoBackendKind {
        match self {
            Self::Scalar => IsoBackendKind::Scalar,
            Self::TorchSvd(_) => IsoBackendKind::TorchSvd,
        }
    }

    /// Compatibility path for synchronous callers. Production Torch callers
    /// should use `flatten_owned`, which never blocks a Tokio executor thread.
    pub fn flatten(&mut self, matrix: &mut [f32], rows: usize, cols: usize) -> Result<()> {
        validate_shape(matrix.len(), rows, cols, self.kind())?;
        match self {
            Self::Scalar => merge::iso_flatten_spectrum(matrix, rows, cols),
            Self::TorchSvd(pool) => {
                let pool = pool.clone();
                let input = matrix.to_vec();
                let result = std::thread::spawn(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .context("build Torch Iso compatibility runtime")?;
                    runtime.block_on(pool.flatten_owned(input, rows, cols))
                })
                .join()
                .map_err(|_| anyhow!("Torch Iso compatibility thread panicked"))??;
                matrix.copy_from_slice(&result);
            }
        }
        validate_finite(matrix, self.kind())
    }

    /// Queue one owned complete matrix. Queue capacity applies asynchronous
    /// backpressure, while process protocol I/O stays on persistent OS threads.
    pub async fn flatten_owned(
        &self,
        mut matrix: Vec<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        validate_shape(matrix.len(), rows, cols, self.kind())?;
        match self {
            Self::Scalar => {
                matrix = tokio::task::spawn_blocking(move || {
                    merge::iso_flatten_spectrum(&mut matrix, rows, cols);
                    matrix
                })
                .await
                .context("scalar Iso worker task panicked")?;
                validate_finite(&matrix, self.kind())?;
                Ok(matrix)
            }
            Self::TorchSvd(pool) => pool.flatten_owned(matrix, rows, cols).await,
        }
    }

    /// Wait for all accepted work and report any persistent pool poison.
    pub async fn drain(&self) -> Result<()> {
        match self {
            Self::Scalar => Ok(()),
            Self::TorchSvd(pool) => pool.drain().await,
        }
    }
}

fn validate_pool_options(devices: &[String], queue_capacity: usize) -> Result<()> {
    if devices.is_empty() {
        bail!("Torch Iso worker pool requires at least one device");
    }
    if let Some((index, _)) = devices
        .iter()
        .enumerate()
        .find(|(_, device)| device.is_empty())
    {
        bail!("Torch Iso worker device {index} cannot be empty");
    }
    if queue_capacity == 0 {
        bail!("Torch Iso worker queue capacity must be greater than zero");
    }
    Ok(())
}

fn encode_pool_spec(devices: &[String], queue_capacity: usize) -> Result<String> {
    validate_pool_options(devices, queue_capacity)?;
    let mut encoded = format!("{POOL_SPEC_PREFIX}{queue_capacity}:{}:", devices.len());
    for device in devices {
        encoded.push_str(&device.len().to_string());
        encoded.push(':');
        encoded.push_str(device);
    }
    Ok(encoded)
}

fn decode_pool_spec(encoded: &str) -> Result<Option<(Vec<String>, usize)>> {
    let Some(mut rest) = encoded.strip_prefix(POOL_SPEC_PREFIX) else {
        return Ok(None);
    };
    fn take_usize<'a>(rest: &mut &'a str, label: &str) -> Result<usize> {
        let (value, tail) = rest
            .split_once(':')
            .with_context(|| format!("malformed Torch Iso pool spec: missing {label}"))?;
        *rest = tail;
        value
            .parse::<usize>()
            .with_context(|| format!("malformed Torch Iso pool spec {label}"))
    }
    let queue_capacity = take_usize(&mut rest, "queue capacity")?;
    let device_count = take_usize(&mut rest, "device count")?;
    let mut devices = Vec::with_capacity(device_count);
    for index in 0..device_count {
        let len = take_usize(&mut rest, &format!("device {index} length"))?;
        let bytes = rest.as_bytes();
        let device = bytes
            .get(..len)
            .context("malformed Torch Iso pool spec: truncated device")?;
        devices.push(
            std::str::from_utf8(device)
                .context("malformed Torch Iso pool spec: device is not UTF-8")?
                .to_owned(),
        );
        rest = std::str::from_utf8(&bytes[len..])
            .context("malformed Torch Iso pool spec: invalid trailing UTF-8")?;
    }
    if !rest.is_empty() {
        bail!("malformed Torch Iso pool spec: trailing bytes");
    }
    validate_pool_options(&devices, queue_capacity)?;
    Ok(Some((devices, queue_capacity)))
}

fn validate_shape(len: usize, rows: usize, cols: usize, backend: IsoBackendKind) -> Result<()> {
    if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(len) {
        bail!("iso backend {backend} received shape {rows}x{cols} for {len} values");
    }
    Ok(())
}

fn validate_finite(matrix: &[f32], backend: IsoBackendKind) -> Result<()> {
    if matrix.iter().any(|value| !value.is_finite()) {
        bail!("iso backend {backend} returned non-finite values");
    }
    Ok(())
}

trait MatrixWorker: Send + 'static {
    fn flatten(&mut self, matrix: &mut [f32], rows: usize, cols: usize) -> Result<()>;
}

trait WorkerAbort: Send + Sync + 'static {
    fn abort(&self);
}

struct StartedWorker {
    worker: Box<dyn MatrixWorker>,
    abort: Arc<dyn WorkerAbort>,
}

trait WorkerFactory: Send + Sync + 'static {
    fn start(&self, python: &Path, device: &str) -> Result<StartedWorker>;
}

struct ProcessWorkerFactory;

impl WorkerFactory for ProcessWorkerFactory {
    fn start(&self, python: &Path, device: &str) -> Result<StartedWorker> {
        let worker = TorchIsoProcess::spawn(python, device)?;
        let abort = Arc::new(ProcessAbort {
            child: worker.child.clone(),
        });
        Ok(StartedWorker {
            worker: Box::new(worker),
            abort,
        })
    }
}

#[derive(Default)]
struct AbortRegistry {
    handles: Mutex<Vec<Arc<dyn WorkerAbort>>>,
}

impl AbortRegistry {
    fn register(&self, handle: Arc<dyn WorkerAbort>) {
        self.handles
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(handle);
    }

    fn abort_all(&self) {
        let handles = self
            .handles
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        for handle in handles {
            handle.abort();
        }
    }
}

#[derive(Clone)]
pub struct TorchIsoPool {
    inner: Arc<PoolInner>,
}

impl TorchIsoPool {
    fn start(python: PathBuf, devices: Vec<String>, queue_capacity: usize) -> Result<Self> {
        Self::start_with_factory(
            python,
            devices,
            queue_capacity,
            Arc::new(ProcessWorkerFactory),
        )
    }

    fn start_with_factory(
        python: PathBuf,
        devices: Vec<String>,
        queue_capacity: usize,
        factory: Arc<dyn WorkerFactory>,
    ) -> Result<Self> {
        if !cfg!(target_endian = "little") {
            bail!("torch-svd iso backend requires a little-endian host");
        }
        validate_pool_options(&devices, queue_capacity)?;

        let state = Arc::new(PoolState::default());
        let queue = Arc::new(WorkQueue::new(queue_capacity));
        let aborts = Arc::new(AbortRegistry::default());
        let (startup_tx, startup_rx) = std::sync::mpsc::channel();
        let mut threads: Vec<std::thread::JoinHandle<()>> = Vec::with_capacity(devices.len());

        for (worker_index, device) in devices.into_iter().enumerate() {
            let worker_python = python.clone();
            let worker_factory = factory.clone();
            let worker_state = state.clone();
            let worker_queue = queue.clone();
            let worker_aborts = aborts.clone();
            let worker_startup = startup_tx.clone();
            let thread_name = format!("yeto-iso-worker-{worker_index}");
            let handle = match std::thread::Builder::new()
                .name(thread_name)
                .spawn(move || {
                    let mut startup_reported = false;
                    let outcome = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
                        let started = worker_factory
                            .start(&worker_python, &device)
                            .with_context(|| {
                                format!(
                                    "Torch Iso worker {worker_index} ({device}) startup failed"
                                )
                            })?;
                        worker_aborts.register(started.abort);
                        if worker_startup.send(Ok(())).is_err() {
                            return Ok(());
                        }
                        startup_reported = true;
                        let mut worker = started.worker;
                        run_worker(
                            worker_index,
                            &device,
                            worker.as_mut(),
                            &worker_queue,
                            &worker_state,
                            &worker_aborts,
                        );
                        Ok(())
                    }));
                    let message = match outcome {
                        Ok(Ok(())) => return,
                        Ok(Err(error)) => format!("{error:#}"),
                        Err(_) => format!(
                            "Torch Iso worker {worker_index} ({device}) panicked outside protocol handling"
                        ),
                    };
                    fail_pool(&worker_state, &worker_queue, &worker_aborts, &message);
                    if !startup_reported {
                        let _ = worker_startup.send(Err(message));
                    }
                }) {
                Ok(handle) => handle,
                Err(error) => {
                    let message = format!("spawn Torch Iso worker thread {worker_index}: {error}");
                    fail_pool(&state, &queue, &aborts, &message);
                    for thread in threads {
                        let _ = thread.join();
                    }
                    bail!(message);
                }
            };
            threads.push(handle);
        }
        drop(startup_tx);

        let mut startup_error = None;
        for _ in 0..threads.len() {
            match startup_rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    startup_error.get_or_insert(error);
                }
                Err(_) => {
                    startup_error.get_or_insert_with(|| {
                        "Torch Iso worker startup channel closed unexpectedly".to_owned()
                    });
                }
            }
        }
        if let Some(error) = startup_error {
            fail_pool(&state, &queue, &aborts, &error);
            for thread in threads {
                let _ = thread.join();
            }
            bail!(error);
        }

        Ok(Self {
            inner: Arc::new(PoolInner {
                queue,
                state,
                aborts,
                threads: Mutex::new(Some(threads)),
            }),
        })
    }

    pub async fn flatten_owned(
        &self,
        matrix: Vec<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        validate_shape(matrix.len(), rows, cols, IsoBackendKind::TorchSvd)?;
        let reply = self.submit(matrix, rows, cols).await?;
        match reply.await {
            Ok(Ok(matrix)) => {
                validate_finite(&matrix, IsoBackendKind::TorchSvd)?;
                Ok(matrix)
            }
            Ok(Err(message)) => Err(anyhow!(message)),
            Err(_) => {
                Err(anyhow!(self.inner.state.poison_message().unwrap_or_else(
                    || "Torch Iso worker reply channel closed".to_owned()
                )))
            }
        }
    }

    async fn submit(
        &self,
        matrix: Vec<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<oneshot::Receiver<std::result::Result<Vec<f32>, String>>> {
        self.inner.state.ensure_healthy()?;
        let permit = self
            .inner
            .queue
            .slots
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| anyhow!(self.closed_message()))?;
        self.inner.state.ensure_healthy()?;
        let outstanding = self.inner.state.begin_job()?;
        let (reply, receiver) = oneshot::channel();
        self.inner.queue.enqueue(Job {
            matrix,
            rows,
            cols,
            reply: Some(reply),
            _outstanding: outstanding,
            queue_permit: Some(permit),
        })?;
        Ok(receiver)
    }

    pub async fn drain(&self) -> Result<()> {
        self.inner.state.drain().await
    }

    fn closed_message(&self) -> String {
        self.inner
            .state
            .poison_message()
            .unwrap_or_else(|| "Torch Iso worker pool is closed".to_owned())
    }

    #[cfg(test)]
    fn queued_jobs(&self) -> usize {
        self.inner.queue.len()
    }
}

struct PoolInner {
    queue: Arc<WorkQueue>,
    state: Arc<PoolState>,
    aborts: Arc<AbortRegistry>,
    threads: Mutex<Option<Vec<std::thread::JoinHandle<()>>>>,
}

impl Drop for PoolInner {
    fn drop(&mut self) {
        self.queue.close_and_discard();
        self.aborts.abort_all();
        let threads = self
            .threads
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .unwrap_or_default();
        for thread in threads {
            let _ = thread.join();
        }
    }
}

#[derive(Default)]
struct PoolStatus {
    outstanding: usize,
    poison: Option<String>,
}

#[derive(Default)]
struct PoolState {
    status: Mutex<PoolStatus>,
    changed: Notify,
}

impl PoolState {
    fn begin_job(self: &Arc<Self>) -> Result<OutstandingJob> {
        let mut status = self.lock_status();
        if let Some(error) = &status.poison {
            bail!(error.clone());
        }
        status.outstanding = status
            .outstanding
            .checked_add(1)
            .context("Torch Iso outstanding-job counter overflow")?;
        Ok(OutstandingJob {
            state: self.clone(),
        })
    }

    fn finish_job(&self) {
        let mut status = self.lock_status();
        debug_assert!(status.outstanding > 0);
        status.outstanding = status.outstanding.saturating_sub(1);
        drop(status);
        self.changed.notify_waiters();
    }

    fn poison(&self, message: &str) {
        let mut status = self.lock_status();
        if status.poison.is_none() {
            status.poison = Some(message.to_owned());
        }
        drop(status);
        self.changed.notify_waiters();
    }

    fn ensure_healthy(&self) -> Result<()> {
        if let Some(error) = self.poison_message() {
            bail!(error);
        }
        Ok(())
    }

    fn poison_message(&self) -> Option<String> {
        self.lock_status().poison.clone()
    }

    async fn drain(&self) -> Result<()> {
        loop {
            // Register before checking the count to avoid a lost final wakeup.
            let changed = self.changed.notified();
            let (outstanding, poison) = {
                let status = self.lock_status();
                (status.outstanding, status.poison.clone())
            };
            if outstanding == 0 {
                if let Some(error) = poison {
                    bail!(error);
                }
                return Ok(());
            }
            changed.await;
        }
    }

    fn lock_status(&self) -> std::sync::MutexGuard<'_, PoolStatus> {
        self.status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

struct OutstandingJob {
    state: Arc<PoolState>,
}

impl Drop for OutstandingJob {
    fn drop(&mut self) {
        self.state.finish_job();
    }
}

struct Job {
    matrix: Vec<f32>,
    rows: usize,
    cols: usize,
    reply: Option<oneshot::Sender<std::result::Result<Vec<f32>, String>>>,
    _outstanding: OutstandingJob,
    queue_permit: Option<OwnedSemaphorePermit>,
}

struct QueueStatus {
    jobs: VecDeque<Job>,
    closed: bool,
}

struct WorkQueue {
    status: Mutex<QueueStatus>,
    available: Condvar,
    slots: Arc<Semaphore>,
}

impl WorkQueue {
    fn new(capacity: usize) -> Self {
        Self {
            status: Mutex::new(QueueStatus {
                jobs: VecDeque::new(),
                closed: false,
            }),
            available: Condvar::new(),
            slots: Arc::new(Semaphore::new(capacity)),
        }
    }

    fn enqueue(&self, job: Job) -> Result<()> {
        let mut status = self
            .status
            .lock()
            .map_err(|_| anyhow!("Torch Iso queue mutex is poisoned"))?;
        if status.closed {
            bail!("Torch Iso worker pool is closed");
        }
        status.jobs.push_back(job);
        drop(status);
        self.available.notify_one();
        Ok(())
    }

    fn pop(&self) -> Option<Job> {
        let mut status = self
            .status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        loop {
            if status.closed {
                return None;
            }
            if let Some(mut job) = status.jobs.pop_front() {
                // The capacity bounds queued matrices, not in-flight work.
                drop(job.queue_permit.take());
                return Some(job);
            }
            status = self
                .available
                .wait(status)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    fn close_and_discard(&self) {
        self.slots.close();
        let mut status = self
            .status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        status.closed = true;
        status.jobs.clear();
        drop(status);
        self.available.notify_all();
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .jobs
            .len()
    }
}

fn run_worker(
    worker_index: usize,
    device: &str,
    worker: &mut dyn MatrixWorker,
    queue: &WorkQueue,
    state: &PoolState,
    aborts: &AbortRegistry,
) {
    while let Some(mut job) = queue.pop() {
        if let Some(error) = state.poison_message() {
            send_job_result(&mut job, Err(error));
            break;
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            worker.flatten(&mut job.matrix, job.rows, job.cols)
        }));
        let result = match result {
            Ok(Ok(())) => {
                if job.matrix.iter().any(|value| !value.is_finite()) {
                    Err(format!(
                        "Torch Iso worker {worker_index} ({device}) returned non-finite values"
                    ))
                } else if let Some(error) = state.poison_message() {
                    Err(error)
                } else {
                    Ok(std::mem::take(&mut job.matrix))
                }
            }
            Ok(Err(error)) => Err(format!(
                "Torch Iso worker {worker_index} ({device}) failed: {error:#}"
            )),
            Err(_) => Err(format!(
                "Torch Iso worker {worker_index} ({device}) panicked while processing a job"
            )),
        };

        match result {
            Ok(matrix) => send_job_result(&mut job, Ok(matrix)),
            Err(error) => {
                fail_pool(state, queue, aborts, &error);
                send_job_result(&mut job, Err(state.poison_message().unwrap_or(error)));
                break;
            }
        }
    }
}

fn fail_pool(state: &PoolState, queue: &WorkQueue, aborts: &AbortRegistry, error: &str) {
    state.poison(error);
    queue.close_and_discard();
    // Killing every registered child makes blocking protocol reads return.
    // This is what guarantees an explicit drain cannot hang behind another
    // worker after the pool has already become unusable.
    aborts.abort_all();
}

fn send_job_result(job: &mut Job, result: std::result::Result<Vec<f32>, String>) {
    if let Some(reply) = job.reply.take() {
        let _ = reply.send(result);
    }
}

struct TorchIsoProcess {
    child: Arc<SharedChild>,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_request_id: u64,
    failed: bool,
}

impl TorchIsoProcess {
    fn spawn(python: &Path, device: &str) -> Result<Self> {
        if device.is_empty() {
            bail!("--iso-worker-device cannot be empty");
        }
        let child = Command::new(python)
            .args(["-m", "yeto.iso_worker", "--device", device])
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("spawn torch Iso worker via {}", python.display()))?;
        let child = Arc::new(SharedChild {
            child: Mutex::new(child),
        });
        let (stdin, stdout) = {
            let mut process = child
                .child
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let stdin = process
                .stdin
                .take()
                .context("torch Iso worker has no stdin")?;
            let stdout = process
                .stdout
                .take()
                .context("torch Iso worker has no stdout")?;
            (stdin, stdout)
        };
        let mut worker = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_request_id: 1,
            failed: false,
        };

        // A real request is also the readiness/device/protocol handshake.
        let mut probe = [1.0f32];
        worker
            .flatten(&mut probe, 1, 1)
            .context("torch Iso worker startup probe failed")?;
        if probe[0].to_bits() != 1.0f32.to_bits() {
            bail!("torch Iso worker startup probe returned {}", probe[0]);
        }
        Ok(worker)
    }

    fn flatten_inner(&mut self, matrix: &mut [f32], rows: usize, cols: usize) -> Result<()> {
        let request_id = self.next_request_id;
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .context("torch Iso worker request id overflow")?;
        let rows = u64::try_from(rows).context("iso rows do not fit u64")?;
        let cols = u64::try_from(cols).context("iso cols do not fit u64")?;
        let payload_len = u64::try_from(matrix.len())
            .context("iso matrix length does not fit u64")?
            .checked_mul(4)
            .context("iso request payload length overflow")?;
        let header = encode_header(REQUEST_FLATTEN, request_id, rows, cols, payload_len);
        self.stdin
            .write_all(&header)
            .context("write torch Iso request header")?;
        self.stdin
            .write_all(f32_bytes(matrix))
            .context("write torch Iso request payload")?;
        self.stdin.flush().context("flush torch Iso request")?;

        let mut response = [0u8; HEADER_LEN];
        self.stdout
            .read_exact(&mut response)
            .context("read torch Iso response header")?;
        let decoded = decode_header(&response)?;
        if decoded.request_id != request_id {
            bail!(
                "torch Iso response request id {} != {request_id}",
                decoded.request_id
            );
        }
        if decoded.rows != rows || decoded.cols != cols {
            bail!(
                "torch Iso response shape {}x{} != {rows}x{cols}",
                decoded.rows,
                decoded.cols
            );
        }
        if decoded.code != RESPONSE_OK {
            if decoded.payload_len > MAX_ERROR_BYTES {
                bail!(
                    "torch Iso worker error {} has oversized diagnostic payload {}",
                    decoded.code,
                    decoded.payload_len
                );
            }
            let mut message = vec![0u8; decoded.payload_len as usize];
            self.stdout
                .read_exact(&mut message)
                .context("read torch Iso error payload")?;
            bail!(
                "torch Iso worker error {}: {}",
                decoded.code,
                String::from_utf8_lossy(&message)
            );
        }
        if decoded.payload_len != payload_len {
            bail!(
                "torch Iso response payload length {} != {payload_len}",
                decoded.payload_len
            );
        }
        self.stdout
            .read_exact(f32_bytes_mut(matrix))
            .context("read torch Iso response payload")?;
        Ok(())
    }
}

impl MatrixWorker for TorchIsoProcess {
    fn flatten(&mut self, matrix: &mut [f32], rows: usize, cols: usize) -> Result<()> {
        if self.failed {
            bail!("torch Iso worker is poisoned after an earlier protocol failure");
        }
        let result = self.flatten_inner(matrix, rows, cols);
        if result.is_err() {
            self.failed = true;
            self.child.abort();
        }
        result
    }
}

impl Drop for TorchIsoProcess {
    fn drop(&mut self) {
        self.child.abort_and_wait();
    }
}

struct SharedChild {
    child: Mutex<Child>,
}

impl SharedChild {
    fn abort(&self) {
        let _ = self
            .child
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .kill();
    }

    fn abort_and_wait(&self) {
        let mut child = self
            .child
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _ = child.kill();
        let _ = child.wait();
    }
}

impl Drop for SharedChild {
    fn drop(&mut self) {
        let child = self
            .child
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _ = child.kill();
        let _ = child.wait();
    }
}

struct ProcessAbort {
    child: Arc<SharedChild>,
}

impl WorkerAbort for ProcessAbort {
    fn abort(&self) {
        self.child.abort();
    }
}

struct Header {
    code: u32,
    request_id: u64,
    rows: u64,
    cols: u64,
    payload_len: u64,
}

fn encode_header(
    code: u32,
    request_id: u64,
    rows: u64,
    cols: u64,
    payload_len: u64,
) -> [u8; HEADER_LEN] {
    let mut out = [0u8; HEADER_LEN];
    out[0..8].copy_from_slice(MAGIC);
    out[8..12].copy_from_slice(&VERSION.to_le_bytes());
    out[12..16].copy_from_slice(&code.to_le_bytes());
    out[16..24].copy_from_slice(&request_id.to_le_bytes());
    out[24..32].copy_from_slice(&rows.to_le_bytes());
    out[32..40].copy_from_slice(&cols.to_le_bytes());
    out[40..48].copy_from_slice(&payload_len.to_le_bytes());
    out
}

fn decode_header(value: &[u8; HEADER_LEN]) -> Result<Header> {
    if &value[0..8] != MAGIC {
        bail!("bad torch Iso response magic");
    }
    let version = u32::from_le_bytes(value[8..12].try_into().unwrap());
    if version != VERSION {
        bail!("torch Iso response version {version} != {VERSION}");
    }
    Ok(Header {
        code: u32::from_le_bytes(value[12..16].try_into().unwrap()),
        request_id: u64::from_le_bytes(value[16..24].try_into().unwrap()),
        rows: u64::from_le_bytes(value[24..32].try_into().unwrap()),
        cols: u64::from_le_bytes(value[32..40].try_into().unwrap()),
        payload_len: u64::from_le_bytes(value[40..48].try_into().unwrap()),
    })
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn f32_bytes_mut(values: &mut [f32]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::time::Duration;

    struct TestControl {
        started: mpsc::Sender<u32>,
        released: Mutex<HashSet<u32>>,
        release_changed: Condvar,
        fail: Mutex<HashSet<u32>>,
        devices: Mutex<Vec<String>>,
        aborted: AtomicBool,
    }

    impl TestControl {
        fn new(started: mpsc::Sender<u32>) -> Self {
            Self {
                started,
                released: Mutex::new(HashSet::new()),
                release_changed: Condvar::new(),
                fail: Mutex::new(HashSet::new()),
                devices: Mutex::new(Vec::new()),
                aborted: AtomicBool::new(false),
            }
        }

        fn release(&self, id: u32) {
            self.released.lock().unwrap().insert(id);
            self.release_changed.notify_all();
        }

        fn fail(&self, id: u32) {
            self.fail.lock().unwrap().insert(id);
            self.release(id);
        }

        fn wait_for_release(&self, id: u32) -> bool {
            let mut released = self.released.lock().unwrap();
            while !released.contains(&id) && !self.aborted.load(Ordering::SeqCst) {
                released = self.release_changed.wait(released).unwrap();
            }
            self.aborted.load(Ordering::SeqCst)
        }
    }

    struct TestFactory {
        control: Arc<TestControl>,
    }

    impl WorkerFactory for TestFactory {
        fn start(&self, _python: &Path, device: &str) -> Result<StartedWorker> {
            self.control.devices.lock().unwrap().push(device.to_owned());
            Ok(StartedWorker {
                worker: Box::new(TestWorker {
                    control: self.control.clone(),
                }),
                abort: Arc::new(TestAbort {
                    control: self.control.clone(),
                }),
            })
        }
    }

    struct TestAbort {
        control: Arc<TestControl>,
    }

    impl WorkerAbort for TestAbort {
        fn abort(&self) {
            self.control.aborted.store(true, Ordering::SeqCst);
            self.control.release_changed.notify_all();
        }
    }

    struct TestWorker {
        control: Arc<TestControl>,
    }

    impl MatrixWorker for TestWorker {
        fn flatten(&mut self, matrix: &mut [f32], _rows: usize, _cols: usize) -> Result<()> {
            let id = matrix[0] as u32;
            self.control.started.send(id).unwrap();
            if self.control.wait_for_release(id) {
                bail!("injected abort for job {id}");
            }
            if self.control.fail.lock().unwrap().contains(&id) {
                bail!("injected failure for job {id}");
            }
            for value in matrix {
                *value += 1000.0;
            }
            Ok(())
        }
    }

    fn test_pool(
        devices: Vec<&str>,
        queue_capacity: usize,
    ) -> (TorchIsoPool, Arc<TestControl>, mpsc::Receiver<u32>) {
        let (started, started_rx) = mpsc::channel();
        let control = Arc::new(TestControl::new(started));
        let pool = TorchIsoPool::start_with_factory(
            PathBuf::from("unused-python"),
            devices.into_iter().map(str::to_owned).collect(),
            queue_capacity,
            Arc::new(TestFactory {
                control: control.clone(),
            }),
        )
        .unwrap();
        (pool, control, started_rx)
    }

    fn recv_started(rx: &mpsc::Receiver<u32>) -> u32 {
        rx.recv_timeout(Duration::from_secs(2))
            .expect("worker did not start job")
    }

    #[test]
    fn header_roundtrip_is_exact() {
        let encoded = encode_header(7, 11, 13, 17, 19);
        assert_eq!(&encoded[..8], b"YETOISO1");
        let decoded = decode_header(&encoded).unwrap();
        assert_eq!(decoded.code, 7);
        assert_eq!(decoded.request_id, 11);
        assert_eq!(decoded.rows, 13);
        assert_eq!(decoded.cols, 17);
        assert_eq!(decoded.payload_len, 19);
    }

    #[test]
    fn backend_names_parse_strictly() {
        assert_eq!(
            "scalar".parse::<IsoBackendKind>().unwrap(),
            IsoBackendKind::Scalar
        );
        assert_eq!(
            "torch-svd".parse::<IsoBackendKind>().unwrap(),
            IsoBackendKind::TorchSvd
        );
        assert!("fast-ish".parse::<IsoBackendKind>().is_err());
    }

    #[test]
    fn typed_pool_config_reaches_backend_start_without_legacy_fallback() {
        let (started, _) = mpsc::channel();
        let control = Arc::new(TestControl::new(started));
        let config = IsoBackendConfig {
            kind: IsoBackendKind::TorchSvd,
            python: PathBuf::from("unused-python"),
            device: "legacy-cuda:99".to_owned(),
        }
        .with_pool(vec!["cuda:0".to_owned(), "cuda:3".to_owned()], 7)
        .unwrap();
        let backend = IsoBackend::start_with_factory(
            &config,
            Arc::new(TestFactory {
                control: control.clone(),
            }),
        )
        .unwrap();
        let IsoBackend::TorchSvd(pool) = backend else {
            panic!("expected Torch pool");
        };
        let mut devices = control.devices.lock().unwrap().clone();
        devices.sort();
        assert_eq!(devices, vec!["cuda:0", "cuda:3"]);
        assert_eq!(pool.inner.queue.slots.available_permits(), 7);
    }

    #[test]
    fn every_configured_device_starts_one_independent_worker() {
        let (pool, control, _) = test_pool(
            vec![
                "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7",
            ],
            8,
        );
        let mut actual = control.devices.lock().unwrap().clone();
        actual.sort();
        let expected: Vec<_> = (0..8).map(|index| format!("cuda:{index}")).collect();
        assert_eq!(actual, expected);
        drop(pool);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn jobs_run_concurrently_and_can_finish_out_of_order() {
        let (pool, control, started) = test_pool(vec!["cuda:0", "cuda:1"], 2);
        let first = {
            let pool = pool.clone();
            tokio::spawn(async move { pool.flatten_owned(vec![1.0], 1, 1).await })
        };
        let second = {
            let pool = pool.clone();
            tokio::spawn(async move { pool.flatten_owned(vec![2.0], 1, 1).await })
        };
        let ids = [recv_started(&started), recv_started(&started)];
        assert_eq!(
            ids.into_iter().collect::<HashSet<_>>(),
            HashSet::from([1, 2])
        );

        control.release(2);
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), second)
                .await
                .unwrap()
                .unwrap()
                .unwrap(),
            vec![1002.0]
        );
        assert!(!first.is_finished());
        control.release(1);
        assert_eq!(first.await.unwrap().unwrap(), vec![1001.0]);
        pool.drain().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn queue_capacity_applies_async_backpressure() {
        let (pool, control, started) = test_pool(vec!["cuda:0"], 1);
        let first_reply = {
            let pool = pool.clone();
            tokio::spawn(async move { pool.submit(vec![1.0], 1, 1).await })
                .await
                .unwrap()
                .unwrap()
        };
        assert_eq!(recv_started(&started), 1);
        let second_reply = {
            let pool = pool.clone();
            tokio::spawn(async move { pool.submit(vec![2.0], 1, 1).await })
                .await
                .unwrap()
                .unwrap()
        };
        assert_eq!(pool.queued_jobs(), 1);

        let third = {
            let pool = pool.clone();
            tokio::spawn(async move { pool.submit(vec![3.0], 1, 1).await })
        };
        tokio::task::yield_now().await;
        assert!(!third.is_finished());
        assert_eq!(pool.queued_jobs(), 1);

        control.release(1);
        assert_eq!(first_reply.await.unwrap().unwrap(), vec![1001.0]);
        assert_eq!(recv_started(&started), 2);
        let third_reply = tokio::time::timeout(Duration::from_secs(2), third)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        control.release(2);
        assert_eq!(second_reply.await.unwrap().unwrap(), vec![1002.0]);
        assert_eq!(recv_started(&started), 3);
        control.release(3);
        assert_eq!(third_reply.await.unwrap().unwrap(), vec![1003.0]);
        pool.drain().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn one_worker_failure_poisons_subsequent_work_and_drain() {
        let (pool, control, started) = test_pool(vec!["cuda:0", "cuda:1"], 2);
        let blocked = {
            let pool = pool.clone();
            tokio::spawn(async move { pool.flatten_owned(vec![6.0], 1, 1).await })
        };
        control.fail(7);
        let failure = {
            let pool = pool.clone();
            tokio::spawn(async move { pool.flatten_owned(vec![7.0], 1, 1).await })
        };
        let ids = [recv_started(&started), recv_started(&started)];
        assert_eq!(
            ids.into_iter().collect::<HashSet<_>>(),
            HashSet::from([6, 7])
        );
        let error = failure.await.unwrap().unwrap_err().to_string();
        assert!(error.contains("injected failure"), "{error}");
        let blocked_error = tokio::time::timeout(Duration::from_secs(2), blocked)
            .await
            .expect("other in-flight worker was not aborted")
            .unwrap()
            .unwrap_err()
            .to_string();
        assert!(
            blocked_error.contains("injected failure"),
            "{blocked_error}"
        );

        let error = pool
            .flatten_owned(vec![8.0], 1, 1)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("injected failure"), "{error}");
        let error = tokio::time::timeout(Duration::from_secs(2), pool.drain())
            .await
            .expect("poisoned pool drain hung")
            .unwrap_err()
            .to_string();
        assert!(error.contains("injected failure"), "{error}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn drain_waits_for_every_outstanding_job() {
        let (pool, control, started) = test_pool(vec!["cuda:0"], 1);
        let work = {
            let pool = pool.clone();
            tokio::spawn(async move { pool.flatten_owned(vec![9.0], 1, 1).await })
        };
        assert_eq!(recv_started(&started), 9);
        let drain = {
            let pool = pool.clone();
            tokio::spawn(async move { pool.drain().await })
        };
        tokio::task::yield_now().await;
        assert!(!drain.is_finished());

        control.release(9);
        assert_eq!(work.await.unwrap().unwrap(), vec![1009.0]);
        tokio::time::timeout(Duration::from_secs(2), drain)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }

    #[test]
    fn pool_configuration_rejects_empty_devices_and_zero_capacity() {
        let (started, _) = mpsc::channel();
        let control = Arc::new(TestControl::new(started));
        let factory: Arc<dyn WorkerFactory> = Arc::new(TestFactory { control });
        assert!(TorchIsoPool::start_with_factory(
            PathBuf::from("unused"),
            Vec::new(),
            1,
            factory.clone(),
        )
        .is_err());
        assert!(TorchIsoPool::start_with_factory(
            PathBuf::from("unused"),
            vec!["cuda:0".to_owned()],
            0,
            factory,
        )
        .is_err());
    }
}
