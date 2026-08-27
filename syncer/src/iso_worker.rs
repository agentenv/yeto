//! Iso spectrum-flattening backends.
//!
//! The scalar backend is the small-matrix reference implementation in
//! `merge.rs`. Production matrices use a bounded pool of persistent
//! Python/Torch workers: one process per configured device and one complete
//! row-major f32 matrix per job.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::io::{BufReader, Read, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

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
const DEFAULT_STARTUP_TIMEOUT_S: u64 = 300;
// Leave one minute for drain failure propagation and child reaping while the
// entire fail-closed drain remains bounded by one hour.
const DEFAULT_REQUEST_TIMEOUT_S: u64 = 3540;
const DEFAULT_DRAIN_TIMEOUT_S: u64 = 3600;
const STARTUP_TIMEOUT_ENV: &str = "YETO_ISO_WORKER_STARTUP_TIMEOUT_S";
const REQUEST_TIMEOUT_ENV: &str = "YETO_ISO_WORKER_REQUEST_TIMEOUT_S";
const DRAIN_TIMEOUT_ENV: &str = "YETO_ISO_WORKER_DRAIN_TIMEOUT_S";
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
/// an explicit ordered device vector and bounded admitted-resident capacity.
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

    /// Admit one owned complete matrix. Capacity applies asynchronous
    /// backpressure across queued plus running work, while process protocol I/O
    /// stays on persistent OS threads.
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
    fn startup(&mut self) -> Result<()> {
        Ok(())
    }

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

struct AbortRegistry {
    handles: Mutex<Vec<Arc<dyn WorkerAbort>>>,
    aborted: AtomicBool,
}

impl Default for AbortRegistry {
    fn default() -> Self {
        Self {
            handles: Mutex::new(Vec::new()),
            aborted: AtomicBool::new(false),
        }
    }
}

impl AbortRegistry {
    fn register(&self, handle: Arc<dyn WorkerAbort>) {
        let mut handles = self
            .handles
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.aborted.load(Ordering::SeqCst) {
            drop(handles);
            handle.abort();
            return;
        }
        handles.push(handle);
    }

    fn abort_all(&self) {
        self.aborted.store(true, Ordering::SeqCst);
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

#[derive(Clone, Copy, Debug)]
struct LifecycleLimits {
    startup: Duration,
    request: Duration,
    drain: Duration,
}

impl LifecycleLimits {
    fn from_env() -> Result<Self> {
        Ok(Self {
            startup: timeout_from_env(STARTUP_TIMEOUT_ENV, DEFAULT_STARTUP_TIMEOUT_S)?,
            request: timeout_from_env(REQUEST_TIMEOUT_ENV, DEFAULT_REQUEST_TIMEOUT_S)?,
            drain: timeout_from_env(DRAIN_TIMEOUT_ENV, DEFAULT_DRAIN_TIMEOUT_S)?,
        })
    }

    #[cfg(test)]
    fn for_tests(startup: Duration, request: Duration, drain: Duration) -> Self {
        Self {
            startup,
            request,
            drain,
        }
    }
}

fn timeout_from_env(name: &str, default_seconds: u64) -> Result<Duration> {
    let raw = match std::env::var(name) {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => return Ok(Duration::from_secs(default_seconds)),
        Err(error) => return Err(anyhow!("read {name}: {error}")),
    };
    let seconds = raw
        .parse::<u64>()
        .with_context(|| format!("{name} must be a positive integer number of seconds"))?;
    if seconds == 0 {
        bail!("{name} must be greater than zero");
    }
    Ok(Duration::from_secs(seconds))
}

fn format_worker_identities(workers: &BTreeMap<usize, String>) -> String {
    workers
        .iter()
        .map(|(index, device)| format!("worker {index} ({device})"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn startup_timeout_message(timeout: Duration, pending: &BTreeMap<usize, String>) -> String {
    format!(
        "Torch Iso pool startup timed out after {:.3}s waiting for {}",
        timeout.as_secs_f64(),
        format_worker_identities(pending)
    )
}

struct DeadlineGuard {
    state: Arc<std::sync::atomic::AtomicU8>,
    wake: Option<mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl DeadlineGuard {
    fn arm(
        timeout: Duration,
        timeout_message: String,
        state: Arc<PoolState>,
        queue: Arc<WorkQueue>,
        aborts: Arc<AbortRegistry>,
        worker_abort: Arc<dyn WorkerAbort>,
    ) -> Self {
        const PENDING: u8 = 0;
        const TIMED_OUT: u8 = 2;

        let deadline_state = Arc::new(std::sync::atomic::AtomicU8::new(PENDING));
        let watchdog_state = deadline_state.clone();
        let (wake, receiver) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("yeto-iso-deadline".to_owned())
            .spawn(move || {
                if matches!(
                    receiver.recv_timeout(timeout),
                    Err(mpsc::RecvTimeoutError::Timeout)
                ) && watchdog_state
                    .compare_exchange(PENDING, TIMED_OUT, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    // Abort the timed-out direct child first, then poison and
                    // abort the rest of the pool. This breaks its blocking
                    // protocol read/write as early as possible.
                    worker_abort.abort();
                    fail_pool(&state, &queue, &aborts, &timeout_message);
                }
            })
            .expect("spawn bounded Torch Iso deadline watchdog");
        Self {
            state: deadline_state,
            wake: Some(wake),
            thread: Some(thread),
        }
    }

    fn finish(mut self) {
        const PENDING: u8 = 0;
        const COMPLETED: u8 = 1;
        let _ = self
            .state
            .compare_exchange(PENDING, COMPLETED, Ordering::SeqCst, Ordering::SeqCst);
        if let Some(wake) = self.wake.take() {
            let _ = wake.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
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
        Self::start_with_factory_and_limits(
            python,
            devices,
            queue_capacity,
            factory,
            LifecycleLimits::from_env()?,
        )
    }

    fn start_with_factory_and_limits(
        python: PathBuf,
        devices: Vec<String>,
        queue_capacity: usize,
        factory: Arc<dyn WorkerFactory>,
        limits: LifecycleLimits,
    ) -> Result<Self> {
        if !cfg!(target_endian = "little") {
            bail!("torch-svd iso backend requires a little-endian host");
        }
        validate_pool_options(&devices, queue_capacity)?;

        let state = Arc::new(PoolState::default());
        let queue = Arc::new(WorkQueue::new(queue_capacity));
        let aborts = Arc::new(AbortRegistry::default());
        let (startup_tx, startup_rx) = mpsc::channel();
        let mut threads: Vec<std::thread::JoinHandle<()>> = Vec::with_capacity(devices.len());
        let worker_identities: Vec<_> = devices
            .iter()
            .enumerate()
            .map(|(index, device)| (index, device.clone()))
            .collect();
        let startup_deadline = Instant::now() + limits.startup;

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
                        let worker_abort = started.abort;
                        worker_aborts.register(worker_abort.clone());
                        let mut worker = started.worker;
                        let startup_guard = DeadlineGuard::arm(
                            limits.startup,
                            format!(
                                "Torch Iso worker {worker_index} ({device}) startup timed out after {:.3}s",
                                limits.startup.as_secs_f64()
                            ),
                            worker_state.clone(),
                            worker_queue.clone(),
                            worker_aborts.clone(),
                            worker_abort.clone(),
                        );
                        let startup_result = worker.startup();
                        startup_guard.finish();
                        if let Some(error) = worker_state.poison_message() {
                            bail!(error);
                        }
                        startup_result.with_context(|| {
                            format!("Torch Iso worker {worker_index} ({device}) readiness probe failed")
                        })?;
                        if worker_startup
                            .send((worker_index, device.clone(), Ok(())))
                            .is_err()
                        {
                            return Ok(());
                        }
                        startup_reported = true;
                        run_worker(
                            worker.as_mut(),
                            WorkerRuntime {
                                worker_index,
                                device: device.clone(),
                                worker_abort,
                                queue: worker_queue.clone(),
                                state: worker_state.clone(),
                                aborts: worker_aborts.clone(),
                            },
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
                        let _ = worker_startup.send((worker_index, device, Err(message)));
                    }
                }) {
                Ok(handle) => handle,
                Err(error) => {
                    let message = format!("spawn Torch Iso worker thread {worker_index}: {error}");
                    fail_pool(&state, &queue, &aborts, &message);
                    // Dropping JoinHandle detaches it. The registry is already
                    // aborted, so any child registered late is killed at once.
                    drop(threads);
                    bail!(message);
                }
            };
            threads.push(handle);
        }
        drop(startup_tx);

        let mut pending: BTreeMap<usize, String> = worker_identities.into_iter().collect();
        while !pending.is_empty() {
            let remaining = startup_deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                let error = startup_timeout_message(limits.startup, &pending);
                fail_pool(&state, &queue, &aborts, &error);
                drop(threads);
                bail!(error);
            }
            match startup_rx.recv_timeout(remaining) {
                Ok((worker_index, device, Ok(()))) => {
                    if pending.remove(&worker_index).as_deref() != Some(device.as_str()) {
                        let error = format!(
                            "Torch Iso worker startup returned an unexpected identity {worker_index} ({device})"
                        );
                        fail_pool(&state, &queue, &aborts, &error);
                        drop(threads);
                        bail!(error);
                    }
                }
                Ok((worker_index, device, Err(error))) => {
                    let error = format!(
                        "Torch Iso worker {worker_index} ({device}) could not start: {error}"
                    );
                    fail_pool(&state, &queue, &aborts, &error);
                    drop(threads);
                    bail!(error);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    let error = startup_timeout_message(limits.startup, &pending);
                    fail_pool(&state, &queue, &aborts, &error);
                    drop(threads);
                    bail!(error);
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    let error = format!(
                        "Torch Iso worker startup channel closed with pending workers: {}",
                        format_worker_identities(&pending)
                    );
                    fail_pool(&state, &queue, &aborts, &error);
                    drop(threads);
                    bail!(error);
                }
            }
        }

        Ok(Self {
            inner: Arc::new(PoolInner {
                queue,
                state,
                aborts,
                threads: Mutex::new(Some(threads)),
                next_request_id: AtomicU64::new(1),
                limits,
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
        let request_id = self
            .inner
            .next_request_id
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                current.checked_add(1)
            })
            .map_err(|_| anyhow!("Torch Iso pool request id overflow"))?;
        let request_deadline = Instant::now()
            .checked_add(self.inner.limits.request)
            .context("Torch Iso request deadline exceeds platform Instant range")?;
        let admission = self
            .inner
            .state
            .begin_admission(request_id, rows, cols, matrix.len())?;
        let permit = match tokio::time::timeout(
            request_deadline.saturating_duration_since(Instant::now()),
            self.inner.queue.slots.clone().acquire_owned(),
        )
        .await
        {
            Ok(Ok(permit)) => permit,
            Ok(Err(_)) => bail!(self.closed_message()),
            Err(_) => {
                let error = format!(
                    "Torch Iso request {request_id} timed out after {:.3}s awaiting resident capacity for shape {rows}x{cols} ({} values)",
                    self.inner.limits.request.as_secs_f64(),
                    matrix.len()
                );
                fail_pool(
                    &self.inner.state,
                    &self.inner.queue,
                    &self.inner.aborts,
                    &error,
                );
                bail!(self.inner.state.poison_message().unwrap_or(error))
            }
        };
        self.inner.state.ensure_healthy()?;
        let outstanding = admission.promote()?;
        let (reply, receiver) = oneshot::channel();
        let enqueue_result = self.inner.queue.enqueue(Job {
            matrix,
            request_id,
            request_deadline,
            request_timeout: self.inner.limits.request,
            rows,
            cols,
            reply: Some(reply),
            _outstanding: outstanding,
            _resident_permit: permit,
        });
        if let Err(error) = enqueue_result {
            bail!(self
                .inner
                .state
                .poison_message()
                .unwrap_or_else(|| format!("{error:#}")));
        }
        Ok(receiver)
    }

    pub async fn drain(&self) -> Result<()> {
        self.inner.state.close_admissions();
        match tokio::time::timeout(self.inner.limits.drain, self.inner.state.drain()).await {
            Ok(result) => result,
            Err(_) => {
                let pending = self.inner.state.outstanding_summary();
                let error = format!(
                    "Torch Iso pool drain timed out after {:.3}s with {}",
                    self.inner.limits.drain.as_secs_f64(),
                    pending
                );
                fail_pool(
                    &self.inner.state,
                    &self.inner.queue,
                    &self.inner.aborts,
                    &error,
                );
                bail!(self.inner.state.poison_message().unwrap_or(error))
            }
        }
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
    next_request_id: AtomicU64,
    limits: LifecycleLimits,
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
        // Rust cannot time-bound JoinHandle::join. Direct-child abort breaks
        // worker protocol I/O and each detached worker thread then reaps its
        // own Python child. Detaching here keeps pool destruction bounded even
        // if a custom/test factory violates that cancellation contract.
        drop(threads);
    }
}

struct PoolStatus {
    admissions_closed: bool,
    pending_admissions: BTreeMap<u64, OutstandingInfo>,
    outstanding: BTreeMap<u64, OutstandingInfo>,
    poison: Option<String>,
}

impl Default for PoolStatus {
    fn default() -> Self {
        Self {
            admissions_closed: false,
            pending_admissions: BTreeMap::new(),
            outstanding: BTreeMap::new(),
            poison: None,
        }
    }
}

#[derive(Clone)]
struct OutstandingInfo {
    rows: usize,
    cols: usize,
    values: usize,
    worker: Option<(usize, String)>,
}

#[derive(Default)]
struct PoolState {
    status: Mutex<PoolStatus>,
    changed: Notify,
}

impl PoolState {
    fn begin_admission(
        self: &Arc<Self>,
        request_id: u64,
        rows: usize,
        cols: usize,
        values: usize,
    ) -> Result<AdmissionIntent> {
        let mut status = self.lock_status();
        if let Some(error) = &status.poison {
            bail!(error.clone());
        }
        if status.admissions_closed {
            bail!("Torch Iso worker pool admissions are closed for drain");
        }
        match status.pending_admissions.entry(request_id) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(OutstandingInfo {
                    rows,
                    cols,
                    values,
                    worker: None,
                });
            }
            std::collections::btree_map::Entry::Occupied(_) => {
                bail!("duplicate Torch Iso pool request id {request_id}");
            }
        }
        Ok(AdmissionIntent {
            state: self.clone(),
            request_id,
            pending: true,
        })
    }

    fn promote_admission(&self, request_id: u64) -> Result<()> {
        let mut status = self.lock_status();
        if let Some(error) = &status.poison {
            bail!(error.clone());
        }
        let info = status
            .pending_admissions
            .remove(&request_id)
            .with_context(|| format!("unknown Torch Iso admission {request_id}"))?;
        if status.outstanding.insert(request_id, info).is_some() {
            bail!("duplicate Torch Iso outstanding request id {request_id}");
        }
        Ok(())
    }

    fn cancel_admission(&self, request_id: u64) {
        let mut status = self.lock_status();
        status.pending_admissions.remove(&request_id);
        drop(status);
        self.changed.notify_waiters();
    }

    fn close_admissions(&self) {
        let mut status = self.lock_status();
        status.admissions_closed = true;
        drop(status);
        self.changed.notify_waiters();
    }

    fn mark_active(&self, request_id: u64, worker_index: usize, device: &str) -> Result<()> {
        let mut status = self.lock_status();
        let info = status
            .outstanding
            .get_mut(&request_id)
            .with_context(|| format!("unknown Torch Iso pool request {request_id}"))?;
        info.worker = Some((worker_index, device.to_owned()));
        Ok(())
    }

    fn finish_job(&self, request_id: u64) {
        let mut status = self.lock_status();
        debug_assert!(status.outstanding.contains_key(&request_id));
        status.outstanding.remove(&request_id);
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
            let (pending, outstanding, poison) = {
                let status = self.lock_status();
                (
                    status.pending_admissions.len(),
                    status.outstanding.len(),
                    status.poison.clone(),
                )
            };
            if let Some(error) = poison {
                bail!(error);
            }
            if pending == 0 && outstanding == 0 {
                return Ok(());
            }
            changed.await;
        }
    }

    fn outstanding_summary(&self) -> String {
        let status = self.lock_status();
        if status.pending_admissions.is_empty() && status.outstanding.is_empty() {
            return "no outstanding requests".to_owned();
        }
        let mut details = status
            .pending_admissions
            .iter()
            .map(|(request_id, info)| {
                format!(
                    "request {request_id} awaiting resident capacity, shape {}x{}, values {}",
                    info.rows, info.cols, info.values
                )
            })
            .collect::<Vec<_>>();
        details.extend(status
            .outstanding
            .iter()
            .map(|(request_id, info)| match &info.worker {
                Some((worker_index, device)) => format!(
                    "request {request_id} active on worker {worker_index} ({device}), shape {}x{}, values {}",
                    info.rows, info.cols, info.values
                ),
                None => format!(
                    "request {request_id} queued, shape {}x{}, values {}",
                    info.rows, info.cols, info.values
                ),
            }));
        let details = details.join("; ");
        format!(
            "{} outstanding request(s): {details}",
            status.pending_admissions.len() + status.outstanding.len()
        )
    }

    fn lock_status(&self) -> std::sync::MutexGuard<'_, PoolStatus> {
        self.status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

struct AdmissionIntent {
    state: Arc<PoolState>,
    request_id: u64,
    pending: bool,
}

impl AdmissionIntent {
    fn promote(mut self) -> Result<OutstandingJob> {
        self.state.promote_admission(self.request_id)?;
        self.pending = false;
        Ok(OutstandingJob {
            state: self.state.clone(),
            request_id: self.request_id,
        })
    }
}

impl Drop for AdmissionIntent {
    fn drop(&mut self) {
        if self.pending {
            self.state.cancel_admission(self.request_id);
        }
    }
}

struct OutstandingJob {
    state: Arc<PoolState>,
    request_id: u64,
}

impl Drop for OutstandingJob {
    fn drop(&mut self) {
        self.state.finish_job(self.request_id);
    }
}

struct Job {
    matrix: Vec<f32>,
    request_id: u64,
    request_deadline: Instant,
    request_timeout: Duration,
    rows: usize,
    cols: usize,
    reply: Option<oneshot::Sender<std::result::Result<Vec<f32>, String>>>,
    _outstanding: OutstandingJob,
    // This is intentionally retained for the full queued + running lifetime.
    // Dropping the completed/discarded Job releases resident capacity.
    _resident_permit: OwnedSemaphorePermit,
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
            if let Some(job) = status.jobs.pop_front() {
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

struct WorkerRuntime {
    worker_index: usize,
    device: String,
    worker_abort: Arc<dyn WorkerAbort>,
    queue: Arc<WorkQueue>,
    state: Arc<PoolState>,
    aborts: Arc<AbortRegistry>,
}

fn run_worker(worker: &mut dyn MatrixWorker, runtime: WorkerRuntime) {
    let WorkerRuntime {
        worker_index,
        device,
        worker_abort,
        queue,
        state,
        aborts,
    } = runtime;
    while let Some(mut job) = queue.pop() {
        if let Some(error) = state.poison_message() {
            send_job_result(&mut job, Err(error));
            break;
        }
        if let Err(error) = state.mark_active(job.request_id, worker_index, &device) {
            let error = format!(
                "Torch Iso worker {worker_index} ({device}) could not activate request {}: {error:#}",
                job.request_id
            );
            fail_pool(&state, &queue, &aborts, &error);
            send_job_result(&mut job, Err(state.poison_message().unwrap_or(error)));
            break;
        }
        let remaining = job
            .request_deadline
            .saturating_duration_since(Instant::now());
        let timeout_message = format!(
            "Torch Iso worker {worker_index} ({device}) request {} timed out after {:.3}s for shape {}x{} ({} values)",
            job.request_id,
            job.request_timeout.as_secs_f64(),
            job.rows,
            job.cols,
            job.matrix.len()
        );
        if remaining.is_zero() {
            fail_pool(&state, &queue, &aborts, &timeout_message);
            send_job_result(
                &mut job,
                Err(state.poison_message().unwrap_or(timeout_message)),
            );
            break;
        }
        let deadline = DeadlineGuard::arm(
            remaining,
            timeout_message,
            state.clone(),
            queue.clone(),
            aborts.clone(),
            worker_abort.clone(),
        );
        let result = catch_unwind(AssertUnwindSafe(|| {
            worker.flatten(&mut job.matrix, job.rows, job.cols)
        }));
        deadline.finish();
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
                "Torch Iso worker {worker_index} ({device}) request {} failed: {error:#}",
                job.request_id
            )),
            Err(_) => Err(format!(
                "Torch Iso worker {worker_index} ({device}) request {} panicked",
                job.request_id
            )),
        };

        match result {
            Ok(matrix) => {
                if let Err(error) = send_job_success_if_healthy(&mut job, &state, matrix) {
                    fail_pool(&state, &queue, &aborts, &error);
                    send_job_result(&mut job, Err(state.poison_message().unwrap_or(error)));
                    break;
                }
            }
            Err(error) => {
                fail_pool(&state, &queue, &aborts, &error);
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

fn send_job_success_if_healthy(
    job: &mut Job,
    state: &PoolState,
    matrix: Vec<f32>,
) -> std::result::Result<(), String> {
    // Poison and success delivery share this lock. A success that linearizes
    // first is complete; any request still pending when poison linearizes
    // observes the persistent first error instead of a late success.
    let status = state.lock_status();
    if let Some(error) = &status.poison {
        return Err(error.clone());
    }
    if let Some(reply) = job.reply.take() {
        let _ = reply.send(Ok(matrix));
    }
    Ok(())
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
        if python.file_name().and_then(|name| name.to_str()) == Some("docker_python_iso_worker.sh")
        {
            bail!(
                "docker_python_iso_worker.sh is not a cancellable direct Python child; run the syncer inside miles_node and pass its python3 executable"
            );
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
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_request_id: 1,
            failed: false,
        })
    }

    fn startup_probe(&mut self) -> Result<()> {
        // A real request is also the readiness/device/protocol handshake.
        let mut probe = [1.0f32];
        self.flatten(&mut probe, 1, 1)
            .context("torch Iso worker startup probe failed")?;
        if probe[0].to_bits() != 1.0f32.to_bits() {
            bail!("torch Iso worker startup probe returned {}", probe[0]);
        }
        Ok(())
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
            .with_context(|| format!("write torch Iso request {request_id} header"))?;
        self.stdin
            .write_all(f32_bytes(matrix))
            .with_context(|| format!("write torch Iso request {request_id} payload"))?;
        self.stdin
            .flush()
            .with_context(|| format!("flush torch Iso request {request_id}"))?;

        let mut response = [0u8; HEADER_LEN];
        self.stdout
            .read_exact(&mut response)
            .with_context(|| format!("read torch Iso response {request_id} header"))?;
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
                .with_context(|| format!("read torch Iso response {request_id} error payload"))?;
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
            .with_context(|| format!("read torch Iso response {request_id} payload"))?;
        Ok(())
    }
}

impl MatrixWorker for TorchIsoProcess {
    fn startup(&mut self) -> Result<()> {
        self.startup_probe()
    }

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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
        test_pool_with_limits(
            devices,
            queue_capacity,
            LifecycleLimits::for_tests(
                Duration::from_secs(2),
                Duration::from_secs(2),
                Duration::from_secs(2),
            ),
        )
    }

    fn test_pool_with_limits(
        devices: Vec<&str>,
        queue_capacity: usize,
        limits: LifecycleLimits,
    ) -> (TorchIsoPool, Arc<TestControl>, mpsc::Receiver<u32>) {
        let (started, started_rx) = mpsc::channel();
        let control = Arc::new(TestControl::new(started));
        let pool = TorchIsoPool::start_with_factory_and_limits(
            PathBuf::from("unused-python"),
            devices.into_iter().map(str::to_owned).collect(),
            queue_capacity,
            Arc::new(TestFactory {
                control: control.clone(),
            }),
            limits,
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
    async fn resident_capacity_covers_queued_and_running_jobs() {
        let (pool, control, started) = test_pool(vec!["cuda:0"], 1);
        let first_reply = {
            let pool = pool.clone();
            tokio::spawn(async move { pool.submit(vec![1.0], 1, 1).await })
                .await
                .unwrap()
                .unwrap()
        };
        assert_eq!(recv_started(&started), 1);
        let second = {
            let pool = pool.clone();
            tokio::spawn(async move { pool.submit(vec![2.0], 1, 1).await })
        };
        tokio::task::yield_now().await;
        assert!(!second.is_finished());
        assert_eq!(pool.queued_jobs(), 0);
        assert_eq!(pool.inner.queue.slots.available_permits(), 0);

        control.release(1);
        assert_eq!(first_reply.await.unwrap().unwrap(), vec![1001.0]);
        let second_reply = tokio::time::timeout(Duration::from_secs(2), second)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(recv_started(&started), 2);
        control.release(2);
        assert_eq!(second_reply.await.unwrap().unwrap(), vec![1002.0]);
        pool.drain().await.unwrap();
        assert_eq!(pool.inner.queue.slots.available_permits(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn drain_linearizes_after_existing_permit_waiter_and_closes_new_admissions() {
        let (pool, control, started) = test_pool(vec!["cuda:0"], 1);
        let first = {
            let pool = pool.clone();
            tokio::spawn(async move { pool.flatten_owned(vec![11.0], 1, 1).await })
        };
        assert_eq!(recv_started(&started), 11);
        let second = {
            let pool = pool.clone();
            tokio::spawn(async move { pool.flatten_owned(vec![12.0], 1, 1).await })
        };
        loop {
            if pool.inner.state.lock_status().pending_admissions.len() == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        let drain = {
            let pool = pool.clone();
            tokio::spawn(async move { pool.drain().await })
        };
        tokio::task::yield_now().await;
        assert!(!drain.is_finished());

        control.release(11);
        assert_eq!(first.await.unwrap().unwrap(), vec![1011.0]);
        assert_eq!(recv_started(&started), 12);
        assert!(!drain.is_finished());
        control.release(12);
        assert_eq!(second.await.unwrap().unwrap(), vec![1012.0]);
        tokio::time::timeout(Duration::from_secs(1), drain)
            .await
            .expect("drain ignored a pre-existing permit waiter")
            .unwrap()
            .unwrap();

        let error = pool
            .flatten_owned(vec![13.0], 1, 1)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("admissions are closed"), "{error}");
    }

    struct HungStartupControl {
        entered: mpsc::Sender<()>,
        aborted: AtomicBool,
        abort_count: AtomicUsize,
        changed: Condvar,
        lock: Mutex<()>,
    }

    struct NoopWorker;

    impl MatrixWorker for NoopWorker {
        fn flatten(&mut self, _matrix: &mut [f32], _rows: usize, _cols: usize) -> Result<()> {
            Ok(())
        }
    }

    struct LateFactoryControl {
        entered: mpsc::Sender<()>,
        aborted: mpsc::Sender<()>,
        released: Mutex<bool>,
        changed: Condvar,
    }

    struct LateFactory {
        control: Arc<LateFactoryControl>,
    }

    struct LateAbort {
        control: Arc<LateFactoryControl>,
    }

    impl WorkerFactory for LateFactory {
        fn start(&self, _python: &Path, _device: &str) -> Result<StartedWorker> {
            self.control.entered.send(()).unwrap();
            let mut released = self.control.released.lock().unwrap();
            while !*released {
                released = self.control.changed.wait(released).unwrap();
            }
            Ok(StartedWorker {
                worker: Box::new(NoopWorker),
                abort: Arc::new(LateAbort {
                    control: self.control.clone(),
                }),
            })
        }
    }

    impl WorkerAbort for LateAbort {
        fn abort(&self) {
            let _ = self.control.aborted.send(());
        }
    }

    #[test]
    fn hung_factory_is_bounded_and_late_child_registration_is_aborted() {
        let (entered, entered_rx) = mpsc::channel();
        let (aborted, aborted_rx) = mpsc::channel();
        let control = Arc::new(LateFactoryControl {
            entered,
            aborted,
            released: Mutex::new(false),
            changed: Condvar::new(),
        });
        let factory: Arc<dyn WorkerFactory> = Arc::new(LateFactory {
            control: control.clone(),
        });
        let (result_tx, result_rx) = mpsc::channel();
        let constructor = std::thread::spawn(move || {
            let result = TorchIsoPool::start_with_factory_and_limits(
                PathBuf::from("unused"),
                vec!["cuda:6".to_owned()],
                1,
                factory,
                LifecycleLimits::for_tests(
                    Duration::from_millis(100),
                    Duration::from_secs(2),
                    Duration::from_secs(2),
                ),
            )
            .err()
            .map(|error| error.to_string());
            let _ = result_tx.send(result);
        });
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let error = result_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("hung factory constructor exceeded outer test bound")
            .expect("hung factory unexpectedly succeeded");
        constructor.join().unwrap();
        assert!(error.contains("worker 0 (cuda:6)"), "{error}");
        assert!(error.contains("startup timed out"), "{error}");

        *control.released.lock().unwrap() = true;
        control.changed.notify_all();
        aborted_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("late child handle was not aborted after registration");
    }

    struct HungStartupFactory {
        control: Arc<HungStartupControl>,
    }

    struct HungStartupWorker {
        control: Arc<HungStartupControl>,
    }

    struct HungStartupAbort {
        control: Arc<HungStartupControl>,
    }

    impl WorkerFactory for HungStartupFactory {
        fn start(&self, _python: &Path, _device: &str) -> Result<StartedWorker> {
            Ok(StartedWorker {
                worker: Box::new(HungStartupWorker {
                    control: self.control.clone(),
                }),
                abort: Arc::new(HungStartupAbort {
                    control: self.control.clone(),
                }),
            })
        }
    }

    impl MatrixWorker for HungStartupWorker {
        fn startup(&mut self) -> Result<()> {
            self.control.entered.send(()).unwrap();
            let mut guard = self.control.lock.lock().unwrap();
            while !self.control.aborted.load(Ordering::SeqCst) {
                guard = self.control.changed.wait(guard).unwrap();
            }
            bail!("startup probe interrupted")
        }

        fn flatten(&mut self, _matrix: &mut [f32], _rows: usize, _cols: usize) -> Result<()> {
            unreachable!()
        }
    }

    impl WorkerAbort for HungStartupAbort {
        fn abort(&self) {
            self.control.abort_count.fetch_add(1, Ordering::SeqCst);
            self.control.aborted.store(true, Ordering::SeqCst);
            self.control.changed.notify_all();
        }
    }

    #[test]
    fn hung_startup_is_bounded_and_aborts_registered_worker() {
        let (entered, entered_rx) = mpsc::channel();
        let control = Arc::new(HungStartupControl {
            entered,
            aborted: AtomicBool::new(false),
            abort_count: AtomicUsize::new(0),
            changed: Condvar::new(),
            lock: Mutex::new(()),
        });
        let factory: Arc<dyn WorkerFactory> = Arc::new(HungStartupFactory {
            control: control.clone(),
        });
        let started_at = Instant::now();
        let (result_tx, result_rx) = mpsc::channel();
        let constructor = std::thread::spawn(move || {
            let result = TorchIsoPool::start_with_factory_and_limits(
                PathBuf::from("unused"),
                vec!["cuda:7".to_owned()],
                1,
                factory,
                LifecycleLimits::for_tests(
                    Duration::from_millis(100),
                    Duration::from_secs(2),
                    Duration::from_secs(2),
                ),
            )
            .err()
            .map(|error| error.to_string());
            let _ = result_tx.send(result);
        });
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let error = result_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("hung startup constructor exceeded outer test bound")
            .expect("hung startup unexpectedly succeeded");
        constructor.join().unwrap();
        assert!(started_at.elapsed() < Duration::from_secs(1));
        assert!(error.contains("worker 0 (cuda:7)"), "{error}");
        assert!(error.contains("startup"), "{error}");
        assert!(error.contains("timed out"), "{error}");
        assert!(control.abort_count.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn shared_child_kill_targets_and_reaps_direct_python_process() {
        let child = Command::new("python3")
            .args(["-c", "import time; time.sleep(60)"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("python3 is required for direct-child lifecycle test");
        let child = SharedChild {
            child: Mutex::new(child),
        };
        child.abort_and_wait();
        let status = child
            .child
            .lock()
            .unwrap()
            .try_wait()
            .unwrap()
            .expect("direct Python child was not reaped");
        assert!(!status.success());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn hung_request_times_out_with_device_request_and_shape_then_poisons() {
        let (pool, _control, started) = test_pool_with_limits(
            vec!["cuda:3"],
            1,
            LifecycleLimits::for_tests(
                Duration::from_secs(2),
                Duration::from_millis(100),
                Duration::from_secs(2),
            ),
        );
        let work = {
            let pool = pool.clone();
            tokio::spawn(async move { pool.flatten_owned(vec![42.0], 1, 1).await })
        };
        assert_eq!(recv_started(&started), 42);
        let error = tokio::time::timeout(Duration::from_secs(1), work)
            .await
            .expect("request timeout did not bound the operation")
            .unwrap()
            .unwrap_err()
            .to_string();
        assert!(error.contains("worker 0 (cuda:3)"), "{error}");
        assert!(error.contains("request 1"), "{error}");
        assert!(error.contains("shape 1x1"), "{error}");
        assert!(error.contains("timed out"), "{error}");
        let drain_error = pool.drain().await.unwrap_err().to_string();
        assert_eq!(drain_error, error);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn drain_timeout_aborts_active_request_and_returns_bounded_error() {
        let (pool, _control, started) = test_pool_with_limits(
            vec!["cuda:5"],
            2,
            LifecycleLimits::for_tests(
                Duration::from_secs(2),
                Duration::from_secs(5),
                Duration::from_millis(100),
            ),
        );
        let work = {
            let pool = pool.clone();
            tokio::spawn(async move { pool.flatten_owned(vec![55.0], 1, 1).await })
        };
        assert_eq!(recv_started(&started), 55);
        let queued = {
            let pool = pool.clone();
            tokio::spawn(async move { pool.flatten_owned(vec![56.0], 1, 1).await })
        };
        while pool.queued_jobs() != 1 {
            tokio::task::yield_now().await;
        }
        let error = tokio::time::timeout(Duration::from_secs(1), pool.drain())
            .await
            .expect("drain exceeded its external bound")
            .unwrap_err()
            .to_string();
        assert!(error.contains("drain timed out"), "{error}");
        assert!(
            error.contains("request 1 active on worker 0 (cuda:5)"),
            "{error}"
        );
        let work_error = tokio::time::timeout(Duration::from_secs(1), work)
            .await
            .expect("drain did not abort active worker")
            .unwrap()
            .unwrap_err()
            .to_string();
        assert_eq!(work_error, error);
        let queued_error = tokio::time::timeout(Duration::from_secs(1), queued)
            .await
            .expect("drain did not discard queued work")
            .unwrap()
            .unwrap_err()
            .to_string();
        assert_eq!(queued_error, error);
        assert_eq!(pool.inner.queue.slots.available_permits(), 2);
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
