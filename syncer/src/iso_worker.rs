//! Iso spectrum-flattening backends.
//!
//! The scalar backend is the small-matrix reference implementation in
//! `merge.rs`.  Production-sized matrices can instead be sent to one
//! persistent Python/Torch worker over raw, framed stdin/stdout.  Rust keeps
//! ownership of weighted averaging and the outer optimizer; the worker sees
//! exactly one complete row-major f32 matrix at a time.

use std::fmt;
use std::io::{BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::str::FromStr;

use anyhow::{bail, Context, Result};

use crate::merge;

const MAGIC: &[u8; 8] = b"YETOISO1";
const VERSION: u32 = 1;
const REQUEST_FLATTEN: u32 = 1;
const RESPONSE_OK: u32 = 0;
const HEADER_LEN: usize = 48;
const MAX_ERROR_BYTES: u64 = 64 * 1024;

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

pub enum IsoBackend {
    Scalar,
    TorchSvd(TorchIsoWorker),
}

impl IsoBackend {
    pub fn start(config: &IsoBackendConfig) -> Result<Self> {
        match config.kind {
            IsoBackendKind::Scalar => Ok(Self::Scalar),
            IsoBackendKind::TorchSvd => Ok(Self::TorchSvd(TorchIsoWorker::spawn(
                &config.python,
                &config.device,
            )?)),
        }
    }

    pub fn kind(&self) -> IsoBackendKind {
        match self {
            Self::Scalar => IsoBackendKind::Scalar,
            Self::TorchSvd(_) => IsoBackendKind::TorchSvd,
        }
    }

    pub fn flatten(&mut self, matrix: &mut [f32], rows: usize, cols: usize) -> Result<()> {
        if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(matrix.len()) {
            bail!(
                "iso backend {} received shape {rows}x{cols} for {} values",
                self.kind(),
                matrix.len()
            );
        }
        match self {
            Self::Scalar => merge::iso_flatten_spectrum(matrix, rows, cols),
            Self::TorchSvd(worker) => worker.flatten(matrix, rows, cols)?,
        }
        if matrix.iter().any(|value| !value.is_finite()) {
            bail!("iso backend {} returned non-finite values", self.kind());
        }
        Ok(())
    }
}

pub struct TorchIsoWorker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_request_id: u64,
    failed: bool,
}

impl TorchIsoWorker {
    fn spawn(python: &std::path::Path, device: &str) -> Result<Self> {
        if !cfg!(target_endian = "little") {
            bail!("torch-svd iso backend requires a little-endian host");
        }
        if device.is_empty() {
            bail!("--iso-worker-device cannot be empty");
        }
        let mut child = Command::new(python)
            .args(["-m", "yeto.iso_worker", "--device", device])
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("spawn torch Iso worker via {}", python.display()))?;
        let stdin = child
            .stdin
            .take()
            .context("torch Iso worker has no stdin")?;
        let stdout = child
            .stdout
            .take()
            .context("torch Iso worker has no stdout")?;
        let mut worker = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_request_id: 1,
            failed: false,
        };

        // A real request is also the readiness/device/protocol handshake.  It
        // catches import errors and CUDA placement failures before the first
        // model-sized merge can commit.
        let mut probe = [1.0f32];
        worker
            .flatten(&mut probe, 1, 1)
            .context("torch Iso worker startup probe failed")?;
        if probe[0].to_bits() != 1.0f32.to_bits() {
            bail!("torch Iso worker startup probe returned {}", probe[0]);
        }
        Ok(worker)
    }

    fn flatten(&mut self, matrix: &mut [f32], rows: usize, cols: usize) -> Result<()> {
        if self.failed {
            bail!("torch Iso worker is poisoned after an earlier protocol failure");
        }
        let result = self.flatten_inner(matrix, rows, cols);
        if result.is_err() {
            self.failed = true;
            let _ = self.child.kill();
        }
        result
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

impl Drop for TorchIsoWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
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
    // Every bit pattern is valid f32 storage, alignment is only relaxed, and
    // the process is rejected above on non-little-endian hosts.
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
}
