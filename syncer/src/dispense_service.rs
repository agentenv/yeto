//! Task dispenser connections (S27 P2): the wire codec of the TASK_*
//! messages (`protocol.rs`), the dispenser shared by every connection, its
//! `task-dispense.jsonl` log and its checkpoint record.
//!
//! An island process opens one dispense connection to the actor syncer with
//! TASK_HELLO (HMAC-sealed with `TASK_HELLO_MAC_DOMAIN`), then sends
//! CLAIM / REPORT / HEARTBEAT requests; every request gets exactly one
//! answer, in order. A refusal is MSG_ERROR with the `dispense.rs` prefix
//! and closes the connection.
//!
//! Expiry needs no timer: every request first expires the silent holders,
//! so a dead island's positions return before the next claim of any island.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use tokio::io::AsyncWriteExt;
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tracing::{info, warn};

use crate::dispense::{
    Claim, DataDecl, DispenseConfig, DispenseError, Dispenser, Holder, OpenOutcome, ReportEntry, ReportKind, ReportTotals,
    StreamDecl, MAX_STREAMS, MAX_STREAM_NAME, WAIT_RETRY,
};
use crate::protocol::{
    read_frame_limited, Reader, MAGIC, MAX_TASK_REPORT, MSG_ERROR, MSG_TASK_ACK, MSG_TASK_CLAIM, MSG_TASK_GRANT,
    MSG_TASK_HEARTBEAT, MSG_TASK_REPORT, MSG_TASK_WELCOME, TASK_HELLO_MAC_DOMAIN, TASK_PROTOCOL_VERSION,
    TASK_PROTOCOL_VERSION_STREAMS,
};

pub const GRANT_STATUS_GRANT: u8 = 0;
pub const GRANT_STATUS_WAIT: u8 = 1;
pub const GRANT_STATUS_END: u8 = 2;

/// Suffix of the dispenser record next to `--checkpoint-path`.
pub const RECORD_SUFFIX: &str = ".dispense";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TaskHello {
    pub holder: Holder,
    pub data: DataDecl,
    /// The streams the island takes positions of (None: every stream).
    pub accept: Option<Vec<u16>>,
    /// The protocol version the island spoke (the WELCOME answers in it).
    pub version: u16,
}

/// The TASK_HELLO body (without the HMAC). Version 1 (P2) when the data is
/// one plain stream and the island accepts it; version 2 otherwise:
/// `u16 version | u32 learner_id | u64 incarnation | [32] fingerprint |
/// u16 n | n x (u64 rows, u64 weight, u16 name_len, name) | u16 m |
/// m x u16 stream` (`m = 0`: every stream).
#[cfg_attr(not(test), allow(dead_code))] // the island side encodes (yeto/rl/engine/task_dispense.py)
pub fn encode_task_hello(hello: &TaskHello) -> Vec<u8> {
    let mut out = Vec::with_capacity(54);
    if hello.version == TASK_PROTOCOL_VERSION {
        out.extend_from_slice(&TASK_PROTOCOL_VERSION.to_le_bytes());
        out.extend_from_slice(&hello.holder.learner_id.to_le_bytes());
        out.extend_from_slice(&hello.holder.incarnation.to_le_bytes());
        out.extend_from_slice(&hello.data.total_rows().to_le_bytes());
        out.extend_from_slice(&hello.data.fingerprint);
        return out;
    }
    out.extend_from_slice(&TASK_PROTOCOL_VERSION_STREAMS.to_le_bytes());
    out.extend_from_slice(&hello.holder.learner_id.to_le_bytes());
    out.extend_from_slice(&hello.holder.incarnation.to_le_bytes());
    out.extend_from_slice(&hello.data.fingerprint);
    out.extend_from_slice(&(hello.data.streams.len() as u16).to_le_bytes());
    for stream in &hello.data.streams {
        out.extend_from_slice(&stream.rows.to_le_bytes());
        out.extend_from_slice(&stream.weight.to_le_bytes());
        out.extend_from_slice(&(stream.name.len() as u16).to_le_bytes());
        out.extend_from_slice(stream.name.as_bytes());
    }
    let accept = hello.accept.as_deref().unwrap_or(&[]);
    out.extend_from_slice(&(accept.len() as u16).to_le_bytes());
    for id in accept {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out
}

pub fn decode_task_hello(body: &[u8]) -> Result<TaskHello> {
    let mut r = Reader(body);
    let version = r.u16()?;
    let learner_id = r.u32()?;
    let incarnation = r.u64()?;
    let holder = Holder { learner_id, incarnation };
    if version == TASK_PROTOCOL_VERSION {
        let rows = r.u64()?;
        let fingerprint: [u8; 32] = r.take(32)?.try_into()?;
        ensure!(r.remaining() == 0, "trailing bytes in TASK_HELLO");
        return Ok(TaskHello { holder, data: DataDecl::single(rows, fingerprint), accept: None, version });
    }
    ensure!(
        version == TASK_PROTOCOL_VERSION_STREAMS,
        "TASK_HELLO version {version}, this syncer speaks {TASK_PROTOCOL_VERSION} and {TASK_PROTOCOL_VERSION_STREAMS}"
    );
    let fingerprint: [u8; 32] = r.take(32)?.try_into()?;
    let n = usize::from(r.u16()?);
    ensure!((1..=MAX_STREAMS).contains(&n), "TASK_HELLO declares {n} streams (1..={MAX_STREAMS})");
    let mut streams = Vec::with_capacity(n);
    for _ in 0..n {
        let rows = r.u64()?;
        let weight = r.u64()?;
        let len = usize::from(r.u16()?);
        ensure!(len <= MAX_STREAM_NAME, "TASK_HELLO stream name of {len} bytes");
        let name = String::from_utf8(r.take(len)?.to_vec()).context("TASK_HELLO stream name")?;
        streams.push(StreamDecl { rows, weight, name });
    }
    let m = usize::from(r.u16()?);
    ensure!(m <= n, "TASK_HELLO accepts {m} of {n} streams");
    let mut accept = Vec::with_capacity(m);
    for _ in 0..m {
        accept.push(r.u16()?);
    }
    ensure!(r.remaining() == 0, "trailing bytes in TASK_HELLO");
    Ok(TaskHello {
        holder,
        data: DataDecl { streams, fingerprint },
        accept: (!accept.is_empty()).then_some(accept),
        version,
    })
}

/// Check the TASK_HELLO HMAC (no key: the payload is the body).
pub fn open_task_hello<'a>(key: Option<&[u8]>, payload: &'a [u8]) -> Result<&'a [u8]> {
    let Some(key) = key else { return Ok(payload) };
    ensure!(
        payload.len() >= crate::hmac::HMAC_LEN,
        "TASK_HELLO shorter than its HMAC (island has no YETO_ISLAND_HMAC_KEY?)"
    );
    let (body, mac) = payload.split_at(payload.len() - crate::hmac::HMAC_LEN);
    let mut framed = Vec::with_capacity(TASK_HELLO_MAC_DOMAIN.len() + body.len());
    framed.extend_from_slice(TASK_HELLO_MAC_DOMAIN);
    framed.extend_from_slice(body);
    let expected = crate::hmac::hmac_sha256(key, &framed);
    let diff = expected.iter().zip(mac).fold(0u8, |acc, (x, y)| acc | (x ^ y));
    ensure!(diff == 0, "TASK_HELLO HMAC mismatch (wrong or missing island HMAC key)");
    Ok(body)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Welcome {
    pub lease_ms: u64,
    pub passes: u64,
    pub max_block: u32,
    pub reconnect: bool,
    pub superseded: Option<u64>,
    pub reclaimed: u64,
    pub next: u64,
}

/// The WELCOME in the island's protocol `version` (the layout is the same).
pub fn encode_welcome(w: &Welcome, version: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(48);
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&w.lease_ms.to_le_bytes());
    out.extend_from_slice(&w.passes.to_le_bytes());
    out.extend_from_slice(&w.max_block.to_le_bytes());
    out.push(u8::from(w.reconnect));
    out.push(u8::from(w.superseded.is_some()));
    out.extend_from_slice(&w.superseded.unwrap_or(0).to_le_bytes());
    out.extend_from_slice(&w.reclaimed.to_le_bytes());
    out.extend_from_slice(&w.next.to_le_bytes());
    out
}

fn encode_ranges(out: &mut Vec<u8>, ranges: &[(u64, u64)]) {
    out.extend_from_slice(&(ranges.len() as u32).to_le_bytes());
    for (start, end) in ranges {
        out.extend_from_slice(&start.to_le_bytes());
        out.extend_from_slice(&end.to_le_bytes());
    }
}

fn decode_ranges(r: &mut Reader<'_>) -> Result<Vec<(u64, u64)>> {
    let n = r.u32()? as usize;
    ensure!(n <= r.remaining() / 16, "range count {n} exceeds the payload");
    let mut ranges = Vec::with_capacity(n);
    for _ in 0..n {
        ranges.push((r.u64()?, r.u64()?));
    }
    Ok(ranges)
}

pub fn encode_grant(claim: &Claim) -> Vec<u8> {
    let mut out = Vec::new();
    let (status, block, ranges): (u8, u64, &[(u64, u64)]) = match claim {
        Claim::Grant { block, ranges } => (GRANT_STATUS_GRANT, *block, ranges),
        Claim::Wait => (GRANT_STATUS_WAIT, 0, &[]),
        Claim::End => (GRANT_STATUS_END, 0, &[]),
    };
    out.push(status);
    out.extend_from_slice(&block.to_le_bytes());
    let retry = if status == GRANT_STATUS_WAIT { WAIT_RETRY.as_millis() as u32 } else { 0 };
    out.extend_from_slice(&retry.to_le_bytes());
    encode_ranges(&mut out, ranges);
    out
}

#[cfg_attr(not(test), allow(dead_code))] // the island side encodes (yeto/rl/engine/task_dispense.py)
pub fn encode_report(seq: u64, entries: &[ReportEntry]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for entry in entries {
        out.extend_from_slice(&entry.block.to_le_bytes());
        out.push(entry.kind as u8);
        encode_ranges(&mut out, &entry.ranges);
    }
    out
}

pub fn decode_report(payload: &[u8]) -> Result<(u64, Vec<ReportEntry>)> {
    let mut r = Reader(payload);
    let seq = r.u64()?;
    let n = r.u32()? as usize;
    ensure!(n <= r.remaining() / 13, "report entry count {n} exceeds the payload");
    let mut entries = Vec::with_capacity(n);
    for _ in 0..n {
        let block = r.u64()?;
        let kind = r.u8()?;
        let kind = ReportKind::from_u8(kind).with_context(|| format!("unknown report kind {kind}"))?;
        let ranges = decode_ranges(&mut r)?;
        entries.push(ReportEntry { block, kind, ranges });
    }
    ensure!(r.remaining() == 0, "trailing bytes in TASK_REPORT");
    Ok((seq, entries))
}

#[cfg_attr(not(test), allow(dead_code))] // the island side encodes (yeto/rl/engine/task_dispense.py)
pub fn encode_claim(seq: u64, count: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(12);
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out
}

pub fn decode_claim(payload: &[u8]) -> Result<(u64, u32)> {
    let mut r = Reader(payload);
    let seq = r.u64()?;
    let count = r.u32()?;
    ensure!(r.remaining() == 0, "trailing bytes in TASK_CLAIM");
    Ok((seq, count))
}

pub fn encode_ack(t: &ReportTotals) -> Vec<u8> {
    let mut out = Vec::with_capacity(24);
    out.extend_from_slice(&t.settled.to_le_bytes());
    out.extend_from_slice(&t.requeued.to_le_bytes());
    out.extend_from_slice(&t.exhausted.to_le_bytes());
    out
}

/// The record path next to the checkpoint.
pub fn record_path(checkpoint: &Path) -> PathBuf {
    let mut name = checkpoint.as_os_str().to_owned();
    name.push(RECORD_SUFFIX);
    PathBuf::from(name)
}

struct Inner {
    dispenser: Dispenser,
    log: Option<std::fs::File>,
}

/// The dispenser of one syncer process, shared by every dispense connection.
pub struct DispenseService {
    inner: Mutex<Inner>,
    log_path: Option<PathBuf>,
    syncer_epoch: u64,
    hmac_key: Option<Vec<u8>>,
}

impl DispenseService {
    /// `restore`: a record to resume from (`--resume` with an existing record).
    pub fn new(
        cfg: DispenseConfig,
        log_path: Option<PathBuf>,
        syncer_epoch: u64,
        hmac_key: Option<Vec<u8>>,
        restore: Option<&Path>,
    ) -> Result<Self> {
        cfg.validate()?;
        let mut dispenser = Dispenser::new(cfg);
        if let Some(path) = restore {
            let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
            dispenser.restore_persisted(&bytes).with_context(|| format!("restore dispenser record {}", path.display()))?;
            info!(path = %path.display(), next = dispenser.next_position(), queued = dispenser.queued(), "dispenser record restored");
        }
        let log = match &log_path {
            Some(path) => Some(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .with_context(|| format!("open {}", path.display()))?,
            ),
            None => None,
        };
        let service = Self { inner: Mutex::new(Inner { dispenser, log }), log_path, syncer_epoch, hmac_key };
        service.with(|_, _| ());
        Ok(service)
    }

    /// Run `f` on the dispenser after expiring silent holders; write its events.
    pub fn with<T>(&self, f: impl FnOnce(&mut Dispenser, Instant) -> T) -> T {
        let mut inner = self.inner.lock().unwrap();
        let now = Instant::now();
        inner.dispenser.expire(now);
        let out = f(&mut inner.dispenser, now);
        let events = inner.dispenser.take_events();
        if !events.is_empty() {
            if let Some(file) = inner.log.as_mut() {
                let time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0);
                let mut text = String::new();
                for event in events {
                    text.push_str(&format!(
                        "{{{},\"syncer_epoch\":{},\"time_unix\":{time:.3}}}\n",
                        event.json_fields(),
                        self.syncer_epoch
                    ));
                }
                if let Err(error) = file.write_all(text.as_bytes()) {
                    let path = self.log_path.as_deref().unwrap_or(Path::new("?"));
                    warn!(path = %path.display(), "task-dispense log write failed: {error}");
                }
            }
        }
        out
    }

    pub fn hmac_key(&self) -> Option<&[u8]> {
        self.hmac_key.as_deref()
    }
}

impl crate::state::CheckpointSidecar for DispenseService {
    fn suffix(&self) -> &'static str {
        RECORD_SUFFIX
    }

    fn snapshot(&self) -> Vec<u8> {
        self.with(|d, _| d.checkpoint_record())
    }
}

async fn send(wr: &mut OwnedWriteHalf, msg_type: u8, payload: &[u8]) -> Result<()> {
    let mut header = [0u8; 13];
    header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    header[4] = msg_type;
    header[5..13].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    wr.write_all(&header).await?;
    wr.write_all(payload).await?;
    Ok(())
}

async fn refuse(wr: &mut OwnedWriteHalf, error: impl std::fmt::Display) -> Result<()> {
    let message = error.to_string();
    let _ = send(wr, MSG_ERROR, message.as_bytes()).await;
    bail!(message)
}

/// Serve one dispense connection whose first frame (TASK_HELLO) is `first`.
pub async fn serve(mut rd: OwnedReadHalf, mut wr: OwnedWriteHalf, first: &[u8], service: Arc<DispenseService>) -> Result<()> {
    let hello = match open_task_hello(service.hmac_key(), first).and_then(decode_task_hello) {
        Ok(hello) => hello,
        Err(error) => return refuse(&mut wr, format!("invalid TASK_HELLO: {error:#}")).await,
    };
    let holder = hello.holder;
    let opened = service.with(|d, now| {
        d.open_with(holder, &hello.data, hello.accept.as_deref(), now).map(|outcome| {
            let cfg = d.config();
            let (reconnect, superseded, reclaimed) = match outcome {
                OpenOutcome::Reconnected => (true, None, 0),
                OpenOutcome::Joined { superseded, reclaimed } => (false, superseded, reclaimed),
            };
            Welcome {
                lease_ms: cfg.lease.as_millis() as u64,
                passes: cfg.passes.unwrap_or(0),
                max_block: cfg.max_block,
                reconnect,
                superseded,
                reclaimed,
                next: d.next_position(),
            }
        })
    });
    match opened {
        Ok(welcome) => send(&mut wr, MSG_TASK_WELCOME, &encode_welcome(&welcome, hello.version)).await?,
        Err(error) => return refuse(&mut wr, error).await,
    }
    loop {
        let frame = match read_frame_limited(&mut rd, |msg_type| match msg_type {
            MSG_TASK_CLAIM => Ok(12),
            MSG_TASK_REPORT => Ok(MAX_TASK_REPORT),
            MSG_TASK_HEARTBEAT => Ok(0),
            other => bail!("dispense connection: unexpected message type {other}"),
        })
        .await
        {
            Ok(frame) => frame,
            Err(error) => {
                if let Some(io) = error.downcast_ref::<std::io::Error>() {
                    if io.kind() == std::io::ErrorKind::UnexpectedEof {
                        return Ok(()); // the island closed its connection
                    }
                    return Err(error);
                }
                return refuse(&mut wr, format!("{} {error:#}", crate::dispense::REFUSED_PREFIX)).await;
            }
        };
        let answer: std::result::Result<(u8, Vec<u8>), DispenseError> = match frame.msg_type {
            MSG_TASK_CLAIM => match decode_claim(&frame.payload) {
                Ok((seq, wanted)) => service
                    .with(|d, now| d.claim(holder, seq, wanted, now))
                    .map(|c| (MSG_TASK_GRANT, encode_grant(&c))),
                Err(error) => Err(DispenseError::Refused(format!("{error:#}"))),
            },
            MSG_TASK_REPORT => match decode_report(&frame.payload) {
                Ok((seq, entries)) => service
                    .with(|d, now| d.report(holder, seq, &entries, now))
                    .map(|t| (MSG_TASK_ACK, encode_ack(&t))),
                Err(error) => Err(DispenseError::Refused(format!("{error:#}"))),
            },
            _ => service
                .with(|d, now| d.heartbeat(holder, now))
                .map(|()| (MSG_TASK_ACK, encode_ack(&ReportTotals::default()))),
        };
        match answer {
            Ok((msg_type, payload)) => send(&mut wr, msg_type, &payload).await?,
            Err(error) => return refuse(&mut wr, error).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Golden bytes shared with `tests/test_s27_dispense.py` (Python codec).
    #[test]
    fn wire_golden_vectors() {
        let hello = TaskHello {
            holder: Holder { learner_id: 3, incarnation: 0x0102_0304_0506_0708 },
            data: DataDecl::single(480, [0xab; 32]),
            accept: None,
            version: TASK_PROTOCOL_VERSION,
        };
        let body = encode_task_hello(&hello);
        assert_eq!(
            hex(&body),
            format!("0100030000000807060504030201e001000000000000{}", "ab".repeat(32))
        );
        assert_eq!(decode_task_hello(&body).unwrap(), hello);
        let mac = crate::hmac::hmac_sha256(b"k", &[TASK_HELLO_MAC_DOMAIN, &body].concat());
        assert_eq!(
            hex(&mac),
            "78a9ab168d688c2f84a9ed2ac1f6ed21a55667af74969a9266cc7944a832fac9",
            "update the Python golden too if this changes"
        );
        let mut sealed = body.clone();
        sealed.extend_from_slice(&mac);
        assert_eq!(open_task_hello(Some(b"k"), &sealed).unwrap(), body.as_slice());
        assert!(open_task_hello(Some(b"other"), &sealed).is_err());
        let welcome = Welcome { lease_ms: 300_000, passes: 0, max_block: 65536, reconnect: false, superseded: Some(9), reclaimed: 6, next: 40 };
        assert_eq!(
            hex(&encode_welcome(&welcome, TASK_PROTOCOL_VERSION)),
            "0100e0930400000000000000000000000000000001000001090000000000000006000000000000002800000000000000"
        );
        assert_eq!(
            hex(&encode_grant(&Claim::Grant { block: 7, ranges: vec![(4, 10), (20, 22)] })),
            "000700000000000000000000000200000004000000000000000a0000000000000014000000000000001600000000000000"
        );
        assert_eq!(hex(&encode_grant(&Claim::Wait)), "010000000000000000e803000000000000");
        assert_eq!(hex(&encode_grant(&Claim::End)), "0200000000000000000000000000000000");
        let entries = vec![
            ReportEntry { block: 7, kind: ReportKind::Trained, ranges: vec![(4, 6)] },
            ReportEntry { block: 8, kind: ReportKind::Requeue, ranges: vec![] },
        ];
        let report = encode_report(5, &entries);
        assert_eq!(
            hex(&report),
            "050000000000000002000000070000000000000001010000000400000000000000060000000000000008000000000000000300000000"
        );
        assert_eq!(decode_report(&report).unwrap(), (5, entries));
        assert_eq!(hex(&encode_claim(4, 16)), "040000000000000010000000");
        assert_eq!(decode_claim(&encode_claim(4, 16)).unwrap(), (4, 16));
        assert!(decode_report(&report[..report.len() - 1]).is_err());
        assert!(decode_report(&[0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff]).is_err(), "a huge count is refused before allocating");
        assert_eq!(
            hex(&encode_ack(&ReportTotals { settled: 2, requeued: 1, exhausted: 0 })),
            "020000000000000001000000000000000000000000000000"
        );
    }

    /// TASK_HELLO version 2 (streams, S27 P3); shared with `tests/test_s27_dispense_p34.py`.
    #[test]
    fn wire_golden_vectors_with_streams() {
        let hello = TaskHello {
            holder: Holder { learner_id: 3, incarnation: 0x0102_0304_0506_0708 },
            data: DataDecl {
                streams: vec![
                    StreamDecl { rows: 10, weight: 7, name: "a/web".into() },
                    StreamDecl { rows: 5, weight: 3, name: String::new() },
                ],
                fingerprint: [0xab; 32],
            },
            accept: Some(vec![1]),
            version: TASK_PROTOCOL_VERSION_STREAMS,
        };
        let body = encode_task_hello(&hello);
        assert_eq!(
            hex(&body),
            format!(
                "0200030000000807060504030201{}02000a0000000000000007000000000000000500612f77656205000000000000000300000000000000000001000100",
                "ab".repeat(32)
            )
        );
        assert_eq!(decode_task_hello(&body).unwrap(), hello);
        let mac = crate::hmac::hmac_sha256(b"k", &[TASK_HELLO_MAC_DOMAIN, &body].concat());
        assert_eq!(hex(&mac), "ae4b9a4a78ae1aa7fbc05b27d614d78767b3464bc01b72ba7e2a30d6a83a9e97");
        // every stream accepted: m = 0
        let all = TaskHello { accept: None, ..hello.clone() };
        assert_eq!(decode_task_hello(&encode_task_hello(&all)).unwrap(), all);
        // the WELCOME answers in the island's version
        let welcome = Welcome { lease_ms: 1, passes: 0, max_block: 1, reconnect: false, superseded: None, reclaimed: 0, next: 0 };
        assert_eq!(&encode_welcome(&welcome, TASK_PROTOCOL_VERSION_STREAMS)[..2], &[2, 0]);
        // malformed: too many accepted streams, a long name, an unknown version
        let mut bad = body.clone();
        let m_at = bad.len() - 4;
        bad[m_at] = 3;
        assert!(decode_task_hello(&bad).is_err());
        let long = TaskHello {
            data: DataDecl { streams: vec![StreamDecl { rows: 1, weight: 1, name: "x".repeat(65) }], fingerprint: [0; 32] },
            ..hello.clone()
        };
        assert!(decode_task_hello(&encode_task_hello(&long)).is_err());
        let mut v3 = body.clone();
        v3[0] = 3;
        assert!(decode_task_hello(&v3).is_err());
    }
}
