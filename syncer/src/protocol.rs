//! Wire protocol (see docs/PROTOCOL.md). Little-endian framing over TCP.

use anyhow::{bail, Result};
use tokio::io::AsyncReadExt;

pub const MAGIC: u32 = 0xD170_C0DE;
pub const PROTOCOL_VERSION: u16 = 4;

pub const MSG_HELLO: u8 = 1;
pub const MSG_INIT_PARAMS: u8 = 2;
pub const MSG_PULL_REQ: u8 = 3;
pub const MSG_PUSH_FRAGMENT: u8 = 4;
pub const MSG_BCAST_FRAGMENT: u8 = 5;
pub const MSG_HEARTBEAT: u8 = 6;
pub const MSG_SHUTDOWN: u8 = 7;
pub const MSG_DATA_HELLO: u8 = 8;
pub const MSG_CHUNK: u8 = 9;
pub const MSG_ERROR: u8 = 10;
pub const MSG_FINAL_MANIFEST: u8 = 11;
pub const MSG_FINAL_ACK: u8 = 12;
/// Lossless f32 terminal fragment. Ordinary BCAST_FRAGMENT keeps using the
/// negotiated session dtype; final artifacts must preserve coordinator f32.
pub const MSG_FINAL_FRAGMENT: u8 = 13;
/// Benchmark-only report that one learner has frozen at its exact budget.
pub const MSG_BUDGET_DONE: u8 = 14;
// Message ids 15-25 belonged to a retired whole-vector island protocol;
// they stay unassigned so an old island gets a clear first-frame error.
/// An island leaves the pool: u32 learner_id | u8 reason (LEAVE_REASON_*).
/// Sent before the island closes its connection; the reason goes on the
/// `pool_leave` tape record.
pub const MSG_LEAVE: u8 = 27;
/// Local-init startup (S27): every island loaded the same base checkpoint and
/// reports the SHA-256 of each owned fragment's INIT_PARAMS tensor bytes:
/// `u32 learner_id | [32] layout_hash | u32 first_fragment | u32 count |
/// count x [32] digest`. Valid only on f32-bulk sessions.
pub const MSG_INIT_DIGEST: u8 = 28;
/// Syncer -> island: every island's digests and the applied INIT agree, so
/// the island adopts its local weights as version 0 and no initial BCAST
/// follows. Payload: `[32]` SHA-256 over the concatenated fragment digests.
pub const MSG_INIT_CONFIRM: u8 = 29;
/// Task dispenser (S27 P2, `dispense.rs`): the first frame of a dispense
/// connection, its own connection like BW_PROBE (never part of a learner
/// session). Payload: `u16 version (=1) | u32 learner_id | u64 incarnation |
/// u64 rows | [32] data fingerprint`, or with streams (S27 P3, version 2):
/// `u16 version (=2) | u32 learner_id | u64 incarnation | [32] fingerprint |
/// u16 n | n x (u64 rows, u64 weight, u16 name_len, name) | u16 m |
/// m x u16 accepted stream (m = 0: every stream)`; sealed with
/// `TASK_HELLO_MAC_DOMAIN` when the syncer has an island HMAC key.
pub const MSG_TASK_HELLO: u8 = 50;
/// Syncer -> island answer to TASK_HELLO: `u16 version | u64 lease_ms |
/// u64 passes (0 = unlimited) | u32 max_block | u8 reconnect |
/// u8 superseded | u64 superseded_incarnation | u64 reclaimed | u64 next`.
pub const MSG_TASK_WELCOME: u8 = 51;
/// Island -> syncer: `u64 seq | u32 count` positions wanted. `seq` numbers
/// the island's CLAIM / REPORT requests (strictly increasing); a request
/// resent after a lost answer repeats its `seq` and gets the same answer.
pub const MSG_TASK_CLAIM: u8 = 52;
/// Syncer -> island: `u8 status (0 grant, 1 wait, 2 end) | u64 block |
/// u32 retry_ms | u32 n | n x (u64 start, u64 end)`.
pub const MSG_TASK_GRANT: u8 = 53;
/// Island -> syncer: `u64 seq | u32 n | n x (u64 block | u8 kind | u32 m |
/// m x (u64 start, u64 end))`; kind 1 trained, 2 dropped, 3 requeue, 4 release.
pub const MSG_TASK_REPORT: u8 = 54;
/// Syncer -> island answer to REPORT and HEARTBEAT: `u64 settled |
/// u64 requeued | u64 exhausted`.
pub const MSG_TASK_ACK: u8 = 55;
/// Island -> syncer: empty; renews the lease.
pub const MSG_TASK_HEARTBEAT: u8 = 56;
pub const TASK_PROTOCOL_VERSION: u16 = 1;
/// TASK_HELLO / TASK_WELCOME version of an island with streams (S27 P3).
pub const TASK_PROTOCOL_VERSION_STREAMS: u16 = 2;
/// Largest TASK_HELLO payload (256 streams with 64-byte names, the HMAC).
pub const MAX_TASK_HELLO: u64 = 32 * 1024;
/// HMAC domain of TASK_HELLO (distinct from HELLO's).
pub const TASK_HELLO_MAC_DOMAIN: &[u8] = b"yeto-task-hello-mac-v1\0";
/// Largest TASK_REPORT payload (65536 ranges).
pub const MAX_TASK_REPORT: u64 = (1 << 20) + 12;

/// Largest INIT_DIGEST payload for a layout of `fragments` fragments.
pub fn init_digest_payload_len(fragments: usize) -> u64 {
    4 + 32 + 4 + 4 + 32 * fragments as u64
}
pub const LEAVE_REASON_NAMES: [&str; 7] = [
    "finished",
    "lease_expired",
    "stopped",
    "preempted",
    "infra_failed",
    "training_failed",
    "config_mismatch",
];

/// Tape name of a LEAVE reason byte.
pub fn leave_reason_name(reason: u8) -> String {
    LEAVE_REASON_NAMES
        .get(reason as usize)
        .map_or_else(|| format!("unknown_{reason}"), |name| (*name).to_string())
}
/// S26 E3b: link-bandwidth probe. First and only frame of its own connection
/// (never part of a learner session): the island uploads N bytes, the syncer
/// acknowledges with the u64 byte count, then streams M bytes back.
pub const MSG_BW_PROBE: u8 = 26;

/// Version of the final-artifact handshake carried inside FINAL_MANIFEST and
/// FINAL_ACK. Keeping this in the payload makes incompatible peers fail with
/// a specific error instead of interpreting a differently shaped message.
pub const FINALIZATION_REVISION: u16 = 1;

pub const DTYPE_F32: u8 = 1;
pub const DTYPE_BF16: u8 = 2;
/// Session dtype 3: PUSH_FRAGMENT payloads are block-quantized 4-bit E3M0
/// base-relative learner deltas; INIT_PARAMS and BCAST_FRAGMENT travel as
/// bf16 (see `bulk_dtype`, docs/PROTOCOL.md v4).
pub const DTYPE_Q4: u8 = 3;
/// Session dtype 4 (S26 E3): PUSH_FRAGMENT deltas travel as bf16 while
/// INIT_PARAMS, BCAST_FRAGMENT and FINAL stay exact f32. The syncer keeps f32
/// masters and momentum; only the learner's outer gradient is rounded.
pub const DTYPE_F32_PUSH_BF16: u8 = 4;

/// Session dtype 5 (S26 E3): PUSH_FRAGMENT deltas travel as block-scaled
/// int8 (see `decode_i8`); INIT_PARAMS, BCAST_FRAGMENT and FINAL stay f32.
pub const DTYPE_F32_PUSH_I8: u8 = 5;

/// Session dtype 6 (S27 P3): PUSH_FRAGMENT deltas travel in the dtype-3 q4
/// block codec (see `decode_q4`); INIT_PARAMS, BCAST_FRAGMENT and FINAL stay
/// f32, so only the learner's outer gradient is quantized.
pub const DTYPE_F32_PUSH_Q4: u8 = 6;

/// Session dtype 7 (S27 P3): PUSH_FRAGMENT deltas travel as linear symmetric
/// int4 with one absmax scale per block (see `decode_i4`); INIT_PARAMS,
/// BCAST_FRAGMENT and FINAL stay f32.
pub const DTYPE_F32_PUSH_I4: u8 = 7;
/// Session-dtype flag (S26 E3): BCAST_FRAGMENT carries a 41-byte downlink
/// header and, between periodic f32 refreshes, a bf16 delta against the
/// anchor every learner holds (see `crate::downlink`). Valid only on f32-bulk
/// sessions (dtypes 1, 4, 5, 6, 7).
pub const DTYPE_FLAG_BCAST_DELTA: u8 = 0x80;
/// Session-dtype flag (S27): quantization blocks of PUSH_FRAGMENT deltas
/// restart at every tensor boundary of the fragment, so a replicated tensor
/// quantizes identically whatever its neighbours are. Valid only on the
/// block-quantized push dtypes 5, 6 and 7; without it the blocks run
/// across the whole fragment.
pub const DTYPE_FLAG_TENSOR_BLOCKS: u8 = 0x40;

/// Values per q4 scale block (f32 absmax scale + 128 nibble bytes each).
pub const Q4_BLOCK: usize = 256;
/// Bounds of the int8 push block size the learner declares per payload.
pub const I8_BLOCK_MIN: usize = 32;
pub const I8_BLOCK_MAX: usize = 1 << 20;
/// Bounds of the linear int4 push block size (even, so nibbles pair up).
pub const I4_BLOCK_MIN: usize = 64;
pub const I4_BLOCK_MAX: usize = 256;

/// The session dtype without its flags.
pub fn session_base(dtype: u8) -> u8 {
    dtype & !(DTYPE_FLAG_BCAST_DELTA | DTYPE_FLAG_TENSOR_BLOCKS)
}

/// Whether push blocks restart at each tensor boundary of a fragment.
pub fn tensor_blocks(dtype: u8) -> bool {
    dtype & DTYPE_FLAG_TENSOR_BLOCKS != 0
}

/// Whether the session receives bf16 downlink deltas.
pub fn bcast_delta(dtype: u8) -> bool {
    dtype & DTYPE_FLAG_BCAST_DELTA != 0
}

/// Whether `dtype` is a session dtype this syncer accepts.
pub fn valid_session_dtype(dtype: u8) -> bool {
    if tensor_blocks(dtype)
        && !matches!(
            session_base(dtype),
            DTYPE_F32_PUSH_I8 | DTYPE_F32_PUSH_Q4 | DTYPE_F32_PUSH_I4
        )
    {
        return false;
    }
    match session_base(dtype) {
        DTYPE_F32 | DTYPE_F32_PUSH_BF16 | DTYPE_F32_PUSH_I8 | DTYPE_F32_PUSH_Q4 | DTYPE_F32_PUSH_I4 => {
            true
        }
        DTYPE_BF16 | DTYPE_Q4 => !bcast_delta(dtype),
        _ => false,
    }
}

/// Dtype of INIT_PARAMS/BCAST_FRAGMENT tensors for a session dtype: q4
/// applies only to push deltas; full parameter payloads stay bf16.
pub fn bulk_dtype(dtype: u8) -> u8 {
    match session_base(dtype) {
        DTYPE_Q4 => DTYPE_BF16,
        DTYPE_F32_PUSH_BF16 | DTYPE_F32_PUSH_I8 | DTYPE_F32_PUSH_Q4 | DTYPE_F32_PUSH_I4 => DTYPE_F32,
        other => other,
    }
}

/// Dtype of PUSH_FRAGMENT deltas for an unquantized session dtype (q4 and
/// int8 have their own block codecs).
pub fn push_dtype(dtype: u8) -> u8 {
    match session_base(dtype) {
        DTYPE_F32_PUSH_BF16 => DTYPE_BF16,
        other => other,
    }
}

/// Largest PUSH_FRAGMENT delta payload a session dtype may carry.
pub fn max_push_nbytes(dtype: u8, numel: usize) -> Result<usize> {
    max_push_nbytes_segments(dtype, &[numel])
}

/// The value runs whose quantization blocks restart independently: one run
/// per tensor on a tensor-block session, else the whole fragment.
pub fn push_segments(dtype: u8, numel: usize, tensor_numels: &[u64]) -> Result<Vec<usize>> {
    if !tensor_blocks(dtype) {
        return Ok(vec![numel]);
    }
    tensor_numels
        .iter()
        .map(|n| usize::try_from(*n).map_err(|_| anyhow::anyhow!("tensor numel does not fit usize")))
        .collect()
}

/// Largest PUSH_FRAGMENT delta payload for a fragment split into `segments`.
pub fn max_push_nbytes_segments(dtype: u8, segments: &[usize]) -> Result<usize> {
    let total: usize = segments.iter().sum();
    match session_base(dtype) {
        DTYPE_Q4 | DTYPE_F32_PUSH_Q4 => q4_nbytes_segments(segments),
        DTYPE_F32_PUSH_I8 => i8_nbytes_segments(segments, I8_BLOCK_MIN),
        DTYPE_F32_PUSH_I4 => i4_nbytes_segments(segments, I4_BLOCK_MIN),
        other => tensor_nbytes(push_dtype(other), total),
    }
}

/// Decode a PUSH_FRAGMENT delta payload of any session dtype into f32s.
pub fn decode_push(dtype: u8, bytes: &[u8], numel: usize, out: &mut Vec<f32>) -> Result<()> {
    decode_push_segments(dtype, bytes, &[numel], out)
}

/// Decode a PUSH_FRAGMENT delta payload whose blocks restart at each of
/// `segments` (see `push_segments`).
pub fn decode_push_segments(dtype: u8, bytes: &[u8], segments: &[usize], out: &mut Vec<f32>) -> Result<()> {
    match session_base(dtype) {
        DTYPE_Q4 | DTYPE_F32_PUSH_Q4 => decode_q4_segments(bytes, segments, out),
        DTYPE_F32_PUSH_I8 => decode_i8_segments(bytes, segments, out),
        DTYPE_F32_PUSH_I4 => decode_i4_segments(bytes, segments, out),
        other => {
            let numel: usize = segments.iter().sum();
            let expected = tensor_nbytes(push_dtype(other), numel)?;
            if bytes.len() != expected {
                bail!("push delta has {} bytes, expected {expected}", bytes.len());
            }
            decode_tensor(push_dtype(other), bytes, out)
        }
    }
}

/// Upper bound for the first HELLO/DATA_HELLO frame. Layout metadata is
/// compact; keeping this small prevents unauthenticated pre-session sockets
/// from forcing model-sized allocations.
pub const MAX_HELLO_FRAME: u64 = 16 * 1024 * 1024;

#[derive(Debug)]
pub struct Frame {
    pub msg_type: u8,
    pub payload: Vec<u8>,
}

/// Read one frame, checking a message-type-specific limit before allocating
/// or reading its payload.
pub async fn read_frame_limited<R, F>(r: &mut R, limit: F) -> Result<Frame>
where
    R: AsyncReadExt + Unpin,
    F: FnOnce(u8) -> Result<u64>,
{
    let magic = r.read_u32_le().await?;
    if magic != MAGIC {
        bail!("bad magic 0x{magic:08x}");
    }
    let msg_type = r.read_u8().await?;
    let len = r.read_u64_le().await?;
    let max_len = limit(msg_type)?;
    if len > max_len {
        bail!("frame type {msg_type} has length {len}, exceeds limit {max_len}");
    }
    let len =
        usize::try_from(len).map_err(|_| anyhow::anyhow!("frame length does not fit usize"))?;
    let mut payload = Vec::new();
    payload
        .try_reserve_exact(len)
        .map_err(|error| anyhow::anyhow!("cannot allocate frame payload: {error}"))?;
    payload.resize(len, 0);
    r.read_exact(&mut payload).await?;
    Ok(Frame { msg_type, payload })
}

/// Cursor helpers for parsing payloads.
pub struct Reader<'a>(pub &'a [u8]);

impl<'a> Reader<'a> {
    pub fn u8(&mut self) -> Result<u8> {
        let (v, rest) = self.0.split_first().ok_or_else(|| anyhow::anyhow!("eof"))?;
        self.0 = rest;
        Ok(*v)
    }
    pub fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }
    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.0.len() < n {
            bail!("payload truncated");
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }
    pub fn rest(&mut self) -> &'a [u8] {
        std::mem::take(&mut self.0)
    }
    pub fn remaining(&self) -> usize {
        self.0.len()
    }
}

/// Bytes of one per-tensor routed-token tag: `routed u64 | layer_routed u64 |
/// layer_experts u32`.
pub const EXPERT_TAG_BYTES: usize = 20;

/// Bytes of the routed-token tag block of a fragment with `tensors` tensors.
pub fn expert_tags_nbytes(tensors: usize) -> Result<usize> {
    tensors
        .checked_mul(EXPERT_TAG_BYTES)
        .and_then(|n| n.checked_add(4))
        .ok_or_else(|| anyhow::anyhow!("expert tag block length overflow"))
}

/// Read the routed-token tag block that a session with expert token weights
/// puts between a PUSH_FRAGMENT header and its delta bytes:
/// `n u32 | n x (routed u64 | layer_routed u64 | layer_experts u32)`, one
/// entry per fragment tensor in layout order.
///
/// Returns the per-tensor merge scale. A dense tensor is tagged `(0, 0, 0)`
/// and keeps scale 1. An expert tensor gets `routed * layer_experts /
/// layer_routed`: the tokens this island routed to the expert during the
/// outer interval relative to an even split over the layer's experts. The
/// scale is 1 under even routing and 0 for an expert the island never
/// routed to (or a layer that saw no token).
pub fn decode_expert_tags(r: &mut Reader<'_>, tensors: usize) -> Result<Vec<f64>> {
    let n = r.u32()? as usize;
    if n != tensors {
        bail!("expert tag block has {n} entries, fragment has {tensors} tensors");
    }
    if r.remaining() < expert_tags_nbytes(n)? - 4 {
        bail!("expert tag block truncated");
    }
    (0..n)
        .map(|i| {
            let routed = r.u64()?;
            let layer_routed = r.u64()?;
            let experts = r.u32()?;
            if experts == 0 {
                if routed != 0 || layer_routed != 0 {
                    bail!("dense tensor {i} carries routed tokens");
                }
                return Ok(1.0);
            }
            if routed > layer_routed {
                bail!("tensor {i}: expert routed {routed} tokens of only {layer_routed} in its layer");
            }
            if layer_routed == 0 {
                return Ok(0.0);
            }
            Ok(routed as f64 * f64::from(experts) / layer_routed as f64)
        })
        .collect()
}

/// Exact encoded byte length for an unquantized tensor payload.
pub fn tensor_nbytes(dtype: u8, numel: usize) -> Result<usize> {
    let width = match dtype {
        DTYPE_F32 => 4,
        DTYPE_BF16 => 2,
        _ => bail!("unknown tensor dtype {dtype}"),
    };
    numel
        .checked_mul(width)
        .ok_or_else(|| anyhow::anyhow!("tensor byte length overflow"))
}

/// Exact encoded byte length for a q4 learner-delta payload.
pub fn q4_nbytes(numel: usize) -> Result<usize> {
    numel
        .div_ceil(Q4_BLOCK)
        .checked_mul(4 + Q4_BLOCK / 2)
        .ok_or_else(|| anyhow::anyhow!("q4 tensor byte length overflow"))
}

/// Exact encoded byte length of a q4 payload whose blocks restart at each of
/// `segments` (every segment's last block is zero-padded).
pub fn q4_nbytes_segments(segments: &[usize]) -> Result<usize> {
    segments
        .iter()
        .try_fold(0usize, |sum, &n| sum.checked_add(q4_nbytes(n).ok()?))
        .ok_or_else(|| anyhow::anyhow!("q4 tensor byte length overflow"))
}

/// Encode the authoritative final cut: the coordinator's global step and the
/// exact expected version of every fragment, in layout order.
pub fn encode_final_manifest(global_step: u64, versions: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(14 + versions.len() * 8);
    out.extend_from_slice(&FINALIZATION_REVISION.to_le_bytes());
    out.extend_from_slice(&global_step.to_le_bytes());
    out.extend_from_slice(&(versions.len() as u32).to_le_bytes());
    for version in versions {
        out.extend_from_slice(&version.to_le_bytes());
    }
    out
}

/// Decode a learner's acknowledgement of the final manifest.
pub fn decode_final_ack(payload: &[u8]) -> Result<u64> {
    let mut r = Reader(payload);
    let revision = r.u16()?;
    if revision != FINALIZATION_REVISION {
        bail!("unsupported finalization revision {revision}; expected {FINALIZATION_REVISION}");
    }
    let global_step = r.u64()?;
    if !r.rest().is_empty() {
        bail!("FINAL_ACK has trailing bytes");
    }
    Ok(global_step)
}

/// Decode the exact local optimizer-step budget reported by a learner.
pub fn decode_budget_done(payload: &[u8]) -> Result<u64> {
    let mut r = Reader(payload);
    let local_steps = r.u64()?;
    if !r.rest().is_empty() {
        bail!("BUDGET_DONE has trailing bytes");
    }
    Ok(local_steps)
}

/// Decode a tensor payload into f32s.
pub fn decode_tensor(dtype: u8, bytes: &[u8], out: &mut Vec<f32>) -> Result<()> {
    out.clear();
    match dtype {
        DTYPE_F32 => {
            if bytes.len() % 4 != 0 {
                bail!("f32 tensor byte length not divisible by 4");
            }
            out.try_reserve(bytes.len() / 4)
                .map_err(|error| anyhow::anyhow!("cannot allocate f32 tensor: {error}"))?;
            for c in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes(c.try_into().unwrap()));
            }
        }
        DTYPE_BF16 => {
            if bytes.len() % 2 != 0 {
                bail!("bf16 tensor byte length not divisible by 2");
            }
            out.try_reserve(bytes.len() / 2)
                .map_err(|error| anyhow::anyhow!("cannot allocate bf16 tensor: {error}"))?;
            for c in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes(c.try_into().unwrap());
                out.push(f32::from_bits((bits as u32) << 16));
            }
        }
        _ => bail!("unknown dtype {dtype}"),
    }
    if out.iter().any(|value| !value.is_finite()) {
        bail!("tensor payload contains a non-finite value");
    }
    Ok(())
}

/// Encode f32s into the wire dtype.
#[cfg(test)]
pub fn encode_tensor(dtype: u8, vals: &[f32], out: &mut Vec<u8>) -> Result<()> {
    out.clear();
    match dtype {
        DTYPE_F32 => {
            out.reserve(vals.len() * 4);
            for v in vals {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        DTYPE_BF16 => {
            out.reserve(vals.len() * 2);
            for v in vals {
                let bits = half::bf16::from_f32(*v).to_bits();
                out.extend_from_slice(&bits.to_le_bytes());
            }
        }
        _ => bail!("unknown dtype {dtype}"),
    }
    Ok(())
}

/// Exact encoded byte length of an int8 push payload with block size `block`.
pub fn i8_nbytes(numel: usize, block: usize) -> Result<usize> {
    i8_nbytes_segments(&[numel], block)
}

/// As `i8_nbytes` when blocks restart at each of `segments`.
pub fn i8_nbytes_segments(segments: &[usize], block: usize) -> Result<usize> {
    if !(I8_BLOCK_MIN..=I8_BLOCK_MAX).contains(&block) {
        bail!("int8 block size {block} outside {I8_BLOCK_MIN}..={I8_BLOCK_MAX}");
    }
    segments
        .iter()
        .try_fold(4usize, |sum, &n| {
            n.div_ceil(block).checked_mul(4)?.checked_add(n)?.checked_add(sum)
        })
        .ok_or_else(|| anyhow::anyhow!("int8 tensor byte length overflow"))
}

/// Decode an int8 delta payload (yeto/protocol.py `f32_bytes_to_i8`).
///
/// Layout: u32 block size B, then per block of B values (the last block may
/// be shorter): f32 scale (absmax / 127), then one i8 per value. A value
/// decodes to `q as f32 * scale`, so |error| <= scale / 2 = absmax / 254 per
/// element (plus one f32 rounding). The scale is non-negative and finite.
pub fn decode_i8(bytes: &[u8], numel: usize, out: &mut Vec<f32>) -> Result<()> {
    decode_i8_segments(bytes, &[numel], out)
}

/// As `decode_i8`; blocks restart at the start of every segment.
pub fn decode_i8_segments(bytes: &[u8], segments: &[usize], out: &mut Vec<f32>) -> Result<()> {
    if bytes.len() < 4 {
        bail!("int8 payload is shorter than its block header");
    }
    let block = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let expected = i8_nbytes_segments(segments, block)?;
    let numel: usize = segments.iter().sum();
    if bytes.len() != expected {
        bail!(
            "int8 payload has {} bytes, expected {expected} for {numel} values in blocks of {block}",
            bytes.len()
        );
    }
    out.clear();
    out.try_reserve(numel)
        .map_err(|error| anyhow::anyhow!("cannot allocate int8 tensor: {error}"))?;
    let mut at = 4usize;
    let mut b = 0usize;
    for &segment in segments {
        let mut remaining = segment;
        while remaining > 0 {
            let len = remaining.min(block);
            let scale = f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
            if !scale.is_finite() || scale < 0.0 {
                bail!("int8 block {b} has invalid scale {scale}");
            }
            at += 4;
            for &q in &bytes[at..at + len] {
                let q = q as i8;
                if q == i8::MIN {
                    bail!("int8 block {b} uses the reserved value -128");
                }
                out.push(q as f32 * scale);
            }
            at += len;
            remaining -= len;
            b += 1;
        }
    }
    Ok(())
}

/// Encode f32s as an int8 push payload (test counterpart of the Python codec).
#[cfg(test)]
pub fn encode_i8(vals: &[f32], block: usize) -> Vec<u8> {
    let mut out = (block as u32).to_le_bytes().to_vec();
    for chunk in vals.chunks(block) {
        let absmax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = absmax / 127.0;
        out.extend_from_slice(&scale.to_le_bytes());
        for v in chunk {
            let q = if scale > 0.0 { (v / scale).round_ties_even().clamp(-127.0, 127.0) } else { 0.0 };
            out.push(q as i8 as u8);
        }
    }
    out
}

/// Encoded size of a linear int4 push: u32 block, per block an f32 scale,
/// then two values per byte (an odd last block pads one zero nibble).
pub fn i4_nbytes(numel: usize, block: usize) -> Result<usize> {
    i4_nbytes_segments(&[numel], block)
}

/// As `i4_nbytes` when blocks restart at each of `segments`.
pub fn i4_nbytes_segments(segments: &[usize], block: usize) -> Result<usize> {
    if !(I4_BLOCK_MIN..=I4_BLOCK_MAX).contains(&block) || block % 2 != 0 {
        bail!("int4 block size {block} must be even and within {I4_BLOCK_MIN}..={I4_BLOCK_MAX}");
    }
    let mut total = 4usize;
    for &n in segments {
        // Blocks are even, so only a segment's last block can be odd.
        let body = n.div_ceil(block).checked_mul(4).and_then(|s| s.checked_add(n / block * (block / 2)));
        let tail = n % block;
        total = body
            .and_then(|b| b.checked_add(if tail > 0 { tail.div_ceil(2) } else { 0 }))
            .and_then(|b| b.checked_add(total))
            .ok_or_else(|| anyhow::anyhow!("int4 tensor byte length overflow"))?;
    }
    Ok(total)
}

/// Decode a linear int4 delta payload (yeto/protocol.py `f32_bytes_to_i4`).
///
/// Layout: u32 block size B (even, 64..=256), then per block of B values (the
/// last may be shorter): f32 scale (absmax / 7), then the values as signed
/// 4-bit two's-complement nibbles, low nibble first. A value decodes to
/// `q as f32 * scale`; -8 is reserved and an odd block's pad nibble is 0.
/// Round-to-nearest errs by at most scale / 2 = absmax / 14 per element.
pub fn decode_i4(bytes: &[u8], numel: usize, out: &mut Vec<f32>) -> Result<()> {
    decode_i4_segments(bytes, &[numel], out)
}

/// As `decode_i4`; blocks restart at the start of every segment.
pub fn decode_i4_segments(bytes: &[u8], segments: &[usize], out: &mut Vec<f32>) -> Result<()> {
    if bytes.len() < 4 {
        bail!("int4 payload is shorter than its block header");
    }
    let block = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let expected = i4_nbytes_segments(segments, block)?;
    let numel: usize = segments.iter().sum();
    if bytes.len() != expected {
        bail!(
            "int4 payload has {} bytes, expected {expected} for {numel} values in blocks of {block}",
            bytes.len()
        );
    }
    out.clear();
    out.try_reserve(numel)
        .map_err(|error| anyhow::anyhow!("cannot allocate int4 tensor: {error}"))?;
    let mut at = 4usize;
    let mut b = 0usize;
    for &segment in segments {
        let mut remaining = segment;
        while remaining > 0 {
            let len = remaining.min(block);
            let scale = f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
            if !scale.is_finite() || scale < 0.0 {
                bail!("int4 block {b} has invalid scale {scale}");
            }
            at += 4;
            let body = &bytes[at..at + len.div_ceil(2)];
            for i in 0..len {
                let nibble = (body[i / 2] >> (4 * (i % 2))) & 0x0F;
                if nibble == 0x08 {
                    bail!("int4 block {b} uses the reserved value -8");
                }
                // Sign-extend the 4-bit two's-complement value.
                let q = ((nibble << 4) as i8) >> 4;
                out.push(q as f32 * scale);
            }
            if len % 2 == 1 && body[len / 2] >> 4 != 0 {
                bail!("int4 block {b} has a non-zero pad nibble");
            }
            at += body.len();
            remaining -= len;
            b += 1;
        }
    }
    Ok(())
}

/// Encode f32s as a linear int4 push payload (test counterpart of the Python codec).
#[cfg(test)]
pub fn encode_i4(vals: &[f32], block: usize) -> Vec<u8> {
    let mut out = (block as u32).to_le_bytes().to_vec();
    for chunk in vals.chunks(block) {
        let absmax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = absmax / 7.0;
        out.extend_from_slice(&scale.to_le_bytes());
        let q: Vec<u8> = chunk
            .iter()
            .map(|v| {
                let q = if scale > 0.0 { (v / scale).round_ties_even().clamp(-7.0, 7.0) } else { 0.0 };
                (q as i8 as u8) & 0x0F
            })
            .collect();
        for pair in q.chunks(2) {
            out.push(pair[0] | pair.get(1).map_or(0, |hi| hi << 4));
        }
    }
    out
}

/// Decode a q4 delta payload (yeto/tensor_io.py `quantize_q4`) into f32s.
///
/// Per block of `Q4_BLOCK` values: f32 absmax scale, then 128 bytes of
/// packed nibbles (two values per byte, low nibble first). Nibble: bit 3 =
/// sign, bits 0-2 = level; level 0 is exactly zero, level L in 1..=7
/// decodes to sign * 2^(L-7) * scale. The last block is zero-padded to
/// `Q4_BLOCK`; `numel` truncates the tail.
pub fn decode_q4(bytes: &[u8], numel: usize, out: &mut Vec<f32>) -> Result<()> {
    decode_q4_segments(bytes, &[numel], out)
}

/// As `decode_q4`; blocks restart at the start of every segment.
pub fn decode_q4_segments(bytes: &[u8], segments: &[usize], out: &mut Vec<f32>) -> Result<()> {
    let expected = q4_nbytes_segments(segments)?;
    let numel: usize = segments.iter().sum();
    if bytes.len() != expected {
        bail!("q4 payload has {} bytes, expected {expected} for {numel} values", bytes.len());
    }
    out.clear();
    out.try_reserve(numel)
        .map_err(|error| anyhow::anyhow!("cannot allocate q4 tensor: {error}"))?;
    let mut at = 0usize;
    for &segment in segments {
        let len = q4_nbytes(segment)?;
        decode_q4_append(&bytes[at..at + len], segment, out)?;
        at += len;
    }
    Ok(())
}

fn decode_q4_append(bytes: &[u8], numel: usize, out: &mut Vec<f32>) -> Result<()> {
    let block_bytes = 4 + Q4_BLOCK / 2;
    // 2^(level-7) for level 1..=7; level 0 handled as exact zero.
    const LUT: [f32; 8] = [
        0.0,
        1.0 / 64.0,
        1.0 / 32.0,
        1.0 / 16.0,
        1.0 / 8.0,
        1.0 / 4.0,
        1.0 / 2.0,
        1.0,
    ];
    for (b, chunk) in bytes.chunks_exact(block_bytes).enumerate() {
        let scale = f32::from_le_bytes(chunk[..4].try_into().unwrap());
        if !scale.is_finite() || scale < 0.0 {
            bail!("q4 block {b} has invalid scale {scale}");
        }
        let base = b * Q4_BLOCK;
        for (i, byte) in chunk[4..].iter().enumerate() {
            for (j, nibble) in [byte & 0x0F, byte >> 4].into_iter().enumerate() {
                if base + i * 2 + j >= numel {
                    return Ok(());
                }
                let mag = LUT[(nibble & 0x07) as usize] * scale;
                out.push(if nibble >= 8 { -mag } else { mag });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;

    #[test]
    fn int8_push_round_trips_within_half_a_scale_step() {
        let vals: Vec<f32> = (0..1000)
            .map(|i| ((i as f32) * 0.731).sin() * if i % 300 < 150 { 1e-3 } else { 2.5 })
            .collect();
        for block in [I8_BLOCK_MIN, 128, 333, 4096] {
            let wire = encode_i8(&vals, block);
            assert_eq!(wire.len(), i8_nbytes(vals.len(), block).unwrap());
            assert!(wire.len() <= max_push_nbytes(DTYPE_F32_PUSH_I8, vals.len()).unwrap());
            let mut out = Vec::new();
            decode_push(DTYPE_F32_PUSH_I8, &wire, vals.len(), &mut out).unwrap();
            for (chunk, decoded) in vals.chunks(block).zip(out.chunks(block)) {
                let absmax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
                for (v, d) in chunk.iter().zip(decoded) {
                    // |error| <= scale/2 = absmax/254, plus f32 rounding.
                    assert!((v - d).abs() <= absmax / 254.0 * (1.0 + 1e-5), "{v} {d} {absmax}");
                }
            }
        }
        // All-zero blocks decode to exact zeros.
        let mut out = Vec::new();
        decode_i8(&encode_i8(&[0.0; 40], 32), 40, &mut out).unwrap();
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn q4_push_on_f32_session_uses_the_q4_codec() {
        // Two blocks: 256 values then 4 (zero-padded). Every nibble is
        // sign=1, level=7 (0xF), i.e. -scale; the second scale is 0.5.
        let numel = Q4_BLOCK + 4;
        let mut wire = Vec::new();
        for scale in [2.0f32, 0.5] {
            wire.extend_from_slice(&scale.to_le_bytes());
            wire.extend(std::iter::repeat_n(0xFFu8, Q4_BLOCK / 2));
        }
        assert_eq!(wire.len(), max_push_nbytes(DTYPE_F32_PUSH_Q4, numel).unwrap());
        assert_eq!(wire.len(), max_push_nbytes(DTYPE_Q4, numel).unwrap());
        let mut out = Vec::new();
        decode_push(DTYPE_F32_PUSH_Q4 | DTYPE_FLAG_BCAST_DELTA, &wire, numel, &mut out).unwrap();
        assert_eq!(out.len(), numel);
        assert!(out[..Q4_BLOCK].iter().all(|v| *v == -2.0));
        assert!(out[Q4_BLOCK..].iter().all(|v| *v == -0.5));
        assert!(decode_push(DTYPE_F32_PUSH_Q4, &wire[1..], numel, &mut out).is_err());
    }

    #[test]
    fn int4_push_round_trips_within_half_a_scale_step() {
        let vals: Vec<f32> = (0..1001)
            .map(|i| ((i as f32) * 0.731).sin() * if i % 300 < 150 { 1e-3 } else { 2.5 })
            .collect();
        for block in [I4_BLOCK_MIN, 128, 250, I4_BLOCK_MAX] {
            let wire = encode_i4(&vals, block);
            assert_eq!(wire.len(), i4_nbytes(vals.len(), block).unwrap());
            assert!(wire.len() <= max_push_nbytes(DTYPE_F32_PUSH_I4, vals.len()).unwrap());
            let mut out = Vec::new();
            decode_push(DTYPE_F32_PUSH_I4, &wire, vals.len(), &mut out).unwrap();
            for (chunk, decoded) in vals.chunks(block).zip(out.chunks(block)) {
                let absmax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
                for (v, d) in chunk.iter().zip(decoded) {
                    assert!((v - d).abs() <= absmax / 14.0 * (1.0 + 1e-5), "{v} {d} {absmax}");
                }
            }
        }
    }

    #[test]
    fn int4_push_rejects_malformed_payloads() {
        let vals = vec![-1.0f32; 65];
        let mut out = Vec::new();
        let wire = encode_i4(&vals, 64);
        decode_i4(&wire, 65, &mut out).unwrap();
        assert!(out.iter().all(|v| *v == -1.0));
        assert!(decode_i4(&wire[..wire.len() - 1], 65, &mut out).is_err());
        assert!(decode_i4(&encode_i4(&vals, 62), 65, &mut out).is_err());
        assert!(decode_i4(&encode_i4(&vals, 65), 65, &mut out).is_err());
        let mut reserved = wire.clone();
        reserved[8] = 0x08;
        assert!(decode_i4(&reserved, 65, &mut out).is_err());
        let mut pad = wire;
        let last = pad.len() - 1;
        pad[last] |= 0x10;
        assert!(decode_i4(&pad, 65, &mut out).is_err());
    }

    #[test]
    fn int8_push_rejects_malformed_payloads() {
        let vals = vec![1.0f32; 64];
        let mut out = Vec::new();
        let wire = encode_i8(&vals, 32);
        assert!(decode_i8(&wire[..wire.len() - 1], 64, &mut out).is_err());
        assert!(decode_i8(&encode_i8(&vals, 16), 64, &mut out).is_err());
        let mut bad_scale = wire.clone();
        bad_scale[4..8].copy_from_slice(&(-1.0f32).to_le_bytes());
        assert!(decode_i8(&bad_scale, 64, &mut out).is_err());
        let mut reserved = wire;
        reserved[8] = 0x80;
        assert!(decode_i8(&reserved, 64, &mut out).is_err());
    }

    #[test]
    fn session_dtype_flags_compose_only_with_f32_bulk() {
        assert!(valid_session_dtype(DTYPE_F32_PUSH_I8));
        assert!(valid_session_dtype(DTYPE_F32 | DTYPE_FLAG_BCAST_DELTA));
        assert!(valid_session_dtype(DTYPE_F32_PUSH_BF16 | DTYPE_FLAG_BCAST_DELTA));
        assert!(valid_session_dtype(DTYPE_F32_PUSH_I8 | DTYPE_FLAG_BCAST_DELTA));
        assert!(!valid_session_dtype(DTYPE_BF16 | DTYPE_FLAG_BCAST_DELTA));
        assert!(!valid_session_dtype(DTYPE_Q4 | DTYPE_FLAG_BCAST_DELTA));
        assert!(valid_session_dtype(DTYPE_F32_PUSH_Q4));
        assert!(valid_session_dtype(DTYPE_F32_PUSH_Q4 | DTYPE_FLAG_BCAST_DELTA));
        assert!(valid_session_dtype(DTYPE_F32_PUSH_I4 | DTYPE_FLAG_BCAST_DELTA));
        assert!(!valid_session_dtype(8));
        assert_eq!(bulk_dtype(DTYPE_F32_PUSH_Q4 | DTYPE_FLAG_BCAST_DELTA), DTYPE_F32);
        assert_eq!(bulk_dtype(DTYPE_F32_PUSH_I8 | DTYPE_FLAG_BCAST_DELTA), DTYPE_F32);
        assert_eq!(push_dtype(DTYPE_F32_PUSH_BF16 | DTYPE_FLAG_BCAST_DELTA), DTYPE_BF16);
    }

    #[test]
    fn reserved_message_ids_are_stable() {
        assert_eq!(MSG_ERROR, 10);
        assert_eq!(MSG_FINAL_MANIFEST, 11);
        assert_eq!(MSG_FINAL_ACK, 12);
        assert_eq!(MSG_FINAL_FRAGMENT, 13);
        assert_eq!(MSG_BUDGET_DONE, 14);
    }

    #[tokio::test]
    async fn huge_first_frame_is_rejected_before_payload_read() {
        let (mut writer, mut reader) = tokio::io::duplex(32);
        writer.write_u32_le(MAGIC).await.unwrap();
        writer.write_u8(MSG_HELLO).await.unwrap();
        writer.write_u64_le(MAX_HELLO_FRAME + 1).await.unwrap();
        // Keep the writer open and send no payload. A vulnerable reader would
        // allocate and block; the bounded reader must fail from the header.
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            read_frame_limited(&mut reader, |_| Ok(MAX_HELLO_FRAME)),
        )
        .await
        .expect("reader blocked waiting for an oversized payload");
        assert!(result.unwrap_err().to_string().contains("exceeds limit"));
    }

    #[test]
    fn tensor_roundtrip_f32() {
        let vals = vec![1.0f32, -2.5, 3.25e-8];
        let mut bytes = Vec::new();
        encode_tensor(DTYPE_F32, &vals, &mut bytes).unwrap();
        let mut back = Vec::new();
        decode_tensor(DTYPE_F32, &bytes, &mut back).unwrap();
        assert_eq!(vals, back);
    }

    #[test]
    fn tensor_roundtrip_bf16() {
        let vals = vec![1.0f32, -2.5, 0.15625];
        let mut bytes = Vec::new();
        encode_tensor(DTYPE_BF16, &vals, &mut bytes).unwrap();
        let mut back = Vec::new();
        decode_tensor(DTYPE_BF16, &bytes, &mut back).unwrap();
        assert_eq!(vals, back); // all exactly representable in bf16
    }

    #[test]
    fn tensor_decoders_reject_non_finite_values_and_q4_scales() {
        let mut out = Vec::new();
        assert!(decode_tensor(DTYPE_F32, &f32::INFINITY.to_le_bytes(), &mut out).is_err());
        assert!(decode_tensor(DTYPE_BF16, &0x7fc0u16.to_le_bytes(), &mut out).is_err());

        let mut q4 = vec![0u8; 4 + Q4_BLOCK / 2];
        q4[..4].copy_from_slice(&(-1.0f32).to_le_bytes());
        assert!(decode_q4(&q4, 1, &mut out).is_err());
        q4[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(decode_q4(&q4, 1, &mut out).is_err());
    }

    /// Golden vector matching yeto/tensor_io.py quantize_q4 output for
    /// [1.0, -0.5, 0.25, 0.0] (tests/test_tensor_io.py cross-checks the
    /// same bytes from the Python encoder).
    #[test]
    fn q4_golden_vector() {
        let mut bytes = vec![0u8; 132];
        bytes[..4].copy_from_slice(&1.0f32.to_le_bytes()); // block scale
        bytes[4] = 0xE7; // 1.0 -> level 7; -0.5 -> sign|level 6
        bytes[5] = 0x05; // 0.25 -> level 5; 0.0 -> 0
        let mut out = Vec::new();
        decode_q4(&bytes, 4, &mut out).unwrap();
        assert_eq!(out, vec![1.0f32, -0.5, 0.25, 0.0]);
    }

    #[test]
    fn q4_multi_block_and_truncation() {
        // 300 values -> 2 blocks; only the first value of block 2 is set.
        let mut bytes = vec![0u8; 2 * 132];
        bytes[..4].copy_from_slice(&2.0f32.to_le_bytes());
        bytes[4] = 0x07; // 2.0 * 2^0 = 2.0
        bytes[132..136].copy_from_slice(&4.0f32.to_le_bytes());
        bytes[136] = 0x0E; // -(4.0 * 2^-1) = -2.0
        let mut out = Vec::new();
        decode_q4(&bytes, 300, &mut out).unwrap();
        assert_eq!(out.len(), 300);
        assert_eq!(out[0], 2.0);
        assert_eq!(out[256], -2.0);
        assert!(out[1..256].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q4_rejects_bad_length() {
        let mut out = Vec::new();
        assert!(decode_q4(&[0u8; 131], 4, &mut out).is_err());
    }

    #[test]
    fn bulk_dtype_maps_q4_to_bf16() {
        assert_eq!(bulk_dtype(DTYPE_Q4), DTYPE_BF16);
        assert_eq!(bulk_dtype(DTYPE_BF16), DTYPE_BF16);
        assert_eq!(bulk_dtype(DTYPE_F32), DTYPE_F32);
        assert_eq!(bulk_dtype(DTYPE_F32_PUSH_BF16), DTYPE_F32);
        assert_eq!(push_dtype(DTYPE_F32_PUSH_BF16), DTYPE_BF16);
        assert_eq!(push_dtype(DTYPE_F32), DTYPE_F32);
    }

    #[test]
    fn final_manifest_and_ack_have_versioned_exact_shapes() {
        assert_eq!(MSG_ERROR, 10);
        assert_eq!(MSG_FINAL_MANIFEST, 11);
        assert_eq!(MSG_FINAL_ACK, 12);
        assert_eq!(MSG_FINAL_FRAGMENT, 13);
        let manifest = encode_final_manifest(17, &[15, 16, 17]);
        let mut expected = Vec::new();
        expected.extend_from_slice(&FINALIZATION_REVISION.to_le_bytes());
        expected.extend_from_slice(&17u64.to_le_bytes());
        expected.extend_from_slice(&3u32.to_le_bytes());
        for version in [15u64, 16, 17] {
            expected.extend_from_slice(&version.to_le_bytes());
        }
        assert_eq!(manifest, expected);

        let mut ack = Vec::new();
        ack.extend_from_slice(&FINALIZATION_REVISION.to_le_bytes());
        ack.extend_from_slice(&17u64.to_le_bytes());
        assert_eq!(decode_final_ack(&ack).unwrap(), 17);
        ack[0] = 2;
        assert!(decode_final_ack(&ack)
            .unwrap_err()
            .to_string()
            .contains("unsupported finalization revision"));
    }

    #[test]
    fn budget_done_has_a_fixed_shape() {
        let mut report = Vec::new();
        report.extend_from_slice(&17u64.to_le_bytes());
        assert_eq!(decode_budget_done(&report).unwrap(), 17);
        report.push(0);
        assert!(decode_budget_done(&report).is_err());
    }

    /// Encode `vals` so blocks restart at every segment (test counterpart of
    /// the Python tensor-block codecs).
    fn encode_segments(vals: &[f32], segments: &[usize], block: usize, encode: fn(&[f32], usize) -> Vec<u8>) -> Vec<u8> {
        let mut out = (block as u32).to_le_bytes().to_vec();
        let mut at = 0;
        for &n in segments {
            out.extend_from_slice(&encode(&vals[at..at + n], block)[4..]);
            at += n;
        }
        out
    }

    #[test]
    fn tensor_block_flag_is_valid_only_on_block_quantized_push_dtypes() {
        for base in [DTYPE_F32_PUSH_I8, DTYPE_F32_PUSH_Q4, DTYPE_F32_PUSH_I4] {
            assert!(valid_session_dtype(base | DTYPE_FLAG_TENSOR_BLOCKS));
            assert!(valid_session_dtype(base | DTYPE_FLAG_TENSOR_BLOCKS | DTYPE_FLAG_BCAST_DELTA));
            assert_eq!(session_base(base | DTYPE_FLAG_TENSOR_BLOCKS), base);
            assert_eq!(bulk_dtype(base | DTYPE_FLAG_TENSOR_BLOCKS), DTYPE_F32);
        }
        for base in [DTYPE_F32, DTYPE_BF16, DTYPE_Q4, DTYPE_F32_PUSH_BF16] {
            assert!(!valid_session_dtype(base | DTYPE_FLAG_TENSOR_BLOCKS));
        }
        assert!(!tensor_blocks(DTYPE_F32_PUSH_I8));
    }

    #[test]
    fn tensor_blocks_restart_per_segment_for_int8_and_int4() {
        let segments = [3usize, 130, 1, 257];
        let vals: Vec<f32> = (0..391).map(|i| ((i as f32) * 0.37).sin() * (1.0 + (i % 7) as f32)).collect();
        let dtype8 = DTYPE_F32_PUSH_I8 | DTYPE_FLAG_TENSOR_BLOCKS;
        let dtype4 = DTYPE_F32_PUSH_I4 | DTYPE_FLAG_TENSOR_BLOCKS;
        for (dtype, wire, bound) in [
            (dtype8, encode_segments(&vals, &segments, 128, encode_i8), 1.0 / 254.0),
            (dtype4, encode_segments(&vals, &segments, 64, encode_i4), 1.0 / 14.0),
        ] {
            assert!(wire.len() <= max_push_nbytes_segments(dtype, &segments).unwrap());
            let mut out = Vec::new();
            decode_push_segments(dtype, &wire, &segments, &mut out).unwrap();
            assert_eq!(out.len(), vals.len());
            let mut at = 0;
            for &n in &segments {
                let absmax = vals[at..at + n].iter().fold(0f32, |m, v| m.max(v.abs()));
                for k in at..at + n {
                    assert!((out[k] - vals[k]).abs() <= absmax * bound * 1.001 + 1e-6);
                }
                at += n;
            }
            // The whole-fragment layout has a different size, so it is refused.
            assert!(decode_push_segments(dtype, &wire, &[vals.len()], &mut out).is_err());
        }
    }

    #[test]
    fn replicated_tensor_decodes_identically_next_to_different_neighbours() {
        let tensor: Vec<f32> = (0..70).map(|i| 0.01 * (i as f32 - 30.0)).collect();
        let loud = vec![50.0f32; 45];
        let quiet = vec![1e-4f32; 91];
        for (dtype, block, encode) in [
            (DTYPE_F32_PUSH_I8, 128usize, encode_i8 as fn(&[f32], usize) -> Vec<u8>),
            (DTYPE_F32_PUSH_I4, 64, encode_i4),
        ] {
            let dtype = dtype | DTYPE_FLAG_TENSOR_BLOCKS;
            let decode_with = |before: &[f32], after: &[f32]| {
                let vals: Vec<f32> = [before, &tensor, after].concat();
                let segs = [before.len(), tensor.len(), after.len()];
                let wire = encode_segments(&vals, &segs, block, encode);
                let mut out = Vec::new();
                decode_push_segments(dtype, &wire, &segs, &mut out).unwrap();
                out[before.len()..before.len() + tensor.len()].to_vec()
            };
            assert_eq!(decode_with(&loud, &quiet), decode_with(&quiet, &loud));
        }
    }

    #[test]
    fn q4_tensor_blocks_pad_every_segment_and_reject_the_fragment_layout() {
        let block: Vec<u8> = [1.0f32.to_le_bytes().to_vec(), vec![0x77; Q4_BLOCK / 2]].concat();
        let segments = [3usize, 300];
        let wire = block.repeat(3);
        assert_eq!(wire.len(), q4_nbytes_segments(&segments).unwrap());
        let mut out = Vec::new();
        decode_push_segments(DTYPE_F32_PUSH_Q4 | DTYPE_FLAG_TENSOR_BLOCKS, &wire, &segments, &mut out).unwrap();
        assert_eq!(out, vec![1.0f32; 303]);
        assert!(decode_push_segments(DTYPE_F32_PUSH_Q4, &wire, &[303], &mut out).is_err());
        assert_eq!(push_segments(DTYPE_F32_PUSH_Q4, 303, &[3, 300]).unwrap(), vec![303]);
        assert_eq!(
            push_segments(DTYPE_F32_PUSH_Q4 | DTYPE_FLAG_TENSOR_BLOCKS, 303, &[3, 300]).unwrap(),
            vec![3, 300]
        );
    }

    fn expert_tag_block(tags: &[(u64, u64, u32)]) -> Vec<u8> {
        let mut out = (tags.len() as u32).to_le_bytes().to_vec();
        for &(routed, layer, experts) in tags {
            out.extend_from_slice(&routed.to_le_bytes());
            out.extend_from_slice(&layer.to_le_bytes());
            out.extend_from_slice(&experts.to_le_bytes());
        }
        out
    }

    #[test]
    fn expert_tags_scale_by_share_of_an_even_split() {
        let mut bytes = expert_tag_block(&[(0, 0, 0), (30, 40, 4), (0, 40, 4), (10, 40, 4), (5, 0, 0)]);
        bytes.extend_from_slice(b"rest");
        // The dense tag with tokens is refused.
        assert!(decode_expert_tags(&mut Reader(&bytes), 5).is_err());
        let mut bytes = expert_tag_block(&[(0, 0, 0), (30, 40, 4), (0, 40, 4), (10, 40, 4), (0, 0, 2)]);
        bytes.extend_from_slice(b"rest");
        let mut r = Reader(&bytes);
        assert_eq!(decode_expert_tags(&mut r, 5).unwrap(), vec![1.0, 3.0, 0.0, 1.0, 0.0]);
        assert_eq!(r.rest(), b"rest");
        assert_eq!(expert_tags_nbytes(5).unwrap(), bytes.len() - 4);
    }

    #[test]
    fn expert_tags_refuse_bad_blocks() {
        let block = expert_tag_block(&[(1, 2, 4)]);
        assert!(decode_expert_tags(&mut Reader(&block), 2).is_err());
        assert!(decode_expert_tags(&mut Reader(&block[..block.len() - 1]), 1).is_err());
        let over = expert_tag_block(&[(3, 2, 4)]);
        assert!(decode_expert_tags(&mut Reader(&over), 1).is_err());
    }
}
