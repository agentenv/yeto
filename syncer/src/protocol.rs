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

/// Values per q4 scale block (f32 absmax scale + 128 nibble bytes each).
pub const Q4_BLOCK: usize = 256;

/// Dtype of INIT_PARAMS/BCAST_FRAGMENT tensors for a session dtype: q4
/// applies only to push deltas; full parameter payloads stay bf16.
pub fn bulk_dtype(dtype: u8) -> u8 {
    if dtype == DTYPE_Q4 {
        DTYPE_BF16
    } else {
        dtype
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

/// Decode a q4 delta payload (yeto/tensor_io.py `quantize_q4`) into f32s.
///
/// Per block of `Q4_BLOCK` values: f32 absmax scale, then 128 bytes of
/// packed nibbles (two values per byte, low nibble first). Nibble: bit 3 =
/// sign, bits 0-2 = level; level 0 is exactly zero, level L in 1..=7
/// decodes to sign * 2^(L-7) * scale. The last block is zero-padded to
/// `Q4_BLOCK`; `numel` truncates the tail.
pub fn decode_q4(bytes: &[u8], numel: usize, out: &mut Vec<f32>) -> Result<()> {
    let block_bytes = 4 + Q4_BLOCK / 2;
    let expected = q4_nbytes(numel)?;
    if bytes.len() != expected {
        bail!(
            "q4 payload has {} bytes, expected {} for {numel} values",
            bytes.len(),
            expected
        );
    }
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
    out.clear();
    out.try_reserve(numel)
        .map_err(|error| anyhow::anyhow!("cannot allocate q4 tensor: {error}"))?;
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
    fn reserved_message_ids_are_stable() {
        assert_eq!(MSG_ERROR, 10);
        assert_eq!(MSG_FINAL_MANIFEST, 11);
        assert_eq!(MSG_FINAL_ACK, 12);
        assert_eq!(MSG_FINAL_FRAGMENT, 13);
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
}
