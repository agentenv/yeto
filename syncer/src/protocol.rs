//! Wire protocol (see docs/PROTOCOL.md). Little-endian framing over TCP.

use anyhow::{bail, Result};
use tokio::io::AsyncReadExt;

pub const MAGIC: u32 = 0xD170_C0DE;

pub const MSG_HELLO: u8 = 1;
pub const MSG_INIT_PARAMS: u8 = 2;
pub const MSG_PULL_REQ: u8 = 3;
pub const MSG_PUSH_FRAGMENT: u8 = 4;
pub const MSG_BCAST_FRAGMENT: u8 = 5;
pub const MSG_HEARTBEAT: u8 = 6;
pub const MSG_SHUTDOWN: u8 = 7;
pub const MSG_DATA_HELLO: u8 = 8;
pub const MSG_CHUNK: u8 = 9;
/// SCAFFOLD-lite (docs/OTHER_OPTIMIZERS.md #5): the token-normalized mean
/// control vector `c` for one fragment, sent right after its BCAST_FRAGMENT
/// when `--inner-control-variate scaffold_lite` is set. Same envelope as
/// BCAST_FRAGMENT (fid u32, version u64, bulk-wire tensor bytes). Never emitted on
/// the default path, so learners that do not enable control variates ignore it.
pub const MSG_BCAST_CONTROL: u8 = 10;

pub const DTYPE_F32: u8 = 1;
pub const DTYPE_BF16: u8 = 2;
/// Session dtype 3: PUSH_FRAGMENT payloads are block-quantized 4-bit E3M0
/// *deltas* against the fragment value at base_version; INIT_PARAMS and
/// BCAST_FRAGMENT travel as bf16 (see `bulk_dtype`, docs/PROTOCOL.md v3).
pub const DTYPE_Q4: u8 = 3;

/// Values per q4 scale block (f32 absmax scale + 128 nibble bytes each).
pub const Q4_BLOCK: usize = 256;

/// Dtype of INIT_PARAMS/BCAST_FRAGMENT tensors for a session dtype: q4
/// applies only to push deltas; full parameter payloads stay bf16.
pub fn bulk_dtype(dtype: u8) -> u8 {
    if dtype == DTYPE_Q4 { DTYPE_BF16 } else { dtype }
}

/// Hard cap on a single frame; a fragment of a very large model can still be
/// hundreds of MB, but anything past this is a corrupt length prefix.
const MAX_FRAME: u64 = 64 * 1024 * 1024 * 1024;

pub struct Frame {
    pub msg_type: u8,
    pub payload: Vec<u8>,
}

pub async fn read_frame<R: AsyncReadExt + Unpin>(r: &mut R) -> Result<Frame> {
    let magic = r.read_u32_le().await?;
    if magic != MAGIC {
        bail!("bad magic 0x{magic:08x}");
    }
    let msg_type = r.read_u8().await?;
    let len = r.read_u64_le().await?;
    if len > MAX_FRAME {
        bail!("frame too large: {len}");
    }
    let mut payload = vec![0u8; len as usize];
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
}

/// Decode a tensor payload into f32s.
pub fn decode_tensor(dtype: u8, bytes: &[u8], out: &mut Vec<f32>) -> Result<()> {
    out.clear();
    match dtype {
        DTYPE_F32 => {
            if bytes.len() % 4 != 0 {
                bail!("f32 tensor byte length not divisible by 4");
            }
            out.reserve(bytes.len() / 4);
            for c in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes(c.try_into().unwrap()));
            }
        }
        DTYPE_BF16 => {
            if bytes.len() % 2 != 0 {
                bail!("bf16 tensor byte length not divisible by 2");
            }
            out.reserve(bytes.len() / 2);
            for c in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes(c.try_into().unwrap());
                out.push(f32::from_bits((bits as u32) << 16));
            }
        }
        _ => bail!("unknown dtype {dtype}"),
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
    let blocks = numel.div_ceil(Q4_BLOCK);
    let block_bytes = 4 + Q4_BLOCK / 2;
    if bytes.len() != blocks * block_bytes {
        bail!(
            "q4 payload has {} bytes, expected {} for {numel} values",
            bytes.len(),
            blocks * block_bytes
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
    out.reserve(numel);
    for (b, chunk) in bytes.chunks_exact(block_bytes).enumerate() {
        let scale = f32::from_le_bytes(chunk[..4].try_into().unwrap());
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
}
