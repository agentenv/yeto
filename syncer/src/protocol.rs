//! Wire protocol (see docs/PROTOCOL.md). Little-endian framing over TCP.

use anyhow::{bail, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub const MAGIC: u32 = 0xD170_C0DE;

pub const MSG_HELLO: u8 = 1;
pub const MSG_INIT_PARAMS: u8 = 2;
pub const MSG_PUSH_FRAGMENT: u8 = 3;
pub const MSG_BCAST_FRAGMENT: u8 = 4;
pub const MSG_HEARTBEAT: u8 = 5;
pub const MSG_SHUTDOWN: u8 = 6;

pub const DTYPE_F32: u8 = 1;
pub const DTYPE_BF16: u8 = 2;

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

pub async fn write_frame<W: AsyncWriteExt + Unpin>(
    w: &mut W,
    msg_type: u8,
    payload: &[u8],
) -> Result<()> {
    let mut header = [0u8; 13];
    header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    header[4] = msg_type;
    header[5..13].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    w.write_all(&header).await?;
    w.write_all(payload).await?;
    w.flush().await?;
    Ok(())
}

/// Cursor helpers for parsing payloads.
pub struct Reader<'a>(pub &'a [u8]);

impl<'a> Reader<'a> {
    pub fn u8(&mut self) -> Result<u8> {
        let (v, rest) = self.0.split_first().ok_or_else(|| anyhow::anyhow!("eof"))?;
        self.0 = rest;
        Ok(*v)
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
}
