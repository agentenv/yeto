//! SHA-256 with the ARMv8 SHA-2 instructions when the CPU has them.
//!
//! The vendored `sha2` crate only uses those instructions behind its `asm`
//! feature, which pulls an extra build-time crate; without it every digest
//! runs the portable code (about 0.25 GB/s on a Kunpeng 920 versus
//! 1.6 GB/s with the instructions). The syncer hashes every INIT_PARAMS
//! payload on its read loop, so a 120B model's 483 GB of f32 INIT took over
//! half an hour to admit. Digests are byte-identical to `sha2::Sha256`; other
//! targets and CPUs without the extension use `sha2` itself.

use sha2::{Digest, Sha256};

#[cfg(target_arch = "aarch64")]
mod hw {
    use std::arch::aarch64::*;

    pub const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    pub fn available() -> bool {
        std::arch::is_aarch64_feature_detected!("sha2")
    }

    /// Compress whole 64-byte blocks into `state`.
    ///
    /// # Safety
    /// The CPU must have the SHA-2 extension (`available()`), and
    /// `blocks.len()` must be a multiple of 64.
    #[target_feature(enable = "sha2")]
    pub unsafe fn compress(state: &mut [u32; 8], blocks: &[u8]) {
        let mut abcd = vld1q_u32(state.as_ptr());
        let mut efgh = vld1q_u32(state.as_ptr().add(4));
        for block in blocks.chunks_exact(64) {
            let (abcd0, efgh0) = (abcd, efgh);
            let p = block.as_ptr();
            let mut w = [
                vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p))),
                vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p.add(16)))),
                vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p.add(32)))),
                vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p.add(48)))),
            ];
            for i in 0..16 {
                let wk = vaddq_u32(w[i % 4], vld1q_u32(K.as_ptr().add(4 * i)));
                let prev = abcd;
                abcd = vsha256hq_u32(abcd, efgh, wk);
                efgh = vsha256h2q_u32(efgh, prev, wk);
                if i < 12 {
                    // W[4(i+4)..] from W[4i..4i+16), into the slot just consumed.
                    w[i % 4] = vsha256su1q_u32(
                        vsha256su0q_u32(w[i % 4], w[(i + 1) % 4]),
                        w[(i + 2) % 4],
                        w[(i + 3) % 4],
                    );
                }
            }
            abcd = vaddq_u32(abcd, abcd0);
            efgh = vaddq_u32(efgh, efgh0);
        }
        vst1q_u32(state.as_mut_ptr(), abcd);
        vst1q_u32(state.as_mut_ptr().add(4), efgh);
    }
}

const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Incremental SHA-256, the same bytes as `sha2::Sha256`.
pub enum Hasher {
    #[cfg(target_arch = "aarch64")]
    Hw { state: [u32; 8], buf: [u8; 64], used: usize, len: u64 },
    Soft(Sha256),
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher {
    pub fn new() -> Self {
        #[cfg(target_arch = "aarch64")]
        if hw::available() {
            return Self::Hw { state: H0, buf: [0; 64], used: 0, len: 0 };
        }
        Self::Soft(Sha256::new())
    }

    /// The portable implementation, whatever the CPU (tests compare to it).
    #[cfg(test)]
    fn soft() -> Self {
        Self::Soft(Sha256::new())
    }

    pub fn update(&mut self, mut data: &[u8]) {
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Hw { state, buf, used, len } => {
                *len += data.len() as u64;
                if *used > 0 {
                    let take = (64 - *used).min(data.len());
                    buf[*used..*used + take].copy_from_slice(&data[..take]);
                    *used += take;
                    data = &data[take..];
                    if *used < 64 {
                        return;
                    }
                    // SAFETY: the Hw variant exists only when hw::available().
                    unsafe { hw::compress(state, &buf[..]) };
                    *used = 0;
                }
                let whole = data.len() - data.len() % 64;
                // SAFETY: as above; `whole` is a multiple of 64.
                unsafe { hw::compress(state, &data[..whole]) };
                let rest = &data[whole..];
                buf[..rest.len()].copy_from_slice(rest);
                *used = rest.len();
            }
            Self::Soft(h) => h.update(data),
        }
    }

    pub fn finalize(self) -> [u8; 32] {
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Hw { mut state, mut buf, used, len } => {
                buf[used] = 0x80;
                buf[used + 1..].fill(0);
                if used >= 56 {
                    // SAFETY: the Hw variant exists only when hw::available().
                    unsafe { hw::compress(&mut state, &buf[..]) };
                    buf.fill(0);
                }
                buf[56..].copy_from_slice(&(len.wrapping_mul(8)).to_be_bytes());
                // SAFETY: as above.
                unsafe { hw::compress(&mut state, &buf[..]) };
                let mut out = [0u8; 32];
                for (chunk, word) in out.chunks_exact_mut(4).zip(state) {
                    chunk.copy_from_slice(&word.to_be_bytes());
                }
                out
            }
            Self::Soft(h) => h.finalize().into(),
        }
    }
}

/// SHA-256 of `data`.
pub fn digest(data: &[u8]) -> [u8; 32] {
    let mut h = Hasher::new();
    h.update(data);
    h.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect()
    }

    fn reference(data: &[u8]) -> [u8; 32] {
        Sha256::digest(data).into()
    }

    #[test]
    fn digest_matches_sha2_at_every_padding_boundary() {
        for n in (0..=260).chain([511, 512, 513, 4096, 65_535, 65_536, 1 << 20, (1 << 20) + 63]) {
            let data = bytes(n, n as u64);
            assert_eq!(digest(&data), reference(&data), "length {n}");
        }
    }

    #[test]
    fn known_vectors() {
        let hex = |d: [u8; 32]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(hex(digest(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hex(digest(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(
            hex(digest(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn split_updates_match_one_update() {
        let data = bytes(10_000, 7);
        for split in [1usize, 3, 63, 64, 65, 127, 1000] {
            let mut h = Hasher::new();
            for chunk in data.chunks(split) {
                h.update(chunk);
            }
            h.update(&[]);
            assert_eq!(h.finalize(), reference(&data), "chunks of {split}");
        }
        let mut soft = Hasher::soft();
        soft.update(&data);
        assert_eq!(soft.finalize(), reference(&data));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn the_hardware_path_is_taken_when_the_cpu_has_it() {
        assert_eq!(matches!(Hasher::new(), Hasher::Hw { .. }), hw::available());
    }
}
