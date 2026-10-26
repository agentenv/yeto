//! bf16 downlink deltas for BCAST_FRAGMENT (S26 E3, session flag
//! `DTYPE_FLAG_BCAST_DELTA`).
//!
//! Every learner of a session holds, per fragment, an f32 *anchor* that is
//! bit-identical to the syncer's copy here. A broadcast of a new version is
//! either
//!
//! * `FULL`: the f32 values; the anchor becomes exactly those values, or
//! * `DELTA`: `d = bf16(θ − anchor)`; both sides set `anchor += d` in f32
//!   (IEEE addition is exact-rounded, so both sides get the same bits).
//!
//! Each delta is taken against the current anchor, so the rounding error does
//! not accumulate: `|θ − anchor| <= |θ − anchor_prev| * 2^-9` per element
//! after every delta. The learner trains from the anchor; its next push is
//! relative to the anchor, and the syncer adds that push to its exact θ, so
//! θ stays the exact sum of all pushes.
//!
//! Drift guard: every `full_every`-th broadcast of a fragment is FULL; a
//! re-sent fragment (HELLO/reconnect recovery) is always FULL; and every
//! header carries the SHA-256 of the f32 anchor after the update. A learner
//! whose anchor version or digest does not match drops its connection; the
//! reconnect recovery then sends FULL.
//!
//! Anchors live only in memory: a resumed syncer starts with none, so its
//! first broadcast of every fragment is FULL.
//!
//! Header after the usual `fragment_id u32, version u64` prefix:
//! `kind u8 (0 FULL, 1 DELTA), base_version u64, anchor_sha256 [32]`.
//! The tensor that follows is f32 (FULL) or bf16 (DELTA).

use std::sync::Arc;

use anyhow::{bail, Result};
use sha2::{Digest, Sha256};

use crate::protocol::{DTYPE_BF16, DTYPE_F32};

pub const KIND_FULL: u8 = 0;
pub const KIND_DELTA: u8 = 1;
/// fragment_id + version + kind + base_version + digest.
pub const PREFIX_LEN: usize = 4 + 8 + 1 + 8 + 32;

struct Anchor {
    version: u64,
    values: Arc<Vec<f32>>,
    digest: [u8; 32],
    deltas_since_full: u64,
}

/// One encoded BCAST: header prefix plus tensor values in `dtype` (DELTA
/// values are already bf16-representable, so encoding them is exact).
pub struct EncodedBcast {
    pub prefix: Vec<u8>,
    pub dtype: u8,
    pub values: Arc<Vec<f32>>,
}

impl EncodedBcast {
    #[cfg(test)]
    pub fn kind(&self) -> u8 {
        self.prefix[12]
    }
}

pub struct DownlinkAnchors {
    full_every: u64,
    anchors: Vec<Option<Anchor>>,
}

pub fn digest_f32(values: &[f32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    // Hash in bounded slices to avoid a second full-size byte buffer.
    let mut buf = Vec::with_capacity(4 * 65536);
    for chunk in values.chunks(65536) {
        buf.clear();
        for v in chunk {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        hasher.update(&buf);
    }
    hasher.finalize().into()
}

/// S27 G-FH: the header of a FULL that carries `params` exactly (no anchor
/// at this version): what `resend` sends then, without copying the values.
pub fn exact_full_prefix(fragment: usize, version: u64, params: &[f32]) -> Vec<u8> {
    prefix(fragment, version, KIND_FULL, version, &digest_f32(params))
}

fn prefix(fragment: usize, version: u64, kind: u8, base_version: u64, digest: &[u8; 32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(PREFIX_LEN);
    out.extend_from_slice(&(fragment as u32).to_le_bytes());
    out.extend_from_slice(&version.to_le_bytes());
    out.push(kind);
    out.extend_from_slice(&base_version.to_le_bytes());
    out.extend_from_slice(digest);
    out
}

impl DownlinkAnchors {
    pub fn new(fragments: usize, full_every: u64) -> Result<Self> {
        if full_every < 2 {
            bail!("--bcast-delta-full-every must be at least 2");
        }
        Ok(Self {
            full_every,
            anchors: (0..fragments).map(|_| None).collect(),
        })
    }

    /// Encode the broadcast of a newly committed `version` of fragment `p`
    /// and advance the shared anchor. Call exactly once per new version.
    pub fn publish(&mut self, p: usize, version: u64, params: &[f32]) -> Result<EncodedBcast> {
        let slot = self
            .anchors
            .get_mut(p)
            .ok_or_else(|| anyhow::anyhow!("downlink anchor for unknown fragment {p}"))?;
        let delta_ok = matches!(
            slot,
            Some(a) if a.deltas_since_full + 1 < self.full_every
                && a.values.len() == params.len()
                && a.version < version
        );
        if !delta_ok {
            return Ok(Self::reset(slot, p, version, params));
        }
        let anchor = slot.as_mut().unwrap();
        let base_version = anchor.version;
        let values = Arc::make_mut(&mut anchor.values);
        let mut delta = Vec::new();
        delta
            .try_reserve_exact(params.len())
            .map_err(|error| anyhow::anyhow!("cannot allocate downlink delta: {error}"))?;
        for (a, t) in values.iter_mut().zip(params) {
            let d = half::bf16::from_f32(t - *a).to_f32();
            if !d.is_finite() {
                // bf16 overflow of a huge step: fall back to FULL.
                return Ok(Self::reset(slot, p, version, params));
            }
            delta.push(d);
            *a += d;
        }
        anchor.version = version;
        anchor.digest = digest_f32(values);
        anchor.deltas_since_full += 1;
        Ok(EncodedBcast {
            prefix: prefix(p, version, KIND_DELTA, base_version, &anchor.digest),
            dtype: DTYPE_BF16,
            values: Arc::new(delta),
        })
    }

    /// Encode a newly committed `version` of fragment `p` as FULL (exact f32)
    /// and reset the shared anchor to it.
    pub fn publish_full(&mut self, p: usize, version: u64, params: &[f32]) -> EncodedBcast {
        match self.anchors.get_mut(p) {
            Some(slot) => Self::reset(slot, p, version, params),
            None => EncodedBcast {
                prefix: prefix(p, version, KIND_FULL, version, &digest_f32(params)),
                dtype: DTYPE_F32,
                values: Arc::new(params.to_vec()),
            },
        }
    }

    /// FULL re-send for recovery (HELLO / reconnect): the anchor every other
    /// learner holds, when it is at `version`; otherwise the exact params.
    pub fn resend(&self, p: usize, version: u64, params: &[f32]) -> EncodedBcast {
        self.anchored_full(p, version).unwrap_or_else(|| EncodedBcast {
            prefix: exact_full_prefix(p, version, params),
            dtype: DTYPE_F32,
            values: Arc::new(params.to_vec()),
        })
    }

    /// S27 G-FH: the FULL re-send of `version` when the shared anchor is at
    /// that version (no copy, no hashing); None otherwise, and the exact
    /// params then go out FULL under `exact_full_prefix`.
    pub fn anchored_full(&self, p: usize, version: u64) -> Option<EncodedBcast> {
        match self.anchors.get(p).and_then(Option::as_ref) {
            Some(anchor) if anchor.version == version => Some(EncodedBcast {
                prefix: prefix(p, version, KIND_FULL, version, &anchor.digest),
                dtype: DTYPE_F32,
                values: Arc::clone(&anchor.values),
            }),
            _ => None,
        }
    }

    fn reset(slot: &mut Option<Anchor>, p: usize, version: u64, params: &[f32]) -> EncodedBcast {
        let values = Arc::new(params.to_vec());
        let digest = digest_f32(&values);
        *slot = Some(Anchor {
            version,
            values: Arc::clone(&values),
            digest,
            deltas_since_full: 0,
        });
        EncodedBcast {
            prefix: prefix(p, version, KIND_FULL, version, &digest),
            dtype: DTYPE_F32,
            values,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A learner-side mirror: applies broadcasts exactly as yeto/protocol.py.
    fn apply(anchor: &mut Option<(u64, Vec<f32>)>, enc: &EncodedBcast) -> Result<()> {
        let version = u64::from_le_bytes(enc.prefix[4..12].try_into().unwrap());
        let base = u64::from_le_bytes(enc.prefix[13..21].try_into().unwrap());
        let digest: [u8; 32] = enc.prefix[21..53].try_into().unwrap();
        let next = match enc.kind() {
            KIND_FULL => enc.values.to_vec(),
            KIND_DELTA => {
                let Some((held, values)) = anchor.as_ref() else { bail!("no anchor") };
                if *held != base {
                    bail!("anchor version {held} != base {base}");
                }
                values.iter().zip(enc.values.iter()).map(|(a, d)| a + d).collect()
            }
            _ => bail!("bad kind"),
        };
        if digest_f32(&next) != digest {
            bail!("digest mismatch");
        }
        *anchor = Some((version, next));
        Ok(())
    }

    fn params(seed: u64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f64 * 0.37 + seed as f64 * 1.3).sin() * 3.0) as f32)
            .collect()
    }

    #[test]
    fn deltas_track_params_within_bf16_error_and_refresh_every_m() {
        let mut syncer = DownlinkAnchors::new(1, 4).unwrap();
        let mut learner = None;
        let mut theta = params(0, 1000);
        let mut kinds = vec![];
        for version in 1..=9u64 {
            let step = params(version, 1000);
            let prev_anchor = learner.as_ref().map(|(_, v): &(u64, Vec<f32>)| v.clone());
            for (t, s) in theta.iter_mut().zip(&step) {
                *t += 0.01 * s;
            }
            let enc = syncer.publish(0, version, &theta).unwrap();
            kinds.push(enc.kind());
            apply(&mut learner, &enc).unwrap();
            let (_, held) = learner.as_ref().unwrap();
            if enc.kind() == KIND_FULL {
                assert_eq!(held, &theta);
            } else {
                let prev = prev_anchor.unwrap();
                for ((h, t), a) in held.iter().zip(&theta).zip(&prev) {
                    // one bf16 rounding of (θ − anchor_prev): rel. 2^-8 bound
                    assert!((h - t).abs() <= (t - a).abs() / 256.0 + 2.0 * f32::EPSILON * t.abs());
                }
            }
        }
        assert_eq!(
            kinds,
            vec![KIND_FULL, KIND_DELTA, KIND_DELTA, KIND_DELTA, KIND_FULL, KIND_DELTA, KIND_DELTA, KIND_DELTA, KIND_FULL]
        );
    }

    #[test]
    fn resend_is_the_shared_anchor_and_lets_a_reconnected_learner_continue() {
        let mut syncer = DownlinkAnchors::new(1, 100).unwrap();
        let mut a = None;
        let theta1 = params(1, 64);
        apply(&mut a, &syncer.publish(0, 1, &theta1).unwrap()).unwrap();
        let theta2: Vec<f32> = theta1.iter().map(|v| v + 0.123).collect();
        apply(&mut a, &syncer.publish(0, 2, &theta2).unwrap()).unwrap();
        // A fresh learner (reconnect) gets the anchor, not θ, as FULL.
        let mut b = None;
        let full = syncer.resend(0, 2, &theta2);
        assert_eq!(full.kind(), KIND_FULL);
        apply(&mut b, &full).unwrap();
        assert_eq!(a, b);
        let theta3: Vec<f32> = theta2.iter().map(|v| v - 0.05).collect();
        let enc = syncer.publish(0, 3, &theta3).unwrap();
        assert_eq!(enc.kind(), KIND_DELTA);
        apply(&mut a, &enc).unwrap();
        apply(&mut b, &enc).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn a_learner_with_a_wrong_anchor_detects_it() {
        let mut syncer = DownlinkAnchors::new(1, 100).unwrap();
        let mut a = None;
        apply(&mut a, &syncer.publish(0, 1, &params(1, 32)).unwrap()).unwrap();
        // Corrupt the learner's anchor bits: the digest check must fail.
        if let Some((_, v)) = a.as_mut() {
            v[3] += 1e-3;
        }
        let enc = syncer.publish(0, 2, &params(2, 32)).unwrap();
        assert!(apply(&mut a, &enc).unwrap_err().to_string().contains("digest"));
        // A learner that skipped a version fails the base-version check.
        let mut stale = Some((0u64, params(1, 32)));
        let enc = syncer.publish(0, 3, &params(3, 32)).unwrap();
        assert!(apply(&mut stale, &enc).unwrap_err().to_string().contains("base"));
    }

    #[test]
    fn full_every_below_two_is_refused() {
        assert!(DownlinkAnchors::new(1, 1).is_err());
    }
}
