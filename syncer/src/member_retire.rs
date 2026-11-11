//! MEMBER_RETIRE: the controller retires one rollout island process now
//! (scale-down, evict, a preemption the provider saw), the WorkContract's
//! `retire` on the syncer data path (`yeto/rl/engine/work_contract.py`).
//!
//! One connection, one request, one answer. The request is the first frame
//! (MSG_MEMBER_RETIRE):
//! `u16 version (=1) | u32 learner_id | u8 has_incarnation |
//! u64 incarnation | u16 reason_len | reason (UTF-8, at most
//! MAX_RETIRE_REASON bytes)`, sealed with MEMBER_RETIRE_MAC_DOMAIN when the
//! syncer has an island HMAC key; without a key only a loopback peer (the
//! head itself) is accepted. `has_incarnation = 0` retires the id's current
//! incarnation; an explicit incarnation that never joined is fenced ahead.
//! The answer (MSG_MEMBER_RETIRED): `u8 status (0 retired now, 1 already
//! retired, 2 no current incarnation) | u64 incarnation | u64 reclaimed`
//! (positions the dispenser put back at the queue front, not counted toward
//! `max_redo`). Effects, at once, on the one member table of the process:
//! the dispenser reclaims the holder's positions, the sample relay refuses
//! its PUTs and a subscriber connection of it closes (both check
//! `Members::is_retired`). Idempotent.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{bail, ensure, Result};
use tokio::net::tcp::OwnedWriteHalf;
use tracing::info;

use crate::members::{Holder, SharedMembers};
use crate::protocol::Reader;

pub const MSG_MEMBER_RETIRE: u8 = 57;
pub const MSG_MEMBER_RETIRED: u8 = 58;
pub const MEMBER_RETIRE_VERSION: u16 = 1;
pub const MEMBER_RETIRE_MAC_DOMAIN: &[u8] = b"yeto-member-retire-mac-v1\0";
pub const MAX_RETIRE_REASON: usize = 256;
/// Largest MEMBER_RETIRE payload (body, the longest reason, the HMAC).
pub const MAX_MEMBER_RETIRE: u64 = 1024;

pub const RETIRED_NOW: u8 = 0;
pub const RETIRED_ALREADY: u8 = 1;
pub const RETIRED_NO_MEMBER: u8 = 2;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetireRequest {
    pub learner_id: u32,
    pub incarnation: Option<u64>,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetireAnswer {
    pub status: u8,
    pub incarnation: u64,
    pub reclaimed: u64,
}

#[cfg_attr(not(test), allow(dead_code))] // the controller side encodes (yeto/rl/engine/task_dispense.py)
pub fn encode_retire(request: &RetireRequest) -> Vec<u8> {
    let mut out = Vec::with_capacity(17 + request.reason.len());
    out.extend_from_slice(&MEMBER_RETIRE_VERSION.to_le_bytes());
    out.extend_from_slice(&request.learner_id.to_le_bytes());
    out.push(u8::from(request.incarnation.is_some()));
    out.extend_from_slice(&request.incarnation.unwrap_or(0).to_le_bytes());
    out.extend_from_slice(&(request.reason.len() as u16).to_le_bytes());
    out.extend_from_slice(request.reason.as_bytes());
    out
}

pub fn decode_retire(body: &[u8]) -> Result<RetireRequest> {
    let mut r = Reader(body);
    let version = r.u16()?;
    ensure!(version == MEMBER_RETIRE_VERSION, "MEMBER_RETIRE version {version} (this syncer speaks {MEMBER_RETIRE_VERSION})");
    let learner_id = r.u32()?;
    let has = r.u8()?;
    ensure!(has <= 1, "MEMBER_RETIRE has_incarnation must be 0 or 1");
    let incarnation = r.u64()?;
    let len = usize::from(r.u16()?);
    ensure!(len <= MAX_RETIRE_REASON, "MEMBER_RETIRE reason longer than {MAX_RETIRE_REASON} bytes");
    let reason = std::str::from_utf8(r.take(len)?)?.to_string();
    ensure!(!reason.chars().any(char::is_control), "MEMBER_RETIRE reason has control characters");
    ensure!(r.remaining() == 0, "trailing bytes in MEMBER_RETIRE");
    Ok(RetireRequest { learner_id, incarnation: (has == 1).then_some(incarnation), reason })
}

pub fn encode_answer(answer: &RetireAnswer) -> Vec<u8> {
    let mut out = Vec::with_capacity(17);
    out.push(answer.status);
    out.extend_from_slice(&answer.incarnation.to_le_bytes());
    out.extend_from_slice(&answer.reclaimed.to_le_bytes());
    out
}

/// Check the request's HMAC (`key`), or, without a key, that the peer is the
/// head itself. Returns the body.
pub fn open_retire<'a>(key: Option<&[u8]>, peer: Option<SocketAddr>, payload: &'a [u8]) -> Result<&'a [u8]> {
    let Some(key) = key else {
        ensure!(
            peer.is_some_and(|p| p.ip().is_loopback()),
            "MEMBER_RETIRE from {} refused: without an island HMAC key only the head itself may retire members",
            peer.map_or("an unknown peer".to_string(), |p| p.to_string())
        );
        return Ok(payload);
    };
    ensure!(payload.len() >= crate::hmac::HMAC_LEN, "MEMBER_RETIRE shorter than its HMAC");
    let (body, mac) = payload.split_at(payload.len() - crate::hmac::HMAC_LEN);
    let expected = crate::hmac::hmac_sha256(key, &[MEMBER_RETIRE_MAC_DOMAIN, body].concat());
    let diff = expected.iter().zip(mac).fold(0u8, |acc, (x, y)| acc | (x ^ y));
    ensure!(diff == 0, "MEMBER_RETIRE HMAC mismatch (wrong or missing island HMAC key)");
    Ok(body)
}

/// Apply a retire on the process's member table and, when it runs one, its
/// dispenser (whose reclaim is logged in `task-dispense.jsonl`).
pub fn apply(
    request: &RetireRequest,
    members: &SharedMembers,
    dispense: Option<&crate::dispense_service::DispenseService>,
) -> RetireAnswer {
    let (holder, retired, reclaimed) = match dispense {
        Some(service) => service.with(|d, _| d.retire_member(request.learner_id, request.incarnation)),
        None => {
            let (holder, retired) = members.lock().unwrap().retire_now(request.learner_id, request.incarnation);
            (holder, retired, 0)
        }
    };
    let status = match (holder, retired) {
        (None, _) => RETIRED_NO_MEMBER,
        (Some(_), true) => RETIRED_NOW,
        (Some(_), false) => RETIRED_ALREADY,
    };
    let incarnation = holder.map_or(0, |h: Holder| h.incarnation);
    info!(
        learner_id = request.learner_id,
        incarnation = format!("{incarnation:016x}"),
        status,
        reclaimed,
        reason = %request.reason,
        "member retired by the controller"
    );
    RetireAnswer { status, incarnation, reclaimed }
}

/// Serve one MEMBER_RETIRE connection whose first frame is `payload`.
pub async fn serve(
    wr: &mut OwnedWriteHalf,
    payload: &[u8],
    peer: Option<SocketAddr>,
    key: Option<&[u8]>,
    members: &SharedMembers,
    dispense: Option<Arc<crate::dispense_service::DispenseService>>,
) -> Result<()> {
    let request = match open_retire(key, peer, payload).and_then(decode_retire) {
        Ok(request) => request,
        Err(error) => {
            let message = format!("retire refused: {error:#}");
            let _ = send(wr, crate::protocol::MSG_ERROR, message.as_bytes()).await;
            bail!(message);
        }
    };
    let answer = apply(&request, members, dispense.as_deref());
    send(wr, MSG_MEMBER_RETIRED, &encode_answer(&answer)).await
}

async fn send(wr: &mut OwnedWriteHalf, msg_type: u8, payload: &[u8]) -> Result<()> {
    use tokio::io::AsyncWriteExt;
    let mut header = [0u8; 13];
    header[0..4].copy_from_slice(&crate::protocol::MAGIC.to_le_bytes());
    header[4] = msg_type;
    header[5..13].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    wr.write_all(&header).await?;
    wr.write_all(payload).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispense::{Claim, DispenseConfig};
    use std::time::{Duration, Instant};

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Golden bytes shared with `tests/test_member_retire.py` (Python codec).
    #[test]
    fn wire_golden_vectors() {
        let request = RetireRequest { learner_id: 0x8000_0007, incarnation: Some(0x0102_0304_0506_0708), reason: "scale_down".into() };
        let body = encode_retire(&request);
        assert_eq!(hex(&body), "0100070000800108070605040302010a007363616c655f646f776e");
        assert_eq!(decode_retire(&body).unwrap(), request);
        let mac = crate::hmac::hmac_sha256(b"k", &[MEMBER_RETIRE_MAC_DOMAIN, &body].concat());
        assert_eq!(
            hex(&mac),
            "e9c1742d535f0684c680c579c1091feee6d588c661aff106e512642b0acfddcf",
            "update the Python golden too if this changes"
        );
        let sealed = [body.as_slice(), &mac].concat();
        assert_eq!(open_retire(Some(b"k"), None, &sealed).unwrap(), body.as_slice());
        assert!(open_retire(Some(b"other"), None, &sealed).is_err());
        let current = RetireRequest { learner_id: 5, incarnation: None, reason: String::new() };
        assert_eq!(decode_retire(&encode_retire(&current)).unwrap(), current);
        assert_eq!(
            hex(&encode_answer(&RetireAnswer { status: RETIRED_NOW, incarnation: 9, reclaimed: 6 })),
            "0009000000000000000600000000000000"
        );
        let mut trailing = body.clone();
        trailing.push(0);
        assert!(decode_retire(&trailing).is_err());
        let mut bad_flag = body.clone();
        bad_flag[6] = 2;
        assert!(decode_retire(&bad_flag).is_err());
    }

    #[test]
    fn without_a_key_only_the_head_itself_may_retire() {
        let local: SocketAddr = "127.0.0.1:4000".parse().unwrap();
        let remote: SocketAddr = "192.0.2.3:4000".parse().unwrap();
        assert!(open_retire(None, Some(local), b"x").is_ok());
        assert!(open_retire(None, Some(remote), b"x").is_err());
        assert!(open_retire(None, None, b"x").is_err());
    }

    #[test]
    fn retire_reclaims_at_once_on_the_shared_table() {
        let members = crate::members::shared(Duration::from_secs(300));
        let service = crate::dispense_service::DispenseService::new(
            DispenseConfig { passes: Some(1), ..DispenseConfig::default() },
            None,
            1,
            None,
            None,
            members.clone(),
        )
        .unwrap();
        let a = Holder { learner_id: crate::members::SAMPLER_MEMBER_BIT | 1, incarnation: 11 };
        let b = Holder { learner_id: crate::members::SAMPLER_MEMBER_BIT | 2, incarnation: 22 };
        let now = Instant::now();
        service.with(|d, _| {
            d.open(a, 8, [1; 32], now).unwrap();
            d.open(b, 8, [1; 32], now).unwrap();
            assert_eq!(d.claim(a, 1, 3, now).unwrap(), Claim::Grant { block: 1, ranges: vec![(0, 3)] });
        });
        let request = RetireRequest { learner_id: a.learner_id, incarnation: None, reason: "scale_down".into() };
        assert_eq!(apply(&request, &members, Some(&service)), RetireAnswer { status: RETIRED_NOW, incarnation: 11, reclaimed: 3 });
        assert_eq!(apply(&request, &members, Some(&service)).status, RETIRED_NO_MEMBER);
        let again = RetireRequest { incarnation: Some(11), ..request.clone() };
        assert_eq!(apply(&again, &members, Some(&service)), RetireAnswer { status: RETIRED_ALREADY, incarnation: 11, reclaimed: 0 });
        assert!(members.lock().unwrap().is_retired(a));
        service.with(|d, _| {
            assert_eq!(d.claim(b, 1, 4, now).unwrap(), Claim::Grant { block: 2, ranges: vec![(0, 3), (3, 4)] });
            assert!(d.claim(a, 2, 1, now).is_err(), "the retired member is fenced");
            assert_eq!(d.totals.reclaimed, 3);
        });
        // without a dispenser the table alone is retired (relay / subscribers see it)
        let table = crate::members::shared(Duration::from_secs(300));
        table.lock().unwrap().join(b, now);
        let only = RetireRequest { learner_id: b.learner_id, incarnation: None, reason: "evicted".into() };
        assert_eq!(apply(&only, &table, None), RetireAnswer { status: RETIRED_NOW, incarnation: 22, reclaimed: 0 });
        assert!(table.lock().unwrap().is_retired(b));
    }
}
