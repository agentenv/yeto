//! Sample relay (S27 G-EA): a small store-and-forward queue on the head that
//! carries rollout samples from receive-only sampler islands (any cloud) to a
//! training island, so the samplers need no shared volume or object store.
//!
//! A relay connection is its own TCP connection on the syncer port (like
//! BW_PROBE): the first frame is RELAY_HELLO, sealed with an HMAC in the
//! relay domain. It never joins a learner session and never touches merge
//! state, so the relay is not part of the semantic profile hash.
//!
//! Channels: one channel per consuming training island (`channel` = its id).
//! Any number of producers write to a channel; one consumer reads it (a newer
//! consumer connection supersedes the older one).
//!
//! Delivery: a producer PUT carries a per-(producer, incarnation) sequence
//! number; a resend of the last sequence number gets the same answer (no
//! duplicate item). A full channel answers FULL and stores nothing; the
//! producer retries the same sequence later. The consumer TAKEs items after a
//! cursor and acknowledges items through an id; only acknowledged items are
//! deleted, so a consumer that reconnects (or a new consumer process) gets
//! every unacknowledged item again. Items live only in head memory: a head
//! restart loses them (the WELCOME carries a new relay epoch).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, ensure, Result};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::sync::Notify;
use tracing::{info, warn};

use crate::protocol::{read_frame_limited, MAGIC, MSG_ERROR};

pub const MSG_RELAY_HELLO: u8 = 60;
pub const MSG_RELAY_WELCOME: u8 = 61;
pub const MSG_RELAY_PUT: u8 = 62;
pub const MSG_RELAY_PUT_ACK: u8 = 63;
pub const MSG_RELAY_TAKE: u8 = 64;
pub const MSG_RELAY_ITEMS: u8 = 65;
pub const MSG_RELAY_STATUS: u8 = 66;

pub const RELAY_VERSION: u16 = 1;
pub const RELAY_MAC_DOMAIN: &[u8] = b"yeto-relay-hello-mac-v1\0";
pub const ROLE_PRODUCER: u8 = 1;
pub const ROLE_CONSUMER: u8 = 2;
/// A named value: the newest PUT wins and every client may read it (the
/// trainer's fragment layout for subscribers, small status records).
pub const ROLE_BOARD: u8 = 3;
/// Largest board value.
pub const MAX_BOARD_BYTES: u64 = 64 << 20;

pub const PUT_STORED: u8 = 0;
pub const PUT_DUPLICATE: u8 = 1;
pub const PUT_FULL: u8 = 2;

/// RELAY_HELLO body: version u16 | role u8 | channel u32 | client u32 | incarnation u64.
const HELLO_BODY_LEN: usize = 2 + 1 + 4 + 4 + 8;
/// RELAY_TAKE: after u64 | ack_through u64 | max_items u32 | max_bytes u64 | wait_ms u32.
const TAKE_LEN: usize = 8 + 8 + 4 + 8 + 4;
const MAX_WAIT: Duration = Duration::from_secs(60);
const MAX_CONNECTIONS: usize = 256;
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelayConfig {
    pub max_items: usize,
    pub max_bytes: u64,
    pub max_item_bytes: u64,
}

impl Default for RelayConfig {
    fn default() -> Self {
        RelayConfig { max_items: 4096, max_bytes: 2 << 30, max_item_bytes: 64 << 20 }
    }
}

impl RelayConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.max_items >= 1, "--sample-relay-max-items must be >= 1");
        ensure!(self.max_item_bytes >= 1, "--sample-relay-max-item-mib must be >= 1");
        ensure!(
            self.max_bytes >= self.max_item_bytes,
            "--sample-relay-max-mib must be at least --sample-relay-max-item-mib"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Hello {
    pub role: u8,
    pub channel: u32,
    pub client: u32,
    pub incarnation: u64,
}

pub fn parse_hello(body: &[u8]) -> Result<Hello> {
    ensure!(body.len() == HELLO_BODY_LEN, "RELAY_HELLO body has {} bytes, expected {HELLO_BODY_LEN}", body.len());
    let version = u16::from_le_bytes(body[0..2].try_into().unwrap());
    ensure!(version == RELAY_VERSION, "relay version {version}, this syncer speaks {RELAY_VERSION}");
    let role = body[2];
    ensure!(
        role == ROLE_PRODUCER || role == ROLE_CONSUMER || role == ROLE_BOARD,
        "unknown relay role {role}"
    );
    Ok(Hello {
        role,
        channel: u32::from_le_bytes(body[3..7].try_into().unwrap()),
        client: u32::from_le_bytes(body[7..11].try_into().unwrap()),
        incarnation: u64::from_le_bytes(body[11..19].try_into().unwrap()),
    })
}

/// With a key: verify and strip the trailing HMAC-SHA256(key, domain | body).
pub fn open_relay_hello<'a>(key: Option<&[u8]>, payload: &'a [u8]) -> Result<&'a [u8]> {
    let Some(key) = key else { return Ok(payload) };
    ensure!(payload.len() >= crate::hmac::HMAC_LEN, "RELAY_HELLO shorter than its HMAC");
    let (body, mac) = payload.split_at(payload.len() - crate::hmac::HMAC_LEN);
    let mut framed = Vec::with_capacity(RELAY_MAC_DOMAIN.len() + body.len());
    framed.extend_from_slice(RELAY_MAC_DOMAIN);
    framed.extend_from_slice(body);
    let expected = crate::hmac::hmac_sha256(key, &framed);
    let diff = expected.iter().zip(mac).fold(0u8, |acc, (x, y)| acc | (x ^ y));
    ensure!(diff == 0, "RELAY_HELLO HMAC mismatch (wrong or missing island HMAC key)");
    Ok(body)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Depth {
    pub items: u32,
    pub bytes: u64,
    pub acked_through: u64,
    pub next_id: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PutReply {
    pub seq: u64,
    pub status: u8,
    pub item_id: u64,
    pub depth: Depth,
}

#[derive(Clone, Debug)]
pub struct Item {
    pub id: u64,
    pub producer: u32,
    pub payload: Arc<Vec<u8>>,
}

#[derive(Default)]
struct Counters {
    stored: u64,
    stored_bytes: u64,
    duplicate: u64,
    full: u64,
    acked: u64,
}

struct Channel {
    next_id: u64,
    items: VecDeque<Item>,
    bytes: u64,
    acked_through: u64,
    /// (client, incarnation) of the current consumer connection.
    consumer: Option<(u32, u64, u64)>,
    /// Last accepted (seq, item id) per producer process.
    producers: HashMap<(u32, u64), (u64, u64)>,
    counters: Counters,
}

impl Channel {
    fn new() -> Self {
        Channel {
            next_id: 1,
            items: VecDeque::new(),
            bytes: 0,
            acked_through: 0,
            consumer: None,
            producers: HashMap::new(),
            counters: Counters::default(),
        }
    }

    fn depth(&self) -> Depth {
        Depth {
            items: self.items.len() as u32,
            bytes: self.bytes,
            acked_through: self.acked_through,
            next_id: self.next_id,
        }
    }
}

/// The pure relay state. Every operation is synchronous; the async layer
/// holds it behind a mutex and wakes waiting consumers after a PUT.
pub struct Relay {
    pub epoch: u64,
    cfg: RelayConfig,
    channels: HashMap<u32, Channel>,
    boards: HashMap<u32, Item>,
    next_board_id: u64,
    next_connection: u64,
}

impl Relay {
    pub fn new(cfg: RelayConfig, epoch: u64) -> Self {
        Relay {
            epoch,
            cfg,
            channels: HashMap::new(),
            boards: HashMap::new(),
            next_board_id: 1,
            next_connection: 1,
        }
    }

    /// Replace board `board` with `payload`; returns its new id.
    pub fn set_board(&mut self, board: u32, client: u32, payload: Vec<u8>) -> Result<u64> {
        ensure!(
            payload.len() as u64 <= MAX_BOARD_BYTES,
            "board value has {} bytes, limit {MAX_BOARD_BYTES}",
            payload.len()
        );
        let id = self.next_board_id;
        self.next_board_id += 1;
        self.boards.insert(board, Item { id, producer: client, payload: Arc::new(payload) });
        Ok(id)
    }

    /// The board value when its id is newer than `after`.
    pub fn board(&self, board: u32, after: u64) -> Option<Item> {
        self.boards.get(&board).filter(|item| item.id > after).cloned()
    }

    pub fn config(&self) -> RelayConfig {
        self.cfg
    }

    fn channel(&mut self, channel: u32) -> &mut Channel {
        self.channels.entry(channel).or_insert_with(Channel::new)
    }

    pub fn depth(&mut self, channel: u32) -> Depth {
        self.channel(channel).depth()
    }

    /// Register a consumer connection; returns its connection token. A newer
    /// connection supersedes the older one, whose next TAKE is refused.
    pub fn attach_consumer(&mut self, channel: u32, client: u32, incarnation: u64) -> u64 {
        let token = self.next_connection;
        self.next_connection += 1;
        let ch = self.channel(channel);
        if let Some((old_client, old_inc, _)) = ch.consumer {
            info!(
                channel,
                old_client, old_inc, client, incarnation, "sample relay consumer superseded"
            );
        }
        ch.consumer = Some((client, incarnation, token));
        token
    }

    pub fn put(&mut self, channel: u32, producer: u32, incarnation: u64, seq: u64, payload: Vec<u8>) -> Result<PutReply> {
        let max_item = self.cfg.max_item_bytes;
        let max_items = self.cfg.max_items;
        let max_bytes = self.cfg.max_bytes;
        ensure!(seq >= 1, "relay PUT sequence numbers start at 1");
        ensure!(
            payload.len() as u64 <= max_item,
            "relay item has {} bytes, limit {max_item}",
            payload.len()
        );
        let ch = self.channel(channel);
        if let Some(&(last_seq, last_id)) = ch.producers.get(&(producer, incarnation)) {
            if seq <= last_seq {
                ch.counters.duplicate += 1;
                let item_id = if seq == last_seq { last_id } else { 0 };
                return Ok(PutReply { seq, status: PUT_DUPLICATE, item_id, depth: ch.depth() });
            }
        }
        let len = payload.len() as u64;
        if ch.items.len() >= max_items || ch.bytes + len > max_bytes {
            ch.counters.full += 1;
            return Ok(PutReply { seq, status: PUT_FULL, item_id: 0, depth: ch.depth() });
        }
        let id = ch.next_id;
        ch.next_id += 1;
        ch.items.push_back(Item { id, producer, payload: Arc::new(payload) });
        ch.bytes += len;
        ch.producers.insert((producer, incarnation), (seq, id));
        ch.counters.stored += 1;
        ch.counters.stored_bytes += len;
        Ok(PutReply { seq, status: PUT_STORED, item_id: id, depth: ch.depth() })
    }

    /// Acknowledge items through `ack_through` (they are deleted), then return
    /// up to `max_items` items with id > `after` (at least one item if any,
    /// further items while their total stays within `max_bytes`).
    pub fn take(
        &mut self,
        channel: u32,
        token: u64,
        after: u64,
        ack_through: u64,
        max_items: u32,
        max_bytes: u64,
    ) -> Result<(Vec<Item>, Depth)> {
        let ch = self.channel(channel);
        match ch.consumer {
            Some((_, _, current)) if current == token => {}
            _ => bail!("relay superseded: a newer consumer owns channel {channel}"),
        }
        ensure!(
            ack_through < ch.next_id && after < ch.next_id,
            "relay TAKE cursor beyond the last item (after {after}, ack {ack_through}, next {})",
            ch.next_id
        );
        while let Some(front) = ch.items.front() {
            if front.id > ack_through {
                break;
            }
            ch.bytes -= front.payload.len() as u64;
            ch.items.pop_front();
            ch.counters.acked += 1;
        }
        ch.acked_through = ch.acked_through.max(ack_through);
        let mut out = Vec::new();
        let mut total = 0u64;
        for item in ch.items.iter().filter(|item| item.id > after) {
            if out.len() as u32 >= max_items {
                break;
            }
            let len = item.payload.len() as u64;
            if !out.is_empty() && total + len > max_bytes {
                break;
            }
            total += len;
            out.push(item.clone());
        }
        Ok((out, ch.depth()))
    }

    pub fn summary(&self) -> Vec<(u32, Depth, u64, u64, u64, u64, u64)> {
        let mut rows: Vec<_> = self
            .channels
            .iter()
            .map(|(id, ch)| {
                let c = &ch.counters;
                (*id, ch.depth(), c.stored, c.stored_bytes, c.duplicate, c.full, c.acked)
            })
            .collect();
        rows.sort_by_key(|row| row.0);
        rows
    }
}

/// Shared relay of one syncer process (None = relay off).
pub struct RelayHub {
    pub state: Mutex<Relay>,
    pub notify: Notify,
    pub key: Option<Vec<u8>>,
}

impl RelayHub {
    pub fn new(cfg: RelayConfig, key: Option<Vec<u8>>) -> Arc<Self> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let epoch = (nanos as u64) ^ ((std::process::id() as u64) << 32) | 1;
        Arc::new(RelayHub { state: Mutex::new(Relay::new(cfg, epoch)), notify: Notify::new(), key })
    }

    /// Spawn a task that logs the per-channel counters every `every` when they change.
    pub fn spawn_reporter(self: &Arc<Self>, every: Duration) {
        let hub = Arc::clone(self);
        tokio::spawn(async move {
            let mut last = Vec::new();
            loop {
                tokio::time::sleep(every).await;
                let rows = hub.state.lock().unwrap().summary();
                if rows != last {
                    for (channel, depth, stored, stored_bytes, duplicate, full, acked) in &rows {
                        info!(
                            channel,
                            depth_items = depth.items,
                            depth_bytes = depth.bytes,
                            stored,
                            stored_bytes,
                            duplicate,
                            full,
                            acked,
                            "sample relay"
                        );
                    }
                    last = rows;
                }
            }
        });
    }
}

struct Slot;

impl Slot {
    fn acquire() -> Option<Self> {
        if ACTIVE.fetch_add(1, Ordering::AcqRel) >= MAX_CONNECTIONS {
            ACTIVE.fetch_sub(1, Ordering::AcqRel);
            return None;
        }
        Some(Slot)
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        ACTIVE.fetch_sub(1, Ordering::AcqRel);
    }
}

async fn write_frame<W: AsyncWrite + Unpin>(wr: &mut W, msg_type: u8, parts: &[&[u8]]) -> Result<()> {
    let len: usize = parts.iter().map(|p| p.len()).sum();
    let mut header = [0u8; 13];
    header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    header[4] = msg_type;
    header[5..13].copy_from_slice(&(len as u64).to_le_bytes());
    wr.write_all(&header).await?;
    for part in parts {
        wr.write_all(part).await?;
    }
    wr.flush().await?;
    Ok(())
}

fn encode_depth(depth: &Depth, out: &mut Vec<u8>) {
    out.extend_from_slice(&depth.items.to_le_bytes());
    out.extend_from_slice(&depth.bytes.to_le_bytes());
    out.extend_from_slice(&depth.acked_through.to_le_bytes());
    out.extend_from_slice(&depth.next_id.to_le_bytes());
}

/// Serve one relay connection whose RELAY_HELLO payload is `first_payload`.
/// `hub` None: the relay is off and the connection is refused.
pub async fn serve<R, W>(rd: &mut R, wr: &mut W, first_payload: &[u8], hub: Option<Arc<RelayHub>>) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let Some(hub) = hub else {
        let message = "relay refused: this syncer runs without --sample-relay";
        let _ = write_frame(wr, MSG_ERROR, &[message.as_bytes()]).await;
        bail!(message);
    };
    let hello = match open_relay_hello(hub.key.as_deref(), first_payload).and_then(parse_hello) {
        Ok(hello) => hello,
        Err(error) => {
            let message = format!("relay refused: invalid RELAY_HELLO: {error:#}");
            let _ = write_frame(wr, MSG_ERROR, &[message.as_bytes()]).await;
            return Err(error);
        }
    };
    let Some(_slot) = Slot::acquire() else {
        let message = "relay refused: too many relay connections";
        let _ = write_frame(wr, MSG_ERROR, &[message.as_bytes()]).await;
        bail!(message);
    };
    let (epoch, cfg, depth, token) = {
        let mut relay = hub.state.lock().unwrap();
        let token = if hello.role == ROLE_CONSUMER {
            relay.attach_consumer(hello.channel, hello.client, hello.incarnation)
        } else {
            0
        };
        (relay.epoch, relay.config(), relay.depth(hello.channel), token)
    };
    info!(
        role = hello.role,
        channel = hello.channel,
        client = hello.client,
        incarnation = hello.incarnation,
        "sample relay connection"
    );
    let mut welcome = Vec::with_capacity(64);
    welcome.extend_from_slice(&RELAY_VERSION.to_le_bytes());
    welcome.extend_from_slice(&epoch.to_le_bytes());
    welcome.extend_from_slice(&cfg.max_item_bytes.to_le_bytes());
    welcome.extend_from_slice(&(cfg.max_items as u32).to_le_bytes());
    welcome.extend_from_slice(&cfg.max_bytes.to_le_bytes());
    encode_depth(&depth, &mut welcome);
    write_frame(wr, MSG_RELAY_WELCOME, &[&welcome]).await?;
    let put_limit = 8 + cfg.max_item_bytes;
    let role = hello.role;
    loop {
        let frame = match read_frame_limited(rd, |msg_type| match (role, msg_type) {
            (ROLE_PRODUCER, MSG_RELAY_PUT) => Ok(put_limit),
            (ROLE_BOARD, MSG_RELAY_PUT) => Ok(8 + MAX_BOARD_BYTES),
            (ROLE_CONSUMER | ROLE_BOARD, MSG_RELAY_TAKE) => Ok(TAKE_LEN as u64),
            (_, MSG_RELAY_STATUS) => Ok(0),
            (_, other) => bail!("relay role {role} cannot send message type {other}"),
        })
        .await
        {
            Ok(frame) => frame,
            Err(error) => {
                if error.downcast_ref::<std::io::Error>().is_some() {
                    // Peer closed or the link broke: a normal end for a relay client.
                    return Ok(());
                }
                let message = format!("relay refused: {error:#}");
                let _ = write_frame(wr, MSG_ERROR, &[message.as_bytes()]).await;
                return Err(error);
            }
        };
        match frame.msg_type {
            MSG_RELAY_STATUS => {
                let depth = hub.state.lock().unwrap().depth(hello.channel);
                let mut out = Vec::with_capacity(28);
                encode_depth(&depth, &mut out);
                write_frame(wr, MSG_RELAY_STATUS, &[&out]).await?;
            }
            MSG_RELAY_PUT if role == ROLE_BOARD => {
                ensure!(frame.payload.len() >= 8, "relay PUT shorter than its sequence number");
                let seq = u64::from_le_bytes(frame.payload[0..8].try_into().unwrap());
                let mut payload = frame.payload;
                payload.drain(0..8);
                let id = hub.state.lock().unwrap().set_board(hello.channel, hello.client, payload);
                let id = match id {
                    Ok(id) => id,
                    Err(error) => {
                        let message = format!("relay refused: {error:#}");
                        let _ = write_frame(wr, MSG_ERROR, &[message.as_bytes()]).await;
                        return Err(error);
                    }
                };
                info!(board = hello.channel, client = hello.client, id, "sample relay board set");
                hub.notify.notify_waiters();
                let mut out = Vec::with_capacity(48);
                out.extend_from_slice(&seq.to_le_bytes());
                out.push(PUT_STORED);
                out.extend_from_slice(&id.to_le_bytes());
                encode_depth(&Depth::default(), &mut out);
                write_frame(wr, MSG_RELAY_PUT_ACK, &[&out]).await?;
            }
            MSG_RELAY_PUT => {
                ensure!(frame.payload.len() >= 8, "relay PUT shorter than its sequence number");
                let seq = u64::from_le_bytes(frame.payload[0..8].try_into().unwrap());
                let mut payload = frame.payload;
                payload.drain(0..8);
                let reply = hub.state.lock().unwrap().put(hello.channel, hello.client, hello.incarnation, seq, payload);
                let reply = match reply {
                    Ok(reply) => reply,
                    Err(error) => {
                        let message = format!("relay refused: {error:#}");
                        let _ = write_frame(wr, MSG_ERROR, &[message.as_bytes()]).await;
                        return Err(error);
                    }
                };
                if reply.status == PUT_STORED {
                    hub.notify.notify_waiters();
                }
                let mut out = Vec::with_capacity(48);
                out.extend_from_slice(&reply.seq.to_le_bytes());
                out.push(reply.status);
                out.extend_from_slice(&reply.item_id.to_le_bytes());
                encode_depth(&reply.depth, &mut out);
                write_frame(wr, MSG_RELAY_PUT_ACK, &[&out]).await?;
            }
            MSG_RELAY_TAKE => {
                let p = &frame.payload;
                ensure!(p.len() == TAKE_LEN, "relay TAKE has {} bytes, expected {TAKE_LEN}", p.len());
                let after = u64::from_le_bytes(p[0..8].try_into().unwrap());
                let ack = u64::from_le_bytes(p[8..16].try_into().unwrap());
                let max_items = u32::from_le_bytes(p[16..20].try_into().unwrap()).max(1);
                let max_bytes = u64::from_le_bytes(p[20..28].try_into().unwrap());
                let wait = Duration::from_millis(u32::from_le_bytes(p[28..32].try_into().unwrap()) as u64).min(MAX_WAIT);
                let deadline = Instant::now() + wait;
                let (items, depth) = loop {
                    let notified = hub.notify.notified();
                    let result = if role == ROLE_BOARD {
                        let found = hub.state.lock().unwrap().board(hello.channel, after);
                        Ok((found.into_iter().collect(), Depth::default()))
                    } else {
                        hub.state.lock().unwrap().take(hello.channel, token, after, ack, max_items, max_bytes)
                    };
                    let (items, depth) = match result {
                        Ok(found) => found,
                        Err(error) => {
                            let message = format!("{error:#}");
                            warn!(channel = hello.channel, "sample relay TAKE refused: {message}");
                            let _ = write_frame(wr, MSG_ERROR, &[message.as_bytes()]).await;
                            return Err(error);
                        }
                    };
                    let now = Instant::now();
                    if !items.is_empty() || now >= deadline {
                        break (items, depth);
                    }
                    let _ = tokio::time::timeout(deadline - now, notified).await;
                };
                let mut head = Vec::with_capacity(4 + 28);
                head.extend_from_slice(&(items.len() as u32).to_le_bytes());
                encode_depth(&depth, &mut head);
                let mut metas: Vec<[u8; 20]> = Vec::with_capacity(items.len());
                for item in &items {
                    let mut meta = [0u8; 20];
                    meta[0..8].copy_from_slice(&item.id.to_le_bytes());
                    meta[8..12].copy_from_slice(&item.producer.to_le_bytes());
                    meta[12..20].copy_from_slice(&(item.payload.len() as u64).to_le_bytes());
                    metas.push(meta);
                }
                let mut parts: Vec<&[u8]> = Vec::with_capacity(1 + 2 * items.len());
                parts.push(&head);
                for (meta, item) in metas.iter().zip(&items) {
                    parts.push(meta);
                    parts.push(item.payload.as_slice());
                }
                write_frame(wr, MSG_RELAY_ITEMS, &parts).await?;
            }
            other => bail!("unexpected relay message type {other}"),
        }
    }
}

/// Read the remaining bytes of a frame type (used by tests only).
#[cfg(test)]
async fn read_reply<R: AsyncRead + Unpin>(rd: &mut R) -> Result<crate::protocol::Frame> {
    read_frame_limited(rd, |_| Ok(1 << 30)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(items: usize, bytes: u64, item: u64) -> RelayConfig {
        RelayConfig { max_items: items, max_bytes: bytes, max_item_bytes: item }
    }

    #[test]
    fn put_dedupes_resends_and_refuses_when_full() {
        let mut relay = Relay::new(cfg(2, 100, 50), 7);
        let a = relay.put(0, 1, 11, 1, vec![1; 10]).unwrap();
        assert_eq!((a.status, a.item_id), (PUT_STORED, 1));
        // Resend of the same sequence: same item, nothing stored.
        let again = relay.put(0, 1, 11, 1, vec![1; 10]).unwrap();
        assert_eq!((again.status, again.item_id, again.depth.items), (PUT_DUPLICATE, 1, 1));
        // A new incarnation of the same producer starts its own sequence.
        let b = relay.put(0, 1, 12, 1, vec![2; 10]).unwrap();
        assert_eq!((b.status, b.item_id), (PUT_STORED, 2));
        let full = relay.put(0, 1, 12, 2, vec![3; 10]).unwrap();
        assert_eq!((full.status, full.depth.items), (PUT_FULL, 2));
        // Too large for the item limit is a protocol error.
        assert!(relay.put(0, 1, 12, 3, vec![0; 51]).is_err());
        // Channels are independent.
        assert_eq!(relay.put(1, 1, 12, 9, vec![0; 50]).unwrap().status, PUT_STORED);
    }

    #[test]
    fn take_returns_after_cursor_and_deletes_only_acked() {
        let mut relay = Relay::new(cfg(16, 1000, 100), 7);
        for seq in 1..=4 {
            relay.put(0, 3, 1, seq, vec![seq as u8; 10]).unwrap();
        }
        let token = relay.attach_consumer(0, 0, 99);
        let (items, depth) = relay.take(0, token, 0, 0, 2, 1000).unwrap();
        assert_eq!(items.iter().map(|i| i.id).collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(depth.items, 4);
        // Re-delivery after a reconnect: same cursor gives the same items.
        let token2 = relay.attach_consumer(0, 0, 99);
        assert!(relay.take(0, token, 0, 0, 2, 1000).is_err(), "old connection is superseded");
        let (items, _) = relay.take(0, token2, 0, 0, 2, 1000).unwrap();
        assert_eq!(items.iter().map(|i| i.id).collect::<Vec<_>>(), vec![1, 2]);
        // Ack 1..2, read after 2; byte cap keeps at least one item.
        let (items, depth) = relay.take(0, token2, 2, 2, 8, 5).unwrap();
        assert_eq!(items.iter().map(|i| i.id).collect::<Vec<_>>(), vec![3]);
        assert_eq!((depth.items, depth.bytes, depth.acked_through), (2, 20, 2));
        assert!(relay.take(0, token2, 9, 2, 8, 5).is_err(), "cursor beyond the last item");
        // Space freed by the ack is reusable.
        let (_, depth) = relay.take(0, token2, 4, 4, 8, 1000).unwrap();
        assert_eq!(depth.items, 0);
    }

    #[test]
    fn board_keeps_the_newest_value() {
        let mut relay = Relay::new(cfg(4, 100, 10), 7);
        assert!(relay.board(3, 0).is_none());
        let first = relay.set_board(3, 1, b"layout-v1".to_vec()).unwrap();
        let second = relay.set_board(3, 1, b"layout-v2".to_vec()).unwrap();
        assert!(second > first);
        let item = relay.board(3, 0).unwrap();
        assert_eq!((item.id, item.payload.as_slice()), (second, &b"layout-v2"[..]));
        assert!(relay.board(3, second).is_none(), "nothing newer than the newest");
        assert!(relay.board(4, 0).is_none(), "boards are independent");
        // A board value may exceed the channel item limit (10 bytes here).
        assert!(relay.set_board(5, 1, vec![0; 1000]).is_ok());
    }

    #[test]
    fn hello_parse_and_mac() {
        let mut body = Vec::new();
        body.extend_from_slice(&RELAY_VERSION.to_le_bytes());
        body.push(ROLE_CONSUMER);
        body.extend_from_slice(&5u32.to_le_bytes());
        body.extend_from_slice(&7u32.to_le_bytes());
        body.extend_from_slice(&9u64.to_le_bytes());
        let hello = parse_hello(&body).unwrap();
        assert_eq!(hello, Hello { role: ROLE_CONSUMER, channel: 5, client: 7, incarnation: 9 });
        let mut framed = RELAY_MAC_DOMAIN.to_vec();
        framed.extend_from_slice(&body);
        let mut sealed = body.clone();
        sealed.extend_from_slice(&crate::hmac::hmac_sha256(b"k", &framed));
        assert_eq!(open_relay_hello(Some(b"k"), &sealed).unwrap(), &body[..]);
        assert!(open_relay_hello(Some(b"other"), &sealed).is_err());
        // The learner HELLO domain does not open a relay HELLO.
        let mut learner = b"yeto-hello-mac-v1\0".to_vec();
        learner.extend_from_slice(&body);
        let mut wrong = body.clone();
        wrong.extend_from_slice(&crate::hmac::hmac_sha256(b"k", &learner));
        assert!(open_relay_hello(Some(b"k"), &wrong).is_err());
        let mut bad = body.clone();
        bad[2] = 9;
        assert!(parse_hello(&bad).is_err());
    }

    fn hello_frame(role: u8, channel: u32, client: u32, inc: u64) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&RELAY_VERSION.to_le_bytes());
        body.push(role);
        body.extend_from_slice(&channel.to_le_bytes());
        body.extend_from_slice(&client.to_le_bytes());
        body.extend_from_slice(&inc.to_le_bytes());
        body
    }

    #[tokio::test]
    async fn serve_round_trip_over_duplex() {
        let hub = RelayHub::new(cfg(8, 1 << 20, 1 << 16), None);
        // Producer.
        let (client, server) = tokio::io::duplex(1 << 20);
        let (mut srd, mut swr) = tokio::io::split(server);
        let h = Arc::clone(&hub);
        let hello = hello_frame(ROLE_PRODUCER, 0, 4, 1);
        let task = tokio::spawn(async move { serve(&mut srd, &mut swr, &hello, Some(h)).await });
        let (mut crd, mut cwr) = tokio::io::split(client);
        let welcome = read_reply(&mut crd).await.unwrap();
        assert_eq!(welcome.msg_type, MSG_RELAY_WELCOME);
        let mut put = 1u64.to_le_bytes().to_vec();
        put.extend_from_slice(b"samples-1");
        write_frame(&mut cwr, MSG_RELAY_PUT, &[&put]).await.unwrap();
        let ack = read_reply(&mut crd).await.unwrap();
        assert_eq!(ack.msg_type, MSG_RELAY_PUT_ACK);
        assert_eq!(ack.payload[8], PUT_STORED);
        // A producer may not TAKE.
        write_frame(&mut cwr, MSG_RELAY_TAKE, &[&[0u8; TAKE_LEN]]).await.unwrap();
        let err = read_reply(&mut crd).await.unwrap();
        assert_eq!(err.msg_type, MSG_ERROR);
        assert!(task.await.unwrap().is_err());
        // Consumer.
        let (client, server) = tokio::io::duplex(1 << 20);
        let (mut srd, mut swr) = tokio::io::split(server);
        let h = Arc::clone(&hub);
        let hello = hello_frame(ROLE_CONSUMER, 0, 0, 2);
        tokio::spawn(async move { serve(&mut srd, &mut swr, &hello, Some(h)).await });
        let (mut crd, mut cwr) = tokio::io::split(client);
        read_reply(&mut crd).await.unwrap();
        let mut take = Vec::new();
        take.extend_from_slice(&0u64.to_le_bytes());
        take.extend_from_slice(&0u64.to_le_bytes());
        take.extend_from_slice(&16u32.to_le_bytes());
        take.extend_from_slice(&(1u64 << 20).to_le_bytes());
        take.extend_from_slice(&100u32.to_le_bytes());
        write_frame(&mut cwr, MSG_RELAY_TAKE, &[&take]).await.unwrap();
        let items = read_reply(&mut crd).await.unwrap();
        assert_eq!(items.msg_type, MSG_RELAY_ITEMS);
        assert_eq!(u32::from_le_bytes(items.payload[0..4].try_into().unwrap()), 1);
        assert_eq!(&items.payload[4 + 28 + 20..], b"samples-1");
        // Relay off: refused with an ERROR.
        let (client, server) = tokio::io::duplex(1 << 16);
        let (mut srd, mut swr) = tokio::io::split(server);
        let hello = hello_frame(ROLE_CONSUMER, 0, 0, 2);
        let task = tokio::spawn(async move { serve(&mut srd, &mut swr, &hello, None).await });
        let (mut crd, _cwr) = tokio::io::split(client);
        let err = read_reply(&mut crd).await.unwrap();
        assert_eq!(err.msg_type, MSG_ERROR);
        assert!(task.await.unwrap().is_err());
    }
}
