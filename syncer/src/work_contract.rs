//! WorkContract conformance of the syncer dispenser.
//!
//! The contract (claim / done / drop / fail / release / beat / fence /
//! retire / data end) is stated in `yeto/rl/engine/work_contract.py`; its
//! reference implementation is the store ledger
//! (`yeto/rl/engine/store_dispense.py`). Both run the cases of
//! `tests/fixtures/work_contract_cases.json`: the Python driver is
//! `tests/test_work_contract.py`, this module is the Rust one, on
//! [`Dispenser`] with every request preceded by an expiry pass (as
//! `dispense_service.rs` does). `fence` and `retire` both map to
//! MEMBER_RETIRE ([`Dispenser::retire_member`]): the syncer has no separate
//! incarnation lease. `drain` / `drained` are MEMBER_RETIRE in drain mode
//! ([`Dispenser::set_drain`]; drained when it returns 0 held).

#![cfg(test)]

use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

use crate::dispense::{Claim, DispenseConfig, Dispenser, Holder, OpenOutcome, ReportEntry, ReportKind};

const CASES: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/fixtures/work_contract_cases.json"));

// ------------------------------------------------------------ a small JSON reader

#[derive(Clone, Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn at(&self, key: &str) -> &Json {
        self.get(key).unwrap_or_else(|| panic!("missing {key:?} in {self:?}"))
    }

    fn str(&self) -> &str {
        match self {
            Json::Str(s) => s,
            other => panic!("not a string: {other:?}"),
        }
    }

    fn num(&self) -> f64 {
        match self {
            Json::Num(n) => *n,
            other => panic!("not a number: {other:?}"),
        }
    }

    fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(items) => items,
            other => panic!("not an array: {other:?}"),
        }
    }
}

struct Parser<'a> {
    text: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn parse(text: &str) -> Json {
        let mut p = Parser { text: text.as_bytes(), at: 0 };
        let value = p.value();
        p.ws();
        assert_eq!(p.at, p.text.len(), "trailing JSON");
        value
    }

    fn ws(&mut self) {
        while self.at < self.text.len() && self.text[self.at].is_ascii_whitespace() {
            self.at += 1;
        }
    }

    fn eat(&mut self, byte: u8) {
        self.ws();
        assert_eq!(self.text[self.at], byte, "JSON: expected {:?} at {}", byte as char, self.at);
        self.at += 1;
    }

    fn peek(&mut self) -> u8 {
        self.ws();
        self.text[self.at]
    }

    fn value(&mut self) -> Json {
        match self.peek() {
            b'{' => {
                self.eat(b'{');
                let mut fields = Vec::new();
                if self.peek() == b'}' {
                    self.at += 1;
                    return Json::Obj(fields);
                }
                loop {
                    let key = self.string();
                    self.eat(b':');
                    fields.push((key, self.value()));
                    match self.peek() {
                        b',' => self.at += 1,
                        _ => {
                            self.eat(b'}');
                            return Json::Obj(fields);
                        }
                    }
                }
            }
            b'[' => {
                self.eat(b'[');
                let mut items = Vec::new();
                if self.peek() == b']' {
                    self.at += 1;
                    return Json::Arr(items);
                }
                loop {
                    items.push(self.value());
                    match self.peek() {
                        b',' => self.at += 1,
                        _ => {
                            self.eat(b']');
                            return Json::Arr(items);
                        }
                    }
                }
            }
            b'"' => Json::Str(self.string()),
            b't' => self.word("true", Json::Bool(true)),
            b'f' => self.word("false", Json::Bool(false)),
            b'n' => self.word("null", Json::Null),
            _ => {
                let start = self.at;
                while self.at < self.text.len() && b"+-0123456789.eE".contains(&self.text[self.at]) {
                    self.at += 1;
                }
                let text = std::str::from_utf8(&self.text[start..self.at]).unwrap();
                Json::Num(text.parse().unwrap_or_else(|_| panic!("JSON: bad number {text:?}")))
            }
        }
    }

    fn word(&mut self, word: &str, value: Json) -> Json {
        assert!(self.text[self.at..].starts_with(word.as_bytes()), "JSON: expected {word}");
        self.at += word.len();
        value
    }

    fn string(&mut self) -> String {
        self.eat(b'"');
        let mut out = Vec::new();
        loop {
            let byte = self.text[self.at];
            self.at += 1;
            match byte {
                b'"' => return String::from_utf8(out).unwrap(),
                b'\\' => {
                    let escaped = self.text[self.at];
                    self.at += 1;
                    out.push(match escaped {
                        b'n' => b'\n',
                        b't' => b'\t',
                        other => other, // \" \\ \/ (the case table needs no \u)
                    });
                }
                other => out.push(other),
            }
        }
    }
}

// ------------------------------------------------------------ the driver

#[derive(Debug, PartialEq)]
enum ClaimAnswer {
    Items(Vec<u64>),
    Wait,
    End,
    Fenced,
}

/// The WorkContract over one [`Dispenser`] (module doc).
struct SyncerContract {
    d: Dispenser,
    t0: Instant,
    elapsed: Duration,
    rows: u64,
    holders: HashMap<String, u32>,
    seq: HashMap<Holder, u64>,
    /// position -> block, per member (what it was granted and has not reported)
    held: HashMap<Holder, BTreeMap<u64, u64>>,
}

impl SyncerContract {
    fn new(config: &Json) -> Self {
        let cfg = DispenseConfig {
            lease: Duration::from_secs_f64(config.at("lease_s").num()),
            passes: Some(config.at("passes").num() as u64),
            // no max_redo in the case: the dispenser's default (the same as the store ledger's)
            max_redo: config.get("max_redo").map_or(crate::dispense::DEFAULT_MAX_REDO, |v| v.num() as u32),
            max_block: 1 << 16,
        };
        Self {
            d: Dispenser::new(cfg),
            t0: Instant::now(),
            elapsed: Duration::ZERO,
            rows: config.at("rows").num() as u64,
            holders: HashMap::new(),
            seq: HashMap::new(),
            held: HashMap::new(),
        }
    }

    fn now(&self) -> Instant {
        self.t0 + self.elapsed
    }

    /// "A#1" -> a sampler member id and incarnation 1.
    fn member(&mut self, text: &str) -> Holder {
        let (name, inc) = text.split_once('#').expect("holder#incarnation");
        let next = self.holders.len() as u32;
        let id = *self.holders.entry(name.to_string()).or_insert(crate::members::SAMPLER_MEMBER_BIT | (next + 1));
        Holder { learner_id: id, incarnation: inc.parse().expect("numeric incarnation") }
    }

    /// Every request of the service first expires the silent holders.
    fn tick(&mut self) -> Instant {
        let now = self.now();
        self.d.expire(now);
        now
    }

    fn join(&mut self, m: Holder) -> u64 {
        let now = self.tick();
        match self.d.open(m, self.rows, [7; 32], now).expect("join") {
            OpenOutcome::Joined { reclaimed, .. } => reclaimed,
            OpenOutcome::Reconnected => 0,
        }
    }

    fn next_seq(&mut self, m: Holder) -> u64 {
        let seq = self.seq.entry(m).or_insert(0);
        *seq += 1;
        *seq
    }

    fn claim(&mut self, m: Holder, n: u32) -> ClaimAnswer {
        let now = self.tick();
        let seq = self.next_seq(m);
        match self.d.claim(m, seq, n, now) {
            Ok(Claim::Grant { block, ranges }) => {
                let mut items = Vec::new();
                for (start, end) in ranges {
                    for p in start..end {
                        self.held.entry(m).or_default().insert(p, block);
                        items.push(p);
                    }
                }
                ClaimAnswer::Items(items)
            }
            Ok(Claim::Wait) => ClaimAnswer::Wait,
            Ok(Claim::End) => ClaimAnswer::End,
            Err(crate::dispense::DispenseError::Fenced(_)) => ClaimAnswer::Fenced,
            Err(other) => panic!("claim refused: {other}"),
        }
    }

    /// Report `items` of `m` as `kind`; false ("not held") when the member
    /// does not hold every one of them or the dispenser refuses the report.
    fn report(&mut self, m: Holder, items: &[u64], kind: ReportKind) -> bool {
        let now = self.tick();
        let held = self.held.entry(m).or_default();
        if !items.iter().all(|p| held.contains_key(p)) {
            return false;
        }
        let mut by_block: BTreeMap<u64, Vec<(u64, u64)>> = BTreeMap::new();
        for &p in items {
            by_block.entry(held[&p]).or_default().push((p, p + 1));
        }
        let entries: Vec<ReportEntry> =
            by_block.into_iter().map(|(block, ranges)| ReportEntry { block, kind, ranges }).collect();
        let seq = self.next_seq(m);
        let ok = self.d.report(m, seq, &entries, now).is_ok();
        let held = self.held.entry(m).or_default();
        for p in items {
            held.remove(p);
        }
        ok
    }

    fn beat(&mut self, m: Holder) {
        let now = self.tick();
        let _ = self.d.heartbeat(m, now); // a retired member's beat is refused: nothing to check here
    }

    /// TASK_HOLDS, as the island's task source asks it (a refusal: not held).
    fn holds(&mut self, m: Holder, item: u64) -> bool {
        let now = self.tick();
        self.d.holds_request(m, &[item], now).is_ok_and(|held| held[0])
    }

    fn retire(&mut self, m: Holder) -> u64 {
        self.tick();
        self.d.retire_member(m.learner_id, Some(m.incarnation)).2
    }

    fn drain(&mut self, m: Holder, on: bool) {
        self.tick();
        self.d.set_drain(m.learner_id, on);
    }

    /// The controller's drain answer: drained once the id holds nothing.
    fn drained(&mut self, m: Holder) -> bool {
        self.tick();
        self.d.set_drain(m.learner_id, true) == 0
    }

    fn totals(&mut self) -> (u64, u64, u64) {
        self.tick();
        let t = self.d.totals;
        (t.trained, t.dropped, t.exhausted)
    }
}

fn run_case(case: &Json) {
    let name = case.at("name").str();
    let mut c = SyncerContract::new(case.at("config"));
    for (index, step) in case.at("steps").arr().iter().enumerate() {
        let where_ = format!("{name} step {index}: {step:?}");
        let op = step.at("op").str();
        match op {
            "advance" => {
                c.elapsed += Duration::from_secs_f64(step.at("s").num());
                continue;
            }
            "totals" => {
                let want = step.at("expect");
                let n = |k: &str| want.at(k).num() as u64;
                assert_eq!(c.totals(), (n("trained"), n("dropped"), n("exhausted")), "{where_}");
                continue;
            }
            _ => {}
        }
        let m = c.member(step.at("m").str());
        match op {
            "join" => assert_eq!(c.join(m), step.at("expect").num() as u64, "{where_}"),
            "claim" => {
                let got = c.claim(m, step.at("n").num() as u32);
                let want = match step.at("expect") {
                    Json::Str(s) if s == "wait" => ClaimAnswer::Wait,
                    Json::Str(s) if s == "end" => ClaimAnswer::End,
                    Json::Str(s) if s == "fenced" => ClaimAnswer::Fenced,
                    other => ClaimAnswer::Items(other.at("items").arr().iter().map(|v| v.num() as u64).collect()),
                };
                let got = match got {
                    ClaimAnswer::Items(mut items) => {
                        items.sort_unstable();
                        ClaimAnswer::Items(items)
                    }
                    other => other,
                };
                let want = match want {
                    ClaimAnswer::Items(mut items) => {
                        items.sort_unstable();
                        ClaimAnswer::Items(items)
                    }
                    other => other,
                };
                assert_eq!(got, want, "{where_}");
            }
            "done" | "drop" | "fail" | "release" => {
                let kind = match op {
                    "done" => ReportKind::Trained,
                    "drop" => ReportKind::Dropped,
                    "fail" => ReportKind::Requeue,
                    _ => ReportKind::Release,
                };
                let items: Vec<u64> = step.at("items").arr().iter().map(|v| v.num() as u64).collect();
                let ok = c.report(m, &items, kind);
                assert_eq!(if ok { "ok" } else { "not_held" }, step.at("expect").str(), "{where_}");
            }
            "holds" => {
                let want = step.at("expect") == &Json::Bool(true);
                assert_eq!(c.holds(m, step.at("item").num() as u64), want, "{where_}");
            }
            "beat" => c.beat(m),
            "fence" => {
                c.retire(m);
            }
            "retire" => assert_eq!(c.retire(m), step.at("expect").num() as u64, "{where_}"),
            "drain" => c.drain(m, true),
            "undrain" => c.drain(m, false),
            "drained" => assert_eq!(c.drained(m), step.at("expect") == &Json::Bool(true), "{where_}"),
            other => panic!("unknown op {other:?} ({where_})"),
        }
    }
}

#[test]
fn the_syncer_dispenser_conforms_to_every_work_contract_case() {
    let table = Parser::parse(CASES);
    let cases = table.at("cases").arr();
    assert!(cases.len() >= 10, "the case table is the shared one");
    for case in cases {
        run_case(case);
    }
}

#[test]
fn the_json_reader_reads_what_the_table_uses() {
    let v = Parser::parse(r#"{"a": [1, 2.5, -3], "b": {"c": "x\"y"}, "d": true, "e": null, "f": []}"#);
    assert_eq!(v.at("a").arr().iter().map(Json::num).collect::<Vec<_>>(), vec![1.0, 2.5, -3.0]);
    assert_eq!(v.at("b").at("c").str(), "x\"y");
    assert_eq!(v.at("d"), &Json::Bool(true));
    assert_eq!(v.at("e"), &Json::Null);
    assert!(v.at("f").arr().is_empty());
}
