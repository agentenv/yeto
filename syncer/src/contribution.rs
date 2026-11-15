//! Per-island contribution and refusal alerts of an island pool (warn only).
//!
//! A slow island whose pushes keep lagging past `--max-base-lag` is refused
//! (`carry_lag_exceeded`) every round: it burns a whole machine, adds nothing
//! to the merge, and `status.json` still reports it healthy (`capacity 1.0`,
//! lease renewed). This tracker counts, per island and over the last `window`
//! merged rounds, the rounds it contributed a positive merge weight to and its
//! refused pushes, and says when to warn. It never changes a merge, an
//! admission or a membership decision.
//!
//! `YETO_SYNCER_CONTRIBUTION_ALERT`: unset = defaults (`window=8`,
//! `min_share=0`, `max_reject_share=0.5`); `0`/`off` = no alert records (the
//! `status.json` fields stay); `window=N,min_share=F,max_reject_share=F`
//! (each optional). An island alerts when its window is full and its
//! contribution share is `<= min_share` or its refused-push share is
//! `> max_reject_share`; then at most once per further `window` rounds.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub const ENV: &str = "YETO_SYNCER_CONTRIBUTION_ALERT";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlertConfig {
    pub window: usize,
    pub min_share: f64,
    pub max_reject_share: f64,
    pub enabled: bool,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self { window: 8, min_share: 0.0, max_reject_share: 0.5, enabled: true }
    }
}

impl AlertConfig {
    /// Parse the env value; a malformed value keeps the defaults with the
    /// error returned for the caller to log (an alert never stops a syncer).
    pub fn parse(raw: Option<&str>) -> (Self, Option<String>) {
        let mut cfg = Self::default();
        let raw = raw.map(str::trim).unwrap_or("");
        if raw.is_empty() {
            return (cfg, None);
        }
        if matches!(raw, "0" | "off" | "false") {
            cfg.enabled = false;
            return (cfg, None);
        }
        let mut parsed = cfg;
        for part in raw.split(',') {
            let (key, value) = match part.trim().split_once('=') {
                Some(kv) => kv,
                None => return (cfg, Some(format!("{ENV}={raw:?}: malformed {part:?}"))),
            };
            let ok = match key {
                "window" => value.parse::<usize>().ok().filter(|w| *w > 0).map(|w| parsed.window = w),
                "min_share" => value.parse::<f64>().ok().filter(|v| (0.0..=1.0).contains(v)).map(|v| parsed.min_share = v),
                "max_reject_share" => value
                    .parse::<f64>()
                    .ok()
                    .filter(|v| (0.0..=1.0).contains(v))
                    .map(|v| parsed.max_reject_share = v),
                _ => None,
            };
            if ok.is_none() {
                return (cfg, Some(format!("{ENV}={raw:?}: unknown key or bad value in {part:?}")));
            }
        }
        (parsed, None)
    }

    pub fn from_env() -> (Self, Option<String>) {
        Self::parse(std::env::var(ENV).ok().as_deref())
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct RoundEntry {
    contributed: bool,
    merged_pushes: u32,
    rejected_pushes: u32,
    /// Refusal reasons of this round's `rejected_pushes` (sums to it).
    reasons: BTreeMap<String, u32>,
}

#[derive(Debug, Default)]
struct Island {
    window: VecDeque<RoundEntry>,
    pending_rejected: u32,
    pending_reasons: BTreeMap<String, u32>,
    rounds_since_alert: usize,
    alerted: bool,
}

/// One island's window figures (status.json and the alert record).
#[derive(Clone, Debug, PartialEq)]
pub struct IslandFigures {
    pub island_id: u32,
    pub rounds: usize,
    pub contributed_rounds: usize,
    pub contribution_share: f64,
    pub merged_pushes: u32,
    pub rejected_pushes: u32,
    pub reject_share: f64,
    pub reasons: BTreeMap<String, u32>,
}

impl IslandFigures {
    pub fn status_fields(&self) -> String {
        format!(
            "\"contribution_share\":{:?},\"contribution_rounds\":{},\"reject_share\":{:?},\"rejected_pushes\":{}",
            self.contribution_share, self.rounds, self.reject_share, self.rejected_pushes
        )
    }

    pub fn alert_fields(&self, cfg: &AlertConfig) -> String {
        let reasons: Vec<String> = self.reasons.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect();
        format!(
            "\"island_id\":{},\"window\":{},\"contributed_rounds\":{},\"contribution_share\":{:?},\"merged_pushes\":{},\"rejected_pushes\":{},\"reject_share\":{:?},\"reasons\":{{{}}},\"min_share\":{:?},\"max_reject_share\":{:?}",
            self.island_id,
            self.rounds,
            self.contributed_rounds,
            self.contribution_share,
            self.merged_pushes,
            self.rejected_pushes,
            self.reject_share,
            reasons.join(","),
            cfg.min_share,
            cfg.max_reject_share
        )
    }
}

#[derive(Debug, Default)]
pub struct ContributionTracker {
    pub cfg: AlertConfig,
    islands: BTreeMap<u32, Island>,
}

impl ContributionTracker {
    pub fn new(cfg: AlertConfig) -> Self {
        Self { cfg, islands: BTreeMap::new() }
    }

    /// A refused push of `island_id` (counted into its next merged round).
    pub fn note_rejected(&mut self, island_id: u32, reason: &str) {
        let island = self.islands.entry(island_id).or_default();
        island.pending_rejected += 1;
        *island.pending_reasons.entry(reason.to_string()).or_default() += 1;
    }

    /// One merged round: the islands it expected and `(island, weight)` of
    /// each merged push (carried ones included). Returns the islands that
    /// alert now.
    pub fn note_round(&mut self, expected: &[u32], merged: &[(u32, f64)]) -> Vec<IslandFigures> {
        let mut seen: BTreeSet<u32> = expected.iter().copied().collect();
        seen.extend(merged.iter().map(|(id, _)| *id));
        // an island refused this round without being expected (left, re-joining) counts too
        seen.extend(self.islands.iter().filter(|(_, i)| i.pending_rejected > 0).map(|(id, _)| *id));
        let window = self.cfg.window;
        let mut alerts = Vec::new();
        for id in seen {
            let pushes: Vec<f64> = merged.iter().filter(|(m, _)| *m == id).map(|(_, w)| *w).collect();
            let island = self.islands.entry(id).or_default();
            let entry = RoundEntry {
                contributed: pushes.iter().any(|w| *w > 0.0),
                merged_pushes: pushes.len() as u32,
                rejected_pushes: std::mem::take(&mut island.pending_rejected),
                reasons: std::mem::take(&mut island.pending_reasons),
            };
            island.window.push_back(entry);
            while island.window.len() > window {
                island.window.pop_front();
            }
            island.rounds_since_alert += 1;
            let figures = figures(id, island);
            let low = figures.rounds >= window
                && (figures.contribution_share <= self.cfg.min_share
                    || figures.reject_share > self.cfg.max_reject_share);
            if self.cfg.enabled && low && (!island.alerted || island.rounds_since_alert >= window) {
                island.alerted = true;
                island.rounds_since_alert = 0;
                alerts.push(figures);
            }
        }
        alerts
    }

    pub fn figures(&self, island_id: u32) -> Option<IslandFigures> {
        self.islands.get(&island_id).filter(|i| !i.window.is_empty()).map(|i| figures(island_id, i))
    }
}

fn figures(island_id: u32, island: &Island) -> IslandFigures {
    let rounds = island.window.len();
    let contributed_rounds = island.window.iter().filter(|e| e.contributed).count();
    let merged_pushes: u32 = island.window.iter().map(|e| e.merged_pushes).sum();
    let rejected_pushes: u32 = island.window.iter().map(|e| e.rejected_pushes).sum();
    let pushes = merged_pushes + rejected_pushes;
    // over the same window as `rejected_pushes`, so the two always agree
    let mut reasons = BTreeMap::new();
    for (reason, n) in island.window.iter().flat_map(|e| &e.reasons) {
        *reasons.entry(reason.clone()).or_default() += *n;
    }
    IslandFigures {
        island_id,
        rounds,
        contributed_rounds,
        contribution_share: if rounds == 0 { 0.0 } else { contributed_rounds as f64 / rounds as f64 },
        merged_pushes,
        rejected_pushes,
        reject_share: if pushes == 0 { 0.0 } else { f64::from(rejected_pushes) / f64::from(pushes) },
        reasons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults_off_and_values() {
        assert_eq!(AlertConfig::parse(None), (AlertConfig::default(), None));
        assert!(!AlertConfig::parse(Some("off")).0.enabled);
        let (cfg, err) = AlertConfig::parse(Some("window=4,min_share=0.1,max_reject_share=0.9"));
        assert!(err.is_none());
        assert_eq!((cfg.window, cfg.min_share, cfg.max_reject_share), (4, 0.1, 0.9));
        let (cfg, err) = AlertConfig::parse(Some("window=0"));
        assert!(err.is_some());
        assert_eq!(cfg, AlertConfig::default(), "a bad value keeps the defaults");
        assert!(AlertConfig::parse(Some("bogus=1")).1.is_some());
    }

    #[test]
    fn a_refused_slow_island_alerts_once_per_window_and_a_healthy_one_never() {
        // A08 r1: quorum 1, max_base_lag 1 -- island 1 pushes every round and is
        // refused every round; island 0 alone carries the merge.
        let mut t = ContributionTracker::new(AlertConfig { window: 4, ..Default::default() });
        let mut alerts = Vec::new();
        for _ in 0..12 {
            t.note_rejected(1, "carry_lag_exceeded");
            alerts.push(t.note_round(&[0, 1], &[(0, 1.0)]));
        }
        let fired: Vec<usize> = alerts.iter().enumerate().filter(|(_, a)| !a.is_empty()).map(|(i, _)| i).collect();
        assert_eq!(fired, vec![3, 7, 11], "first at a full window, then once per window");
        let a = &alerts[3][0];
        assert_eq!((a.island_id, a.contributed_rounds, a.rejected_pushes), (1, 0, 4));
        assert_eq!(a.reject_share, 1.0);
        assert_eq!(a.reasons.get("carry_lag_exceeded"), Some(&4));
        let ok = t.figures(0).unwrap();
        assert_eq!((ok.contribution_share, ok.reject_share), (1.0, 0.0));
        assert!(a.alert_fields(&t.cfg).contains("\"island_id\":1"));
        // later alerts count reasons over the window, matching rejected_pushes
        for later in [&alerts[7][0], &alerts[11][0]] {
            assert_eq!(later.rejected_pushes, 4);
            assert_eq!(later.reasons.values().sum::<u32>(), later.rejected_pushes);
            assert_eq!(later.reasons.get("carry_lag_exceeded"), Some(&4));
        }
    }

    #[test]
    fn a_carried_contribution_counts_and_off_never_alerts() {
        let mut t = ContributionTracker::new(AlertConfig { window: 2, ..Default::default() });
        for _ in 0..4 {
            assert!(t.note_round(&[0, 1], &[(0, 0.8), (1, 0.2)]).is_empty());
        }
        let mut off = ContributionTracker::new(AlertConfig { window: 2, enabled: false, ..Default::default() });
        for _ in 0..4 {
            assert!(off.note_round(&[0, 1], &[(0, 1.0)]).is_empty());
        }
        assert_eq!(off.figures(1).unwrap().contribution_share, 0.0, "figures stay for status.json");
    }
}
