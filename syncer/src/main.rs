mod elastic;
mod elastic_server;
mod iso_worker;
mod merge;
mod protocol;
mod server;
mod state;

use clap::Parser;

/// Yeto syncer: pull-driven fragment merging (weighted RDA/Avg)
/// with an SGD+Nesterov outer optimizer. See docs/PROTOCOL.md.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// TCP port to listen on.
    #[arg(long, default_value_t = 29400)]
    port: u16,
    /// Number of learners expected before training starts (M).
    #[arg(long)]
    learners: u32,
    /// Minimum quorum of learners per outer step (K).
    #[arg(long, default_value_t = 1)]
    quorum: u32,
    /// Upper bound on the post-quorum grace window, in milliseconds. The
    /// actual wait adapts per round to the learners' compute slack.
    #[arg(long, default_value_t = 1000)]
    grace_ms: u64,
    /// Safety margin on the computed grace slack (γ < 1).
    #[arg(long, default_value_t = 0.8)]
    grace_gamma: f64,
    /// Compute-overlap budget for the grace window, in learner inner steps (τ).
    #[arg(long, default_value_t = 2.0)]
    grace_tau: f64,
    /// Fragment rounds in flight at once ("two fragments in flight" at the
    /// paper's τ=2); 1 = serial rounds. Clamped to the fragment count.
    #[arg(long, default_value_t = 2)]
    pipeline: u32,
    /// Lower bound on time between consecutive round launches, in ms. WAN
    /// latency spaces merges naturally; on LAN/localhost this emulates the
    /// sync interval H the outer optimizer is tuned for. 0 = unthrottled.
    #[arg(long, default_value_t = 0)]
    min_round_interval_ms: u64,
    /// Target sync interval H (inner steps per fragment between merges):
    /// the launch floor adapts to the measured learner step time
    /// (H·ξ_step/P). Never binds where WAN round latency already exceeds
    /// it. 0 disables. Default 24 — the paper's design point; measured to
    /// match synchronous training where H≈2 costs ~+9%.
    #[arg(long, default_value_t = 24.0)]
    sync_interval_steps: f64,
    /// Pre-merge learner-delta correction: "heloco" or "none".
    #[arg(long, default_value = "heloco")]
    delta_correction: String,
    /// Give up waiting for quorum after this long.
    #[arg(long, default_value_t = 900)]
    quorum_timeout_s: u64,
    /// Give up waiting for final learner ACKs after this long. When omitted,
    /// inherit --quorum-timeout-s for backward compatibility.
    #[arg(long)]
    final_ack_timeout_s: Option<u64>,
    /// Total number of outer steps T (each syncs one fragment).
    #[arg(long)]
    total_steps: u64,
    /// Opt into strict dense-policy sweeps of exactly P fragments per logical
    /// local optimizer step.  The decoded layout must contain exactly P
    /// fragments; total_steps remains the number of fragment merges.
    #[arg(long)]
    policy_sweep_fragments: Option<u32>,
    /// Outer learning rate.
    #[arg(long, default_value_t = 0.7)]
    outer_lr: f32,
    /// Outer Nesterov momentum.
    #[arg(long, default_value_t = 0.9)]
    outer_momentum: f32,
    /// Iso spectrum-flattening implementation: the scalar reference kernel,
    /// or a persistent exact f32 Torch SVD worker.
    #[arg(long, default_value = "scalar")]
    iso_backend: String,
    /// Python executable used by --iso-backend=torch-svd.
    #[arg(long, default_value = "python3")]
    iso_worker_python: std::path::PathBuf,
    /// Torch device used by --iso-backend=torch-svd.
    #[arg(long, default_value = "cuda:0")]
    iso_worker_device: String,
    /// Explicit comma-separated Torch devices, one persistent worker per
    /// device. Overrides --iso-worker-device when non-empty.
    #[arg(long, value_delimiter = ',')]
    iso_worker_devices: Vec<String>,
    /// Maximum complete matrices waiting to enter the SVD workers.
    #[arg(long, default_value_t = 16)]
    iso_worker_queue_capacity: usize,
    /// Optional path to dump the final global parameters (flat f32 binary).
    #[arg(long)]
    final_state: Option<std::path::PathBuf>,
    /// Consistent-snapshot file (params, momentum, versions, ledger).
    #[arg(long)]
    checkpoint_path: Option<std::path::PathBuf>,
    /// Write the snapshot every N outer steps (0 disables).
    #[arg(long, default_value_t = 8)]
    checkpoint_every: u64,
    /// Resume from --checkpoint-path if it exists.
    #[arg(long, default_value_t = false)]
    resume: bool,
    /// Mark the terminal checkpoint as publishable after all merges finish.
    #[arg(long, default_value_t = false)]
    mark_final_checkpoint: bool,
    /// Exact local optimizer-step budget per learner (benchmark-only).
    #[arg(long)]
    learner_budget_steps: Option<u64>,
    /// JSONL event tape (merge records plus sweep ledger reconciliation cuts).
    #[arg(long)]
    event_tape: Option<std::path::PathBuf>,
    /// Maximum admitted lag between a round and a learner's base version.
    /// Omitted means unbounded (the existing SFT behavior).
    #[arg(long)]
    max_base_lag: Option<u64>,
    /// Learner contribution weighting used by AVG/RDA merges.
    #[arg(long, default_value = "tokens2-over-steps")]
    learner_weight: String,
    /// Require every HELLO to carry the canonical hash of this server's
    /// semantic launch profile. Generic clients remain compatible unless
    /// this is explicitly enabled.
    #[arg(long, default_value_t = false)]
    require_profile_binding: bool,
    /// Inter-island scheduling mode: legacy (default, unchanged behavior)
    /// or elastic (change rl-inter-island-scheduling).
    #[arg(long, default_value = "legacy")]
    island_scheduling_mode: String,
    /// elastic: capacity fraction that must arrive before an outer step (θ).
    #[arg(long, default_value_t = elastic::DEFAULT_QUORUM_THETA)]
    quorum_theta: f64,
    /// elastic: per-step discount of a late (carried-over) delta (γ).
    #[arg(long, default_value_t = elastic::DEFAULT_CARRY_GAMMA)]
    carry_gamma: f64,
    /// elastic: soft deadline in seconds; defaults to --quorum-timeout-s.
    #[arg(long)]
    soft_deadline_s: Option<u64>,
    /// elastic: minimum arrived islands for a step.
    #[arg(long, default_value_t = elastic::DEFAULT_Q_MIN)]
    q_min: u32,
    /// elastic: largest base lag still carried over; larger is rejected.
    #[arg(long, default_value_t = elastic::DEFAULT_MAX_CARRY_LAG)]
    max_carry_lag: u32,
    /// HMAC-SHA256 island key; falls back to the YETO_ISLAND_HMAC_KEY
    /// environment variable. elastic: signs the elastic message types.
    /// legacy: every HELLO must carry an HMAC under this key.
    #[arg(long)]
    island_hmac_key: Option<String>,
    /// legacy only: run without island authentication (no HMAC key). Only
    /// for local tests and benchmarks; also enabled by
    /// YETO_SYNCER_ALLOW_UNAUTHENTICATED=1. Never set by the launcher's RL path.
    #[arg(long, default_value_t = false)]
    allow_unauthenticated_islands: bool,
    /// Island contract (64 lowercase hex chars) pinned by the head's config:
    /// legacy HELLO session contracts and elastic JOIN identities must match
    /// it, so the first island to connect no longer decides the contract.
    #[arg(long)]
    expected_island_contract: Option<String>,
    /// elastic: membership lease in seconds (no heartbeat -> removed).
    #[arg(long, default_value_t = 30.0)]
    island_lease_s: f64,
    /// elastic: fencing token of this syncer incarnation; bump on every
    /// restart (D-S6 will persist it in the checkpoint).
    #[arg(long, default_value_t = 0)]
    syncer_epoch: u64,
    /// elastic: after total_steps keep answering FINISHED for this many
    /// seconds (or until every member left); defaults to the soft deadline.
    #[arg(long)]
    final_grace_s: Option<u64>,
}

impl Args {
    fn resolved_final_ack_timeout_s(&self) -> u64 {
        self.final_ack_timeout_s.unwrap_or(self.quorum_timeout_s)
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        // Logs are consumed by launchers, tests, and event collectors. Keep
        // their field syntax stable even when CI advertises a color-capable
        // terminal through environment variables.
        .with_ansi(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Args::parse();
    let delta_correction = match args.delta_correction.as_str() {
        "heloco" => true,
        "none" => false,
        other => anyhow::bail!("--delta-correction must be 'heloco' or 'none', got {other:?}"),
    };
    let learner_weight = match args.learner_weight.as_str() {
        "tokens2-over-steps" => server::LearnerWeight::Tokens2OverSteps,
        "equal" => server::LearnerWeight::Equal,
        other => {
            anyhow::bail!("--learner-weight must be 'tokens2-over-steps' or 'equal', got {other:?}")
        }
    };
    let final_ack_timeout_s = args.resolved_final_ack_timeout_s();
    let devices = if args.iso_worker_devices.is_empty() {
        vec![args.iso_worker_device.clone()]
    } else {
        args.iso_worker_devices
    };
    if devices.iter().any(String::is_empty) {
        anyhow::bail!("--iso-worker-devices cannot contain an empty device");
    }
    let unique: std::collections::HashSet<_> = devices.iter().collect();
    if unique.len() != devices.len() {
        anyhow::bail!("--iso-worker-devices cannot contain duplicates");
    }
    let iso_backend = iso_worker::IsoBackendConfig {
        kind: args.iso_backend.parse()?,
        python: args.iso_worker_python,
        device: args.iso_worker_device,
    }
    .with_pool(devices, args.iso_worker_queue_capacity)?;
    let island_scheduling = elastic::IslandSchedulingMode::parse(
        &args.island_scheduling_mode,
        args.quorum_theta,
        args.carry_gamma,
        args.soft_deadline_s.unwrap_or(args.quorum_timeout_s),
        args.q_min,
        args.max_carry_lag,
    )?;
    let island_hmac_key = args
        .island_hmac_key
        .clone()
        .or_else(|| std::env::var("YETO_ISLAND_HMAC_KEY").ok())
        .filter(|k| !k.is_empty())
        .map(String::into_bytes);
    if matches!(island_scheduling, elastic::IslandSchedulingMode::Elastic(_))
        && island_hmac_key.is_none()
    {
        anyhow::bail!("elastic mode requires --island-hmac-key or YETO_ISLAND_HMAC_KEY");
    }
    if island_hmac_key.is_none() {
        let allowed = args.allow_unauthenticated_islands
            || std::env::var("YETO_SYNCER_ALLOW_UNAUTHENTICATED").as_deref() == Ok("1");
        if !allowed {
            anyhow::bail!(
                "legacy mode requires an island HMAC key (--island-hmac-key or \
                 YETO_ISLAND_HMAC_KEY); pass --allow-unauthenticated-islands only for local tests"
            );
        }
    }
    let expected_island_contract = args
        .expected_island_contract
        .as_deref()
        .map(parse_hex32)
        .transpose()?;
    let cfg = server::Config {
        port: args.port,
        learners: args.learners,
        quorum: args.quorum,
        grace_ms: args.grace_ms,
        grace_gamma: args.grace_gamma,
        grace_tau: args.grace_tau,
        pipeline: args.pipeline,
        min_round_interval_ms: args.min_round_interval_ms,
        sync_interval_steps: args.sync_interval_steps,
        delta_correction,
        quorum_timeout_s: args.quorum_timeout_s,
        final_ack_timeout_s,
        total_steps: args.total_steps,
        policy_sweep_fragments: args.policy_sweep_fragments,
        outer_lr: args.outer_lr,
        outer_momentum: args.outer_momentum,
        iso_backend,
        final_state: args.final_state,
        checkpoint_path: args.checkpoint_path,
        checkpoint_every: args.checkpoint_every,
        resume: args.resume,
        mark_final_checkpoint: args.mark_final_checkpoint,
        learner_budget_steps: args.learner_budget_steps,
        event_tape: args.event_tape,
        max_base_lag: args.max_base_lag,
        learner_weight,
        require_profile_binding: args.require_profile_binding,
        island_scheduling,
        island_hmac_key,
        expected_island_contract,
        island_lease_s: args.island_lease_s,
        syncer_epoch: args.syncer_epoch,
        elastic_final_grace_s: args.final_grace_s,
    };
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(server::run(cfg))
}

fn parse_hex32(text: &str) -> anyhow::Result<[u8; 32]> {
    let ok = text.len() == 64 && text.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    if !ok {
        anyhow::bail!("--expected-island-contract must be 64 lowercase hex chars");
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[2 * i..2 * i + 2], 16)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_ack_timeout_is_compatible_when_omitted_and_distinct_when_set() {
        let inherited = Args::try_parse_from([
            "yeto-syncer",
            "--learners",
            "1",
            "--quorum-timeout-s",
            "2",
            "--total-steps",
            "1",
        ])
        .unwrap();
        assert_eq!(inherited.resolved_final_ack_timeout_s(), 2);

        let explicit = Args::try_parse_from([
            "yeto-syncer",
            "--learners",
            "1",
            "--quorum-timeout-s",
            "900",
            "--final-ack-timeout-s",
            "3600",
            "--total-steps",
            "1",
        ])
        .unwrap();
        assert_eq!(explicit.resolved_final_ack_timeout_s(), 3600);
    }

    #[test]
    fn policy_sweep_fragments_is_optional_and_parsed_without_changing_total_steps() {
        let legacy =
            Args::try_parse_from(["yeto-syncer", "--learners", "2", "--total-steps", "8"]).unwrap();
        assert_eq!(legacy.policy_sweep_fragments, None);
        assert_eq!(legacy.total_steps, 8);

        let sweep = Args::try_parse_from([
            "yeto-syncer",
            "--learners",
            "2",
            "--total-steps",
            "8",
            "--policy-sweep-fragments",
            "4",
        ])
        .unwrap();
        assert_eq!(sweep.policy_sweep_fragments, Some(4));
        assert_eq!(sweep.total_steps, 8);
    }

    #[test]
    fn island_scheduling_defaults_to_legacy_and_soft_deadline_inherits() {
        let a = Args::try_parse_from(["yeto-syncer", "--learners", "2", "--total-steps", "8"])
            .unwrap();
        assert_eq!(a.island_scheduling_mode, "legacy");
        assert_eq!((a.quorum_theta, a.carry_gamma, a.q_min, a.max_carry_lag), (0.75, 0.5, 1, 2));
        assert_eq!(a.soft_deadline_s, None);
        let e = Args::try_parse_from([
            "yeto-syncer", "--learners", "2", "--total-steps", "8",
            "--island-scheduling-mode", "elastic", "--quorum-theta", "0.6",
            "--carry-gamma", "0.25", "--soft-deadline-s", "30", "--q-min", "2",
            "--max-carry-lag", "3",
        ])
        .unwrap();
        let mode = elastic::IslandSchedulingMode::parse(
            &e.island_scheduling_mode, e.quorum_theta, e.carry_gamma,
            e.soft_deadline_s.unwrap_or(e.quorum_timeout_s), e.q_min, e.max_carry_lag,
        )
        .unwrap();
        assert_eq!(
            mode,
            elastic::IslandSchedulingMode::Elastic(elastic::ElasticParams {
                quorum_theta: 0.6, carry_gamma: 0.25, soft_deadline_s: 30, q_min: 2, max_carry_lag: 3,
            })
        );
    }
}
