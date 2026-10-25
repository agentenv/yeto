mod fold;
mod hmac;
mod shard;
#[cfg(test)]
mod head_memory;
mod local_init;
mod iso_worker;
mod membership;
mod recovery;
mod merge;
mod dispense;
mod dispense_service;
mod members;
mod downlink;
mod eval_checkpoint;
mod protocol;
mod sample_relay;
mod server;
mod state;
mod subscriber;

use clap::Parser;

/// Yeto syncer: pull-driven fragment merging (weighted RDA/Avg)
/// with an SGD+Nesterov outer optimizer. See docs/PROTOCOL.md.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// TCP port to listen on.
    #[arg(long, default_value_t = 29400)]
    port: u16,
    /// Number of learners (M). Training starts when all M are connected; with
    /// membership options it starts once the launch quorum is connected.
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
    /// Lower bound on the grace window, in milliseconds (still capped by
    /// --grace-ms). 0 = off.
    #[arg(long, default_value_t = 0)]
    grace_min_ms: u64,
    /// Grace floor from measured arrival gaps: the window is at least this
    /// quantile (0..1] of the recent gaps between a round's first push and
    /// each later push of it, late pushes included. Still capped by
    /// --grace-ms. 0 = off.
    #[arg(long, default_value_t = 0.0)]
    grace_arrival_quantile: f64,
    /// Terminal grace, in milliseconds: the last round of each fragment
    /// waits up to this long for every expected island (not capped by
    /// --grace-ms), so a late delta of a last round still reaches the final
    /// parameters. 0 = off.
    #[arg(long, default_value_t = 0)]
    terminal_grace_ms: u64,
    /// Fragment rounds in flight at once (two by default, matching the
    /// default --grace-tau 2); 1 = serial rounds. Clamped to the fragment count.
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
    /// it. 0 disables. Default 24; measured to
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
    /// Delayed Nesterov outer optimizer: apply the outer gradient every
    /// step, but update the
    /// momentum only every N steps with the mean of the N gradients.
    /// Unset or 1 = plain Nesterov.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    outer_delayed_nesterov_period: Option<u32>,
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
    /// Also write the cut at these global steps (the `step=` of the "outer
    /// step" log; comma-separated, strictly increasing, below --total-steps)
    /// to `<checkpoint-path>.step-<t>`, on a background thread while rounds
    /// keep merging. --resume never reads these files.
    #[arg(long, value_delimiter = ',')]
    checkpoint_at_steps: Vec<u64>,
    /// Policy sweep: write per-round checkpoints off the scheduler; each
    /// round is broadcast only after its checkpoint is durable.
    #[arg(long, default_value_t = false)]
    async_checkpoint: bool,
    /// Learners whose session asks for bf16 downlink deltas get a full f32
    /// BCAST every N-th version of a fragment (N >= 2). Unset: such
    /// sessions are refused.
    #[arg(long)]
    bcast_delta_full_every: Option<u64>,
    /// Resume from --checkpoint-path if it exists.
    #[arg(long, default_value_t = false)]
    resume: bool,
    /// Mark the terminal checkpoint as publishable after all merges finish.
    #[arg(long, default_value_t = false)]
    mark_final_checkpoint: bool,
    /// Resume only a checkpoint whose recorded semantic profile (the
    /// `<checkpoint>.profile` file written at every start) matches this
    /// syncer's; a checkpoint without the record is refused too.
    #[arg(long, default_value_t = false)]
    require_profile_record: bool,
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
    /// Capacity fraction that must arrive before a merge (θ). Fragment
    /// rounds: unset keeps the count quorum K (--quorum); set replaces it.
    #[arg(long)]
    quorum_theta: Option<f64>,
    /// Weight discount γ^lag of a late delta. Fragment rounds: lag in
    /// fragment versions, unset = 1 (no discount).
    #[arg(long)]
    carry_gamma: Option<f64>,
    /// Soft deadline in seconds: merge what arrived even below quorum.
    /// Unset = off (also caps the grace window).
    #[arg(long)]
    soft_deadline_s: Option<u64>,
    /// Minimum arrived islands for a capacity-quorum or soft-deadline merge.
    #[arg(long, default_value_t = 1)]
    q_min: u32,
    /// Fragment rounds: relative island capacity for --quorum-theta, as
    /// comma-separated id=capacity (unlisted islands have capacity 1).
    #[arg(long, value_delimiter = ',')]
    island_capacity: Vec<String>,
    /// Fragment rounds: admit island ids 0..N (islands may join after the
    /// first --learners). Unset = 0..--learners.
    #[arg(long)]
    max_learners: Option<u32>,
    /// HMAC-SHA256 island key; falls back to the YETO_ISLAND_HMAC_KEY
    /// environment variable. Every HELLO must carry an HMAC under this key.
    #[arg(long)]
    island_hmac_key: Option<String>,
    /// Run without island authentication (no HMAC key). Only
    /// for local tests and benchmarks; also enabled by
    /// YETO_SYNCER_ALLOW_UNAUTHENTICATED=1. Never set by the launcher's RL path.
    #[arg(long, default_value_t = false)]
    allow_unauthenticated_islands: bool,
    /// Island contract (64 lowercase hex chars) pinned by the head's config:
    /// HELLO session contracts must match it, so the first island to
    /// connect no longer decides the contract.
    #[arg(long)]
    expected_island_contract: Option<String>,
    /// Membership lease in seconds (no heartbeat -> removed; the island
    /// must re-JOIN). Unset = no lease.
    #[arg(long)]
    island_lease_s: Option<f64>,
    /// Fencing token of this syncer incarnation; bump on every restart.
    /// A HELLO with a lower epoch is refused (a HELLO
    /// without one is admitted); persisted next to the checkpoint.
    #[arg(long, default_value_t = 0)]
    syncer_epoch: u64,
    /// Fragment rounds: bound H on one peer recovery, in syncer steps. Unset:
    /// recovery steps are measured and recorded but not checked.
    #[arg(long)]
    recovery_bound_steps: Option<u64>,
    /// Fragment rounds: donors asked before the degraded recovery path.
    #[arg(long, default_value_t = 3)]
    recovery_max_donor_tries: u32,
    /// Fragment rounds: wait for one donor snapshot before asking the next.
    #[arg(long, default_value_t = 600)]
    recovery_donor_timeout_s: u64,
    /// Dynamic membership: after total_steps wait this many seconds for the
    /// islands still connected to acknowledge the final cut, then finish
    /// without the missing ones. Unset = every island at the final cut must
    /// acknowledge within --final-ack-timeout-s.
    #[arg(long)]
    final_grace_s: Option<u64>,
    /// Rounds the exact fold cannot reproduce (quorum below the roster or
    /// token weights): buffered keeps every push until the round closes
    /// (historical bits); streaming folds each push on arrival (S27 P4b).
    #[arg(long, default_value = "buffered")]
    weighted_fold: String,
    /// MoE: weight each expert tensor of a push by the tokens the island
    /// routed to that expert during the outer interval, relative to an even
    /// split over the layer's experts (an expert the island never routed to
    /// gets weight 0). Islands must send the counts. Off by default.
    #[arg(long)]
    expert_token_weights: bool,
    /// What the final cut carries to every island: `full` (every f32
    /// fragment, then the version manifest) or `manifest` (the version
    /// manifest only; islands acknowledge without downloading the final
    /// parameters, which stay in --checkpoint-path / --final-state).
    #[arg(long, default_value = "full", value_parser = ["full", "manifest"])]
    final_cut_payload: String,
    /// S27 P2 task dispenser: islands claim the rows of --data they train
    /// from this syncer (TASK_* connections, docs/PROTOCOL.md "Task
    /// dispenser"). Off by default.
    #[arg(long)]
    dispense: bool,
    /// Dispenser lease: an island process silent this long loses its
    /// unreported rows to the other islands.
    #[arg(long, default_value_t = 300.0)]
    dispense_lease_s: f64,
    /// Passes over the rows before the data ends; unset = unlimited (every
    /// pass after the first is reshuffled on the islands).
    #[arg(long)]
    dispense_passes: Option<u64>,
    /// How often one row may be requeued after its group was too stale to train.
    #[arg(long, default_value_t = 1)]
    dispense_max_redo: u32,
    /// Largest block one claim may ask for.
    #[arg(long, default_value_t = 65536)]
    dispense_max_block: u32,
    /// JSONL log of every dispenser event.
    #[arg(long)]
    dispense_log: Option<std::path::PathBuf>,
    /// Fragment-sharded head (S27 P4b): this process owns the fragment
    /// range of shard --shard-index out of --shard-count and runs only the
    /// global steps t ≡ shard-index+1 (mod shard-count).  Default unsharded.
    #[arg(long, default_value_t = 0)]
    shard_index: u32,
    #[arg(long, default_value_t = 1)]
    shard_count: u32,
    /// S27 G-EA: admit up to N receive-only subscribers at once (sampler
    /// islands on any cloud that follow the global weights as bf16
    /// broadcasts; never in quorum, roster or final cut; their ids are
    /// 2^31 + 31 bits of their sampler id's SHA-256, as the task dispenser's
    /// sampler holders). 0 = off: SUBSCRIBE is refused.
    #[arg(long, default_value_t = 0)]
    max_subscribers: u32,
    /// S27 G-EA: serve the sample relay (sampler islands PUT rollout
    /// samples, a training island TAKEs them) on this port. Off by default.
    #[arg(long)]
    sample_relay: bool,
    /// Sample relay: most items held per channel.
    #[arg(long, default_value_t = 4096)]
    sample_relay_max_items: usize,
    /// Sample relay: most bytes held per channel, in MiB.
    #[arg(long, default_value_t = 2048)]
    sample_relay_max_mib: u64,
    /// Sample relay: largest item, in MiB.
    #[arg(long, default_value_t = 64)]
    sample_relay_max_item_mib: u64,
}

/// Membership options of the fragment-round server; every unset flag keeps
/// the fixed-membership behavior.
fn parse_membership(args: &Args) -> anyhow::Result<membership::MembershipConfig> {
    let mut capacities = std::collections::BTreeMap::new();
    for entry in &args.island_capacity {
        let (id, cap) = entry
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--island-capacity entry {entry:?} is not id=capacity"))?;
        let id: u32 = id.trim().parse()?;
        let cap: f64 = cap.trim().parse()?;
        if capacities.insert(id, cap).is_some() {
            anyhow::bail!("--island-capacity lists island {id} twice");
        }
    }
    let seconds = |s: f64, flag: &str| {
        std::time::Duration::try_from_secs_f64(s)
            .map_err(|e| anyhow::anyhow!("{flag} must be a non-negative number of seconds: {e}"))
    };
    Ok(membership::MembershipConfig {
        quorum_rule: match args.quorum_theta {
            None => membership::QuorumRule::Count,
            Some(theta) => membership::QuorumRule::Capacity { theta, q_min: args.q_min },
        },
        lease: args.island_lease_s.map(|s| seconds(s, "--island-lease-s")).transpose()?,
        soft_deadline: args.soft_deadline_s.map(std::time::Duration::from_secs),
        carry_gamma: args.carry_gamma.unwrap_or(1.0),
        capacities,
        max_learners: args.max_learners,
        final_grace: args.final_grace_s.map(std::time::Duration::from_secs),
    })
}

impl Args {
    fn shard_plan(&self) -> anyhow::Result<shard::ShardPlan> {
        shard::ShardPlan::new(self.shard_index, self.shard_count)
    }

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
    let dispense = dispense_config(&args)?;
    let shard = args.shard_plan()?;
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
    let membership = parse_membership(&args)?;
    let recovery = recovery::RecoveryConfig {
        bound_steps: args.recovery_bound_steps,
        max_donor_tries: args.recovery_max_donor_tries,
        donor_timeout: std::time::Duration::from_secs(args.recovery_donor_timeout_s),
    };
    recovery.validate()?;
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
    let island_hmac_key = args
        .island_hmac_key
        .clone()
        .or_else(|| std::env::var("YETO_ISLAND_HMAC_KEY").ok())
        .filter(|k| !k.is_empty())
        .map(String::into_bytes);
    if island_hmac_key.is_none() {
        let allowed = args.allow_unauthenticated_islands
            || std::env::var("YETO_SYNCER_ALLOW_UNAUTHENTICATED").as_deref() == Ok("1");
        if !allowed {
            anyhow::bail!(
                "the syncer requires an island HMAC key (--island-hmac-key or \
                 YETO_ISLAND_HMAC_KEY); pass --allow-unauthenticated-islands only for local tests"
            );
        }
    }
    if !(0.0..=1.0).contains(&args.grace_arrival_quantile) {
        anyhow::bail!("--grace-arrival-quantile must be in [0, 1] (0 = off)");
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
        grace_min_ms: args.grace_min_ms,
        grace_arrival_quantile: args.grace_arrival_quantile,
        terminal_grace_ms: args.terminal_grace_ms,
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
        // N = 1 is exactly plain Nesterov; keep its code path and hashes.
        outer_delayed_nesterov_period: args.outer_delayed_nesterov_period.filter(|&n| n > 1),
        iso_backend,
        final_state: args.final_state,
        checkpoint_path: args.checkpoint_path,
        checkpoint_every: args.checkpoint_every,
        checkpoint_at_steps: args.checkpoint_at_steps,
        async_checkpoint: args.async_checkpoint,
        bcast_delta_full_every: args.bcast_delta_full_every,
        resume: args.resume,
        mark_final_checkpoint: args.mark_final_checkpoint,
        require_profile_record: args.require_profile_record,
        learner_budget_steps: args.learner_budget_steps,
        event_tape: args.event_tape,
        max_base_lag: args.max_base_lag,
        learner_weight,
        require_profile_binding: args.require_profile_binding,
        island_hmac_key,
        expected_island_contract,
        syncer_epoch: args.syncer_epoch,
        membership,
        recovery,
        weighted_fold: args.weighted_fold.parse()?,
        shard,
        expert_token_weights: args.expert_token_weights,
        final_cut_manifest_only: args.final_cut_payload == "manifest",
        dispense,
        dispense_log: args.dispense_log.clone(),
        max_subscribers: args.max_subscribers,
        sample_relay: args.sample_relay.then_some(sample_relay::RelayConfig {
            max_items: args.sample_relay_max_items,
            max_bytes: args.sample_relay_max_mib << 20,
            max_item_bytes: args.sample_relay_max_item_mib << 20,
        }),
    };
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(server::run(cfg))
}

fn dispense_config(args: &Args) -> anyhow::Result<Option<dispense::DispenseConfig>> {
    let named = args.dispense_passes.is_some() || args.dispense_log.is_some();
    if !args.dispense {
        if named {
            anyhow::bail!("--dispense-passes / --dispense-log need --dispense");
        }
        return Ok(None);
    }
    if !(args.dispense_lease_s.is_finite() && args.dispense_lease_s > 0.0) {
        anyhow::bail!("--dispense-lease-s must be positive");
    }
    let cfg = dispense::DispenseConfig {
        lease: std::time::Duration::from_secs_f64(args.dispense_lease_s),
        passes: args.dispense_passes,
        max_redo: args.dispense_max_redo,
        max_block: args.dispense_max_block,
    };
    cfg.validate()?;
    Ok(Some(cfg))
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
    fn checkpoint_at_steps_is_a_comma_separated_step_list() {
        let base = ["yeto-syncer", "--learners", "1", "--total-steps", "200"];
        assert!(Args::try_parse_from(base).unwrap().checkpoint_at_steps.is_empty());
        let mut argv = base.to_vec();
        argv.extend(["--checkpoint-at-steps", "75,149,190"]);
        assert_eq!(Args::try_parse_from(argv).unwrap().checkpoint_at_steps, vec![75, 149, 190]);
        let mut argv = base.to_vec();
        argv.extend(["--checkpoint-at-steps", "75,x"]);
        assert!(Args::try_parse_from(argv).is_err());
    }

    #[test]
    fn membership_is_static_by_default_and_the_retired_mode_flag_is_refused() {
        let a = Args::try_parse_from(["yeto-syncer", "--learners", "2", "--total-steps", "8"])
            .unwrap();
        assert_eq!((a.quorum_theta, a.carry_gamma, a.q_min), (None, None, 1));
        assert!(parse_membership(&a).unwrap().is_static(), "fragment rounds stay fixed by default");
        assert_eq!((a.soft_deadline_s, a.final_grace_s), (None, None));
        for retired in [["--island-scheduling-mode", "elastic"], ["--max-carry-lag", "2"]] {
            let mut argv = vec!["yeto-syncer", "--learners", "2", "--total-steps", "8"];
            argv.extend(retired);
            assert!(Args::try_parse_from(argv).is_err(), "{retired:?} is no longer a flag");
        }
    }

    #[test]
    fn fragment_round_membership_flags_parse() {
        let a = Args::try_parse_from([
            "yeto-syncer", "--learners", "2", "--total-steps", "8",
            "--quorum-theta", "0.5", "--q-min", "1", "--island-capacity", "0=8,1=4",
            "--island-lease-s", "30", "--soft-deadline-s", "120", "--carry-gamma", "0.5",
            "--max-learners", "4", "--final-grace-s", "45",
        ])
        .unwrap();
        let m = parse_membership(&a).unwrap();
        assert_eq!(m.quorum_rule, membership::QuorumRule::Capacity { theta: 0.5, q_min: 1 });
        assert_eq!(m.capacity(0), 8.0);
        assert_eq!(m.capacity(3), 1.0);
        assert_eq!(m.lease, Some(std::time::Duration::from_secs(30)));
        assert_eq!(m.soft_deadline, Some(std::time::Duration::from_secs(120)));
        assert_eq!(m.carry_gamma, 0.5);
        assert_eq!(m.id_limit(2), 4);
        assert_eq!(m.final_grace, Some(std::time::Duration::from_secs(45)));
        let dup = Args::try_parse_from([
            "yeto-syncer", "--learners", "2", "--total-steps", "8", "--island-capacity", "0=1,0=2",
        ])
        .unwrap();
        assert!(parse_membership(&dup).is_err());
    }
}
