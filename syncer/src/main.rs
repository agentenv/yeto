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
        final_ack_timeout_s: args.resolved_final_ack_timeout_s(),
        total_steps: args.total_steps,
        policy_sweep_fragments: args.policy_sweep_fragments,
        outer_lr: args.outer_lr,
        outer_momentum: args.outer_momentum,
        final_state: args.final_state,
        checkpoint_path: args.checkpoint_path,
        checkpoint_every: args.checkpoint_every,
        resume: args.resume,
        mark_final_checkpoint: args.mark_final_checkpoint,
        learner_budget_steps: args.learner_budget_steps,
        event_tape: args.event_tape,
        max_base_lag: args.max_base_lag,
        learner_weight,
    };
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(server::run(cfg))
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
}
