mod merge;
mod protocol;
mod server;
mod state;

use clap::Parser;

/// Yeto syncer: pull-driven fragment merging (weighted RDA/Avg)
/// with a selectable outer optimizer. See docs/PROTOCOL.md.
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
    /// Give up waiting for quorum and re-send the pull after this long.
    #[arg(long, default_value_t = 900)]
    quorum_timeout_s: u64,
    /// Never lower the configured quorum after disconnects and never commit
    /// a partial round when its timeout expires.
    #[arg(long, default_value_t = false)]
    strict_quorum: bool,
    /// Total number of outer steps T (each syncs one fragment).
    #[arg(long)]
    total_steps: u64,
    /// Outer learning rate.
    #[arg(long, default_value_t = 0.7)]
    outer_lr: f32,
    /// Optional comma-separated learning rates, one per fragment. When set,
    /// these replace --outer-lr for the corresponding fragment.
    #[arg(long)]
    outer_lr_by_fragment: Option<String>,
    /// Outer Nesterov momentum, or beta for normalized EMA variants.
    #[arg(long, default_value_t = 0.9)]
    outer_momentum: f32,
    /// Outer optimizer: nesterov, normalized-ema, or restarted-ema.
    #[arg(long, default_value_t = merge::OuterOptimizer::Nesterov)]
    outer_optimizer: merge::OuterOptimizer,
    /// Restart EMA history when cosine(current delta, previous EMA) is at or
    /// below this threshold. Used only by restarted-ema.
    #[arg(long, default_value_t = 0.0, value_parser = parse_cosine_threshold)]
    outer_restart_cos_threshold: f32,
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
    /// JSONL event tape (one record per merge).
    #[arg(long)]
    event_tape: Option<std::path::PathBuf>,
    /// Optional directory for offline syncer-current fragment probes.
    /// When set, the syncer writes one pre-merge checkpoint per sampled
    /// round, one f32 candidate-fragment file per admitted responder, and
    /// an index.jsonl tying them together.
    #[arg(long)]
    probe_capture_dir: Option<std::path::PathBuf>,
    /// Capture every Nth outer step when --probe-capture-dir is set.
    /// 0 disables capture even if the directory is present.
    #[arg(long, default_value_t = 1)]
    probe_capture_every: u64,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Args::parse();
    validate_outer_optimizer(args.outer_optimizer, args.outer_momentum)?;
    let delta_correction = match args.delta_correction.as_str() {
        "heloco" => true,
        "none" => false,
        other => anyhow::bail!("--delta-correction must be 'heloco' or 'none', got {other:?}"),
    };
    let outer_lr_by_fragment = args
        .outer_lr_by_fragment
        .as_deref()
        .map(parse_outer_lr_by_fragment)
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
        strict_quorum: args.strict_quorum,
        total_steps: args.total_steps,
        outer_lr: args.outer_lr,
        outer_lr_by_fragment,
        outer_momentum: args.outer_momentum,
        outer_optimizer: args.outer_optimizer,
        outer_restart_cos_threshold: args.outer_restart_cos_threshold,
        final_state: args.final_state,
        checkpoint_path: args.checkpoint_path,
        checkpoint_every: args.checkpoint_every,
        resume: args.resume,
        event_tape: args.event_tape,
        probe_capture_dir: args.probe_capture_dir,
        probe_capture_every: args.probe_capture_every,
    };
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(server::run(cfg))
}

fn parse_cosine_threshold(value: &str) -> Result<f32, String> {
    let threshold = value
        .parse::<f32>()
        .map_err(|err| format!("invalid cosine threshold {value:?}: {err}"))?;
    if !threshold.is_finite() || !(-1.0..=1.0).contains(&threshold) {
        return Err(format!(
            "cosine threshold must be finite and in [-1, 1], got {value:?}"
        ));
    }
    Ok(threshold)
}

fn validate_outer_optimizer(
    optimizer: merge::OuterOptimizer,
    outer_momentum: f32,
) -> anyhow::Result<()> {
    if optimizer.uses_normalized_ema()
        && (!outer_momentum.is_finite() || !(0.0..1.0).contains(&outer_momentum))
    {
        anyhow::bail!(
            "--outer-momentum is beta for {optimizer} and must be finite and in [0, 1), got {outer_momentum}"
        );
    }
    Ok(())
}

fn parse_outer_lr_by_fragment(spec: &str) -> anyhow::Result<Vec<f32>> {
    let values: Vec<f32> = spec
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<f32>()
                .map_err(|err| anyhow::anyhow!("invalid fragment outer LR {part:?}: {err}"))
        })
        .collect::<anyhow::Result<_>>()?;
    if values.is_empty() {
        anyhow::bail!("--outer-lr-by-fragment must contain at least one value");
    }
    if values.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        anyhow::bail!("--outer-lr-by-fragment values must be finite and > 0");
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fragment_outer_learning_rates() {
        assert_eq!(
            parse_outer_lr_by_fragment("0.2625, 0.175,0.0875,0.14").unwrap(),
            vec![0.2625, 0.175, 0.0875, 0.14]
        );
        assert!(parse_outer_lr_by_fragment("0.2,-0.1").is_err());
        assert!(parse_outer_lr_by_fragment("").is_err());
    }

    #[test]
    fn cli_defaults_to_existing_nesterov_behavior() {
        let args =
            Args::try_parse_from(["yeto-syncer", "--learners", "1", "--total-steps", "1"]).unwrap();
        assert_eq!(args.outer_optimizer, merge::OuterOptimizer::Nesterov);
        assert_eq!(args.outer_momentum, 0.9);
        assert_eq!(args.outer_restart_cos_threshold, 0.0);
    }

    #[test]
    fn parses_outer_optimizer_contract() {
        for (name, expected) in [
            ("nesterov", merge::OuterOptimizer::Nesterov),
            ("normalized-ema", merge::OuterOptimizer::NormalizedEma),
            ("restarted-ema", merge::OuterOptimizer::RestartedEma),
        ] {
            let args = Args::try_parse_from([
                "yeto-syncer",
                "--learners",
                "1",
                "--total-steps",
                "1",
                "--outer-optimizer",
                name,
            ])
            .unwrap();
            assert_eq!(args.outer_optimizer, expected);
        }
    }

    #[test]
    fn validates_ema_beta_and_restart_threshold() {
        assert!(validate_outer_optimizer(merge::OuterOptimizer::NormalizedEma, 0.9).is_ok());
        assert!(validate_outer_optimizer(merge::OuterOptimizer::RestartedEma, 1.0).is_err());
        assert_eq!(parse_cosine_threshold("-0.25").unwrap(), -0.25);
        assert!(parse_cosine_threshold("1.1").is_err());
        assert!(parse_cosine_threshold("NaN").is_err());
    }
}
