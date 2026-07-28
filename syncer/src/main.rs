mod action_probe;
mod merge;
mod outer_lr_controller;
mod protocol;
mod rho_telemetry;
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
    /// Outer optimizer: nesterov, heavy-ball, normalized-ema, restarted-ema,
    /// rho-adaptive, capped-nesterov[-gc|-r], block-rms, block-yogi, or cheb-sgd.
    /// block-rms/block-yogi are memoryless (beta1=0) per-tensor second-moment
    /// optimizers with a global norm-match back to the plain-SGD step. cheb-sgd
    /// is a memoryless cyclical Chebyshev learning-rate schedule (no buffer).
    #[arg(long, default_value_t = merge::OuterOptimizer::Nesterov)]
    outer_optimizer: merge::OuterOptimizer,
    /// Restart EMA history when cosine(current delta, previous EMA) is at or
    /// below this threshold. Used only by restarted-ema.
    #[arg(long, default_value_t = 0.0, value_parser = parse_cosine_threshold)]
    outer_restart_cos_threshold: f32,
    /// v3 finite-horizon outer bias correction (Adam-style, opt-in): divide
    /// the applied Nesterov outer step at a fragment's t-th outer commit
    /// (1-indexed) by (1 - mu^(t+1)), so every commit's constant-gradient
    /// multiplier is the steady-state 1/(1-mu) instead of the code-true
    /// finite-horizon (1 - mu^(t+1))/(1 - mu) (lean-mechanism
    /// FiniteHorizonOuter.lean). Off by default = bit-identical to the
    /// pre-flag production path. Requires --outer-optimizer nesterov and
    /// --commit-policy token_weighted.
    #[arg(long, default_value_t = false)]
    outer_bias_correction: bool,
    /// Age-aware outer-LR controller: transient, measured-drift, or oracle.
    /// This is opt-in and mutually exclusive with the legacy
    /// --outer-bias-correction alias.
    #[arg(long, value_enum)]
    outer_lr_controller: Option<outer_lr_controller::ControllerMode>,
    /// Versioned factorial response-surface JSON for measured-drift mode.
    #[arg(long)]
    outer_lr_drift_surface: Option<std::path::PathBuf>,
    /// Optional versioned probe-measured spectral-sketch JSON. Named features
    /// are consumed only when referenced by a nonzero drift-surface term.
    #[arg(long)]
    outer_lr_spectral_sketch: Option<std::path::PathBuf>,
    /// Versioned exact per-fragment scale schedule for oracle mode.
    #[arg(long)]
    outer_lr_oracle_schedule: Option<std::path::PathBuf>,
    /// CTTN dimensionless transverse curvature budget.
    #[arg(long, default_value_t = 0.10, value_parser = parse_cttn_rho)]
    cttn_rho: f32,
    /// CTTN internal damping momentum. This is independent of
    /// --outer-momentum, which configures the fallback/control optimizer.
    #[arg(long, default_value_t = 0.9, value_parser = parse_cttn_mu)]
    cttn_mu: f32,
    /// Number of stratified HVP samples for cttn_shadow_v1. With four
    /// fragments, 32 gives eight fragment-local samples each.
    #[arg(long, default_value_t = 32)]
    cttn_shadow_samples: u32,
    /// Post-merge renormalization for mediation-control experiments
    /// (EXP2.39 norm-matched intervention): after the production merge,
    /// rescale the merged delta of every commit to this L2 norm (per
    /// fragment) before the outer-optimizer step. The tape's gnorm still
    /// reports the pre-rescale merged-delta norm. 0 disables (the default;
    /// byte-identical to the production path).
    #[arg(long, default_value_t = 0.0, value_parser = parse_delta_norm_ref)]
    delta_norm_ref: f32,
    /// EXP2.46 3-arm current-anchor causal control: difference each learner's
    /// delta against the RETAINED global fragment value at the learner's pushed
    /// base_version (version-matched anchoring, arms A/B) instead of the current
    /// global (current-anchor, arm C). Retains the last few global snapshots per
    /// fragment and implies --anchor-drift-log. Default false = byte-identical
    /// current-anchor production path. See docs/ANCHOR_DRIFT_CONTROL.md.
    #[arg(long, default_value_t = false)]
    version_matched_anchor: bool,
    /// EXP2.46: log per-push anchor-drift instrumentation into the event tape
    /// (||anchor_drift||, ||true_local_delta||, their ratio, and cos(drift,
    /// outer momentum)) WITHOUT changing the merge — the current-anchor arm
    /// still reports the drift it injects. Implied by --version-matched-anchor.
    #[arg(long, default_value_t = false)]
    anchor_drift_log: bool,
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
    /// Opt-in JSONL pseudo-gradient telemetry. One record is written for each
    /// committed fragment round with lag-1..4 projected autocorrelations,
    /// exact pseudo-gradient norms, and exact cross-worker cosines. Disabled
    /// unless a path is supplied.
    #[arg(long)]
    rho_telemetry: Option<std::path::PathBuf>,
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
    /// Commit policy: token_weighted preserves the production baseline;
    /// probe_shadow evaluates A0-A4 but commits A0; probe_loo_v1 commits the
    /// exact sidecar-selected LOO preview; probe_lr_shadow evaluates the
    /// frozen step-scale grid but commits x1; probe_lr_v1 commits the exact
    /// sidecar-selected scaled preview; cttn_v1 and cttn_scalar_v1 ask the
    /// sidecar to compute a matrix or scalar curvature-trust direction and
    /// commit it through the dedicated path; cttn_shadow_v1 samples both
    /// directions but always commits exact plain SGD.
    #[arg(long, default_value_t = action_probe::CommitPolicy::TokenWeighted)]
    commit_policy: action_probe::CommitPolicy,
    /// Persistent action-probe sidecar endpoint. Probe policies require a
    /// numeric loopback address such as 127.0.0.1:49321.
    #[arg(long = "action-probe-endpoint", alias = "probe-endpoint")]
    action_probe_endpoint: Option<String>,
    /// Hard end-to-end timeout for one sidecar request, in milliseconds.
    #[arg(
        long = "action-probe-timeout-ms",
        alias = "probe-timeout-ms",
        default_value_t = 30_000
    )]
    action_probe_timeout_ms: u64,
    /// Stable run identity used by the sidecar's exact-retry cache.
    #[arg(long = "action-probe-run-uuid", alias = "probe-run-uuid")]
    action_probe_run_uuid: Option<String>,
    /// JSON file containing the expected anchor/config/layout digests,
    /// fragment tensor names, fragment pattern, and LoRA rank.
    #[arg(long = "action-probe-expected-config", alias = "probe-expected-config")]
    action_probe_expected_config: Option<std::path::PathBuf>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Args::parse();
    validate_outer_optimizer(args.outer_optimizer, args.outer_momentum)?;
    if args.outer_bias_correction && args.outer_lr_controller.is_some() {
        anyhow::bail!("--outer-bias-correction and --outer-lr-controller are mutually exclusive");
    }
    if args.outer_bias_correction {
        if args.outer_optimizer != merge::OuterOptimizer::Nesterov {
            anyhow::bail!(
                "--outer-bias-correction requires --outer-optimizer nesterov (the correction is derived for the code-true Nesterov recursion)"
            );
        }
        if args.commit_policy != action_probe::CommitPolicy::TokenWeighted {
            anyhow::bail!(
                "--outer-bias-correction requires --commit-policy token_weighted (CTTN/probe commit paths bypass the corrected outer step)"
            );
        }
    }
    let outer_lr_controller = match args.outer_lr_controller {
        Some(mode) => {
            if mode.uses_transient_normalization()
                && args.outer_optimizer != merge::OuterOptimizer::Nesterov
            {
                anyhow::bail!(
                    "--outer-lr-controller {} requires --outer-optimizer nesterov",
                    mode.as_str()
                );
            }
            if args.commit_policy != action_probe::CommitPolicy::TokenWeighted {
                anyhow::bail!(
                    "--outer-lr-controller {} requires --commit-policy token_weighted",
                    mode.as_str()
                );
            }
            Some(outer_lr_controller::ControllerConfig::load(
                mode,
                args.outer_lr_drift_surface.as_deref(),
                args.outer_lr_spectral_sketch.as_deref(),
                args.outer_lr_oracle_schedule.as_deref(),
            )?)
        }
        None => {
            if args.outer_lr_drift_surface.is_some()
                || args.outer_lr_spectral_sketch.is_some()
                || args.outer_lr_oracle_schedule.is_some()
            {
                anyhow::bail!("outer-LR controller JSON flags require --outer-lr-controller");
            }
            None
        }
    };
    if args.rho_telemetry.is_some()
        && args.commit_policy != action_probe::CommitPolicy::TokenWeighted
    {
        anyhow::bail!(
            "--rho-telemetry currently requires --commit-policy token_weighted so the recorded pseudo-gradient is exactly the committed production aggregate"
        );
    }
    if args.commit_policy.is_cttn_shadow() {
        if args.outer_optimizer != merge::OuterOptimizer::Nesterov
            || args.outer_momentum.to_bits() != 0.0f32.to_bits()
        {
            anyhow::bail!(
                "cttn_shadow_v1 requires --outer-optimizer nesterov --outer-momentum 0 (ordinary SGD)"
            );
        }
        if args.resume {
            anyhow::bail!("cttn_shadow_v1 does not support --resume because its diagnostic buffer is intentionally not checkpointed");
        }
    }
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
    let action_probe = action_probe_config(&args)?;
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
        outer_bias_correction: args.outer_bias_correction,
        outer_lr_controller,
        cttn_rho: args.cttn_rho,
        cttn_mu: args.cttn_mu,
        cttn_shadow_samples: args.cttn_shadow_samples,
        delta_norm_ref: args.delta_norm_ref,
        version_matched_anchor: args.version_matched_anchor,
        anchor_drift_instrument: args.version_matched_anchor || args.anchor_drift_log,
        final_state: args.final_state,
        checkpoint_path: args.checkpoint_path,
        checkpoint_every: args.checkpoint_every,
        resume: args.resume,
        event_tape: args.event_tape,
        rho_telemetry: args.rho_telemetry,
        probe_capture_dir: args.probe_capture_dir,
        probe_capture_every: args.probe_capture_every,
        commit_policy: args.commit_policy,
        action_probe,
    };
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(server::run(cfg))
}

fn action_probe_config(args: &Args) -> anyhow::Result<Option<action_probe::ClientConfig>> {
    if !args.commit_policy.requires_probe() {
        return Ok(None);
    }
    if args.commit_policy.is_leave_one_out() && args.learners != 4 {
        anyhow::bail!(
            "{} requires exactly four configured learners for A1-A4 leave-one-out actions",
            args.commit_policy
        );
    }
    if args.action_probe_timeout_ms == 0 {
        anyhow::bail!("--action-probe-timeout-ms must be positive");
    }
    let endpoint = args
        .action_probe_endpoint
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("probe policies require --action-probe-endpoint"))?;
    let run_uuid = args
        .action_probe_run_uuid
        .clone()
        .ok_or_else(|| anyhow::anyhow!("probe policies require --action-probe-run-uuid"))?;
    let expected = args
        .action_probe_expected_config
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("probe policies require --action-probe-expected-config"))?;
    Ok(Some(action_probe::ClientConfig::from_expected_file(
        endpoint,
        std::time::Duration::from_millis(args.action_probe_timeout_ms),
        run_uuid,
        expected,
        args.commit_policy.cttn_mode().is_some(),
    )?))
}

fn parse_delta_norm_ref(value: &str) -> Result<f32, String> {
    let reference = value
        .parse::<f32>()
        .map_err(|err| format!("invalid delta norm reference {value:?}: {err}"))?;
    if !reference.is_finite() || reference < 0.0 {
        return Err(format!(
            "delta norm reference must be finite and non-negative, got {value:?}"
        ));
    }
    Ok(reference)
}

fn parse_cttn_rho(value: &str) -> Result<f32, String> {
    let rho = value
        .parse::<f32>()
        .map_err(|err| format!("invalid CTTN rho {value:?}: {err}"))?;
    if !rho.is_finite() || rho < 0.0 {
        return Err(format!(
            "CTTN rho must be finite and non-negative, got {value:?}"
        ));
    }
    Ok(rho)
}

fn parse_cttn_mu(value: &str) -> Result<f32, String> {
    let mu = value
        .parse::<f32>()
        .map_err(|err| format!("invalid CTTN mu {value:?}: {err}"))?;
    if !mu.is_finite() || !(0.0..1.0).contains(&mu) {
        return Err(format!(
            "CTTN mu must be finite and in [0, 1), got {value:?}"
        ));
    }
    Ok(mu)
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
    // rho-adaptive v2 and the capped-nesterov family do not consume
    // --outer-momentum: their constants (mu_star/rho_ref/gain bounds, and
    // mu_max/tau_perp/release beta/gc gain bounds respectively) are
    // compile-time constants in merge.rs. The flag is still validated as
    // finite so a typo does not silently ride along into logs and tapes.
    if matches!(
        optimizer,
        merge::OuterOptimizer::RhoAdaptive
            | merge::OuterOptimizer::CappedNesterov
            | merge::OuterOptimizer::CappedNesterovGc
            | merge::OuterOptimizer::CappedNesterovR
            | merge::OuterOptimizer::CappedNesterovCurv
            | merge::OuterOptimizer::CappedNesterovWsub
            | merge::OuterOptimizer::BlockRms
            | merge::OuterOptimizer::BlockYogi
            | merge::OuterOptimizer::ChebSgd
    ) && !outer_momentum.is_finite()
    {
        anyhow::bail!(
            "--outer-momentum is unused by {optimizer} but must still be finite, got {outer_momentum}"
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
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
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
        assert!(!args.outer_bias_correction);
        assert!(args.outer_lr_controller.is_none());
        assert!(args.outer_lr_drift_surface.is_none());
        assert!(args.outer_lr_spectral_sketch.is_none());
        assert!(args.outer_lr_oracle_schedule.is_none());
        assert_eq!(args.cttn_rho, 0.10);
        assert_eq!(args.cttn_mu, 0.9);
        assert_eq!(args.cttn_shadow_samples, 32);
        assert_eq!(args.delta_norm_ref, 0.0);
        assert!(args.rho_telemetry.is_none());
        assert_eq!(
            args.commit_policy,
            action_probe::CommitPolicy::TokenWeighted
        );
        assert!(action_probe_config(&args).unwrap().is_none());
    }

    #[test]
    fn parses_age_aware_outer_lr_controller_modes_and_json_paths() {
        let transient = Args::try_parse_from([
            "yeto-syncer",
            "--learners",
            "1",
            "--total-steps",
            "4",
            "--outer-lr-controller",
            "transient",
        ])
        .unwrap();
        assert_eq!(
            transient.outer_lr_controller,
            Some(outer_lr_controller::ControllerMode::Transient)
        );

        let drift = Args::try_parse_from([
            "yeto-syncer",
            "--learners",
            "1",
            "--total-steps",
            "4",
            "--outer-lr-controller",
            "measured-drift",
            "--outer-lr-drift-surface",
            "surface.json",
            "--outer-lr-spectral-sketch",
            "sketch.json",
        ])
        .unwrap();
        assert_eq!(
            drift.outer_lr_controller,
            Some(outer_lr_controller::ControllerMode::MeasuredDrift)
        );
        assert_eq!(
            drift.outer_lr_drift_surface.as_deref(),
            Some(std::path::Path::new("surface.json"))
        );

        let oracle = Args::try_parse_from([
            "yeto-syncer",
            "--learners",
            "1",
            "--total-steps",
            "4",
            "--outer-lr-controller",
            "oracle",
            "--outer-lr-oracle-schedule",
            "oracle.json",
        ])
        .unwrap();
        assert_eq!(
            oracle.outer_lr_controller,
            Some(outer_lr_controller::ControllerMode::Oracle)
        );
    }

    #[test]
    fn parses_opt_in_rho_telemetry_path() {
        let args = Args::try_parse_from([
            "yeto-syncer",
            "--learners",
            "4",
            "--total-steps",
            "40",
            "--rho-telemetry",
            "/tmp/rho-telemetry.jsonl",
        ])
        .unwrap();
        assert_eq!(
            args.rho_telemetry.as_deref(),
            Some(std::path::Path::new("/tmp/rho-telemetry.jsonl"))
        );
    }

    #[test]
    fn parses_action_probe_commit_policy_and_flags() {
        let args = Args::try_parse_from([
            "yeto-syncer",
            "--learners",
            "4",
            "--total-steps",
            "1",
            "--commit-policy",
            "probe_shadow",
            "--action-probe-endpoint",
            "127.0.0.1:49321",
            "--action-probe-timeout-ms",
            "2500",
            "--action-probe-run-uuid",
            "run-1",
            "--action-probe-expected-config",
            "/tmp/probe.json",
        ])
        .unwrap();
        assert_eq!(args.commit_policy, action_probe::CommitPolicy::ProbeShadow);
        assert_eq!(args.action_probe_timeout_ms, 2500);
        assert_eq!(
            args.action_probe_expected_config.as_deref(),
            Some(std::path::Path::new("/tmp/probe.json"))
        );

        let scalar = Args::try_parse_from([
            "yeto-syncer",
            "--learners",
            "2",
            "--total-steps",
            "1",
            "--commit-policy",
            "probe_lr_v1",
            "--action-probe-endpoint",
            "127.0.0.1:49321",
            "--action-probe-run-uuid",
            "run-lr",
            "--action-probe-expected-config",
            "/tmp/probe.json",
        ])
        .unwrap();
        assert_eq!(scalar.commit_policy, action_probe::CommitPolicy::ProbeLrV1);
        assert!(!scalar.commit_policy.is_leave_one_out());

        let cttn = Args::try_parse_from([
            "yeto-syncer",
            "--learners",
            "2",
            "--total-steps",
            "1",
            "--commit-policy",
            "cttn_v1",
            "--cttn-rho",
            "0.2",
            "--cttn-mu",
            "0.8",
            "--action-probe-endpoint",
            "127.0.0.1:49321",
            "--action-probe-run-uuid",
            "run-cttn",
            "--action-probe-expected-config",
            "/tmp/probe.json",
        ])
        .unwrap();
        assert_eq!(cttn.commit_policy, action_probe::CommitPolicy::ProbeCttnV1);
        assert_eq!(cttn.cttn_rho, 0.2);
        assert_eq!(cttn.cttn_mu, 0.8);

        let scalar_cttn = Args::try_parse_from([
            "yeto-syncer",
            "--learners",
            "2",
            "--total-steps",
            "1",
            "--commit-policy",
            "cttn_scalar_v1",
            "--action-probe-endpoint",
            "127.0.0.1:49321",
            "--action-probe-run-uuid",
            "run-cttn-scalar",
            "--action-probe-expected-config",
            "/tmp/probe.json",
        ])
        .unwrap();
        assert_eq!(
            scalar_cttn.commit_policy,
            action_probe::CommitPolicy::ProbeCttnScalarV1
        );

        let shadow = Args::try_parse_from([
            "yeto-syncer",
            "--learners",
            "4",
            "--total-steps",
            "40",
            "--outer-momentum",
            "0",
            "--commit-policy",
            "cttn_shadow_v1",
            "--cttn-shadow-samples",
            "32",
            "--action-probe-endpoint",
            "127.0.0.1:49321",
            "--action-probe-run-uuid",
            "run-cttn-shadow",
            "--action-probe-expected-config",
            "/tmp/probe.json",
        ])
        .unwrap();
        assert_eq!(shadow.commit_policy, action_probe::CommitPolicy::CttnShadowV1);
        assert_eq!(shadow.cttn_shadow_samples, 32);
    }

    #[test]
    fn loo_probe_requires_four_learners_and_probe_policies_require_connection_contract() {
        let missing = Args::try_parse_from([
            "yeto-syncer",
            "--learners",
            "4",
            "--total-steps",
            "1",
            "--commit-policy",
            "probe_loo_v1",
        ])
        .unwrap();
        assert!(action_probe_config(&missing)
            .unwrap_err()
            .to_string()
            .contains("endpoint"));

        let wrong_m = Args::try_parse_from([
            "yeto-syncer",
            "--learners",
            "3",
            "--total-steps",
            "1",
            "--commit-policy",
            "probe_shadow",
        ])
        .unwrap();
        assert!(action_probe_config(&wrong_m)
            .unwrap_err()
            .to_string()
            .contains("exactly four"));
    }

    #[test]
    fn parses_outer_optimizer_contract() {
        for (name, expected) in [
            ("nesterov", merge::OuterOptimizer::Nesterov),
            ("heavy-ball", merge::OuterOptimizer::HeavyBall),
            ("normalized-ema", merge::OuterOptimizer::NormalizedEma),
            ("restarted-ema", merge::OuterOptimizer::RestartedEma),
            ("capped-nesterov", merge::OuterOptimizer::CappedNesterov),
            ("capped-nesterov-gc", merge::OuterOptimizer::CappedNesterovGc),
            ("capped-nesterov-r", merge::OuterOptimizer::CappedNesterovR),
            (
                "capped-nesterov-curv",
                merge::OuterOptimizer::CappedNesterovCurv,
            ),
            (
                "capped-nesterov-wsub",
                merge::OuterOptimizer::CappedNesterovWsub,
            ),
            ("cheb-sgd", merge::OuterOptimizer::ChebSgd),
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
        assert!(validate_outer_optimizer(merge::OuterOptimizer::CappedNesterov, 0.9).is_ok());
        assert!(validate_outer_optimizer(merge::OuterOptimizer::CappedNesterov, f32::NAN).is_err());
        assert!(validate_outer_optimizer(merge::OuterOptimizer::CappedNesterovGc, 0.9).is_ok());
        assert!(
            validate_outer_optimizer(merge::OuterOptimizer::CappedNesterovGc, f32::NAN).is_err()
        );
        assert!(validate_outer_optimizer(merge::OuterOptimizer::ChebSgd, 0.9).is_ok());
        assert!(validate_outer_optimizer(merge::OuterOptimizer::ChebSgd, f32::NAN).is_err());
        assert!(validate_outer_optimizer(merge::OuterOptimizer::CappedNesterovR, 0.9).is_ok());
        assert!(
            validate_outer_optimizer(merge::OuterOptimizer::CappedNesterovR, f32::NAN).is_err()
        );
        assert_eq!(parse_cosine_threshold("-0.25").unwrap(), -0.25);
        assert!(parse_cosine_threshold("1.1").is_err());
        assert!(parse_cosine_threshold("NaN").is_err());
    }

    #[test]
    fn parses_and_validates_delta_norm_ref() {
        assert_eq!(parse_delta_norm_ref("0").unwrap(), 0.0);
        assert_eq!(parse_delta_norm_ref("2.869").unwrap(), 2.869);
        assert!(parse_delta_norm_ref("-0.1").is_err());
        assert!(parse_delta_norm_ref("NaN").is_err());
        assert!(parse_delta_norm_ref("inf").is_err());

        let args = Args::try_parse_from([
            "yeto-syncer",
            "--learners",
            "1",
            "--total-steps",
            "1",
            "--delta-norm-ref",
            "3.5",
        ])
        .unwrap();
        assert_eq!(args.delta_norm_ref, 3.5);
        assert!(Args::try_parse_from([
            "yeto-syncer",
            "--learners",
            "1",
            "--total-steps",
            "1",
            "--delta-norm-ref",
            "-1",
        ])
        .is_err());
    }
}
