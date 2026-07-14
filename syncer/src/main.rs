mod action_probe;
mod merge;
mod protocol;
mod server;
mod state;
mod stock_vector_tape;

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
    /// Commit ready pipelined rounds in ascending request-step order. Pulls,
    /// uploads, and quorum gathering remain concurrent; only the already
    /// serialized mutation point waits for the oldest in-flight round. This is
    /// opt-in because completion-order commits are the production default.
    #[arg(long, default_value_t = false)]
    deterministic_commit_order: bool,
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
    /// SCAFFOLD inner controls: none, scaffold_lite, or learner-accumulated
    /// scaffold_full Option II.
    #[arg(long, default_value = "none")]
    inner_control_variate: String,
    /// Cyclically derange full controls across learner identities each round.
    #[arg(long, default_value_t = false)]
    scaffold_control_shuffle: bool,
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
    /// Outer optimizer: nesterov, normalized-ema, restarted-ema,
    /// rho-adaptive, capped-nesterov[-gc|-r], block-rms, block-yogi,
    /// cheb-sgd, pti-sgd, or cplg-sgd.
    /// block-rms/block-yogi are memoryless (beta1=0) per-tensor second-moment
    /// optimizers with a global norm-match back to the plain-SGD step. cheb-sgd
    /// is a memoryless cyclical Chebyshev learning-rate schedule (no buffer).
    #[arg(long, default_value_t = merge::OuterOptimizer::Nesterov)]
    outer_optimizer: merge::OuterOptimizer,
    /// Restart EMA history when cosine(current delta, previous EMA) is at or
    /// below this threshold. Used only by restarted-ema.
    #[arg(long, default_value_t = 0.0, value_parser = parse_cosine_threshold)]
    outer_restart_cos_threshold: f32,
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
    /// Fresh output directory for the narrow exact stock-vector shadow tape.
    /// Requires the three identity arguments and frozen stock-SGD semantics.
    #[arg(long)]
    stock_vector_capture_dir: Option<std::path::PathBuf>,
    /// Stable UUID-like identity copied into every stock-vector row.
    #[arg(long)]
    stock_vector_capture_session: Option<String>,
    /// Optional expected canonical live-layout SHA-256. The producer always
    /// derives the actual digest from HELLO and fails if this expectation is
    /// supplied but differs.
    #[arg(long)]
    stock_vector_capture_layout_sha256: Option<String>,
    /// Lowercase SHA-256 of the immutable run configuration.
    #[arg(long)]
    stock_vector_capture_run_config_sha256: Option<String>,
    /// Exact expected stock-vector row count; must equal --total-steps.
    #[arg(long, default_value_t = 0)]
    stock_vector_capture_expected_records: u64,
    /// Fresh atomic receipt for the exact pre-commit layout and f32 global
    /// state. Used in both matched stock-shadow arms.
    #[arg(long)]
    stock_shadow_initial_state_manifest: Option<std::path::PathBuf>,
    /// Atomic completion/timing receipt paired with
    /// --stock-shadow-initial-state-manifest.
    #[arg(long)]
    stock_shadow_completion_manifest: Option<std::path::PathBuf>,
    /// Optional directory for offline syncer-current fragment probes.
    /// When set, the syncer writes one pre-merge checkpoint per sampled
    /// round, one f32 candidate-fragment file per admitted responder, the
    /// applied update vector, and an index.jsonl tying them together.
    #[arg(long)]
    probe_capture_dir: Option<std::path::PathBuf>,
    /// Capture every Nth outer step when --probe-capture-dir is set.
    /// 0 disables capture even if the directory is present.
    #[arg(long, default_value_t = 1)]
    probe_capture_every: u64,
    /// Opt-in exact audited-push/commit JSONL. Requires audited type-12 pushes
    /// and a fresh output path; absent leaves the legacy path unchanged.
    #[arg(long)]
    response_transcript: Option<std::path::PathBuf>,
    /// Capture-session UUID copied into every --response-transcript record.
    #[arg(long)]
    response_transcript_session: Option<String>,
    /// Commit policy: token_weighted preserves the production baseline;
    /// probe_shadow evaluates A0-A4 but commits A0; probe_loo_v1 commits the
    /// exact sidecar-selected LOO preview; probe_lr_shadow evaluates the
    /// frozen step-scale grid but commits x1; probe_lr_v1 commits the exact
    /// sidecar-selected scaled preview.
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
    let delta_correction = match args.delta_correction.as_str() {
        "heloco" => true,
        "none" => false,
        other => anyhow::bail!("--delta-correction must be 'heloco' or 'none', got {other:?}"),
    };
    let control_variate = match args.inner_control_variate.as_str() {
        "scaffold_lite" => server::ControlVariateMode::ScaffoldLite,
        "scaffold_full" => server::ControlVariateMode::ScaffoldFull,
        "none" => server::ControlVariateMode::None,
        other => anyhow::bail!(
            "--inner-control-variate must be 'none', 'scaffold_lite', or 'scaffold_full', got {other:?}"
        ),
    };
    if control_variate != server::ControlVariateMode::None && !args.version_matched_anchor {
        anyhow::bail!(
            "SCAFFOLD requires --version-matched-anchor so every control uses the exact pushed base version"
        );
    }
    if args.scaffold_control_shuffle && control_variate != server::ControlVariateMode::ScaffoldFull
    {
        anyhow::bail!("--scaffold-control-shuffle requires scaffold_full");
    }
    if control_variate == server::ControlVariateMode::ScaffoldFull
        && (args.quorum != args.learners || !args.strict_quorum)
    {
        anyhow::bail!("scaffold_full is restricted to strict full participation");
    }
    if args.scaffold_control_shuffle && args.learners < 2 {
        anyhow::bail!("--scaffold-control-shuffle requires at least two learners");
    }
    if control_variate == server::ControlVariateMode::ScaffoldFull && args.resume {
        anyhow::bail!("scaffold_full resume is not supported in the narrow correctness regime");
    }
    let outer_lr_by_fragment = args
        .outer_lr_by_fragment
        .as_deref()
        .map(parse_outer_lr_by_fragment)
        .transpose()?;
    validate_cplg_treatment(
        args.outer_optimizer,
        args.outer_lr,
        outer_lr_by_fragment.as_deref(),
    )?;
    let action_probe = action_probe_config(&args)?;
    let stock_vector_capture = stock_vector_capture_config(&args, outer_lr_by_fragment.as_deref())?;
    if args.stock_shadow_initial_state_manifest.is_some()
        != args.stock_shadow_completion_manifest.is_some()
    {
        anyhow::bail!(
            "stock-shadow initial-state and completion manifest paths are required together"
        );
    }
    if args.stock_shadow_initial_state_manifest.is_some() {
        validate_stock_shadow_identity(&args, outer_lr_by_fragment.as_deref())?;
    }
    match (
        args.response_transcript.as_ref(),
        args.response_transcript_session.as_deref(),
    ) {
        (Some(_), Some(session)) if !session.trim().is_empty() => {}
        (Some(_), _) => anyhow::bail!(
            "--response-transcript requires a non-empty --response-transcript-session"
        ),
        (None, Some(_)) => {
            anyhow::bail!("--response-transcript-session requires --response-transcript")
        }
        (None, None) => {}
    }
    let cfg = server::Config {
        port: args.port,
        learners: args.learners,
        quorum: args.quorum,
        grace_ms: args.grace_ms,
        grace_gamma: args.grace_gamma,
        grace_tau: args.grace_tau,
        pipeline: args.pipeline,
        deterministic_commit_order: args.deterministic_commit_order,
        min_round_interval_ms: args.min_round_interval_ms,
        sync_interval_steps: args.sync_interval_steps,
        delta_correction,
        control_variate,
        scaffold_control_shuffle: args.scaffold_control_shuffle,
        quorum_timeout_s: args.quorum_timeout_s,
        strict_quorum: args.strict_quorum,
        total_steps: args.total_steps,
        outer_lr: args.outer_lr,
        outer_lr_by_fragment,
        outer_momentum: args.outer_momentum,
        outer_optimizer: args.outer_optimizer,
        outer_restart_cos_threshold: args.outer_restart_cos_threshold,
        delta_norm_ref: args.delta_norm_ref,
        version_matched_anchor: args.version_matched_anchor,
        anchor_drift_instrument: args.version_matched_anchor || args.anchor_drift_log,
        final_state: args.final_state,
        checkpoint_path: args.checkpoint_path,
        checkpoint_every: args.checkpoint_every,
        resume: args.resume,
        event_tape: args.event_tape,
        stock_vector_capture,
        stock_shadow_initial_state_manifest: args.stock_shadow_initial_state_manifest,
        stock_shadow_completion_manifest: args.stock_shadow_completion_manifest,
        probe_capture_dir: args.probe_capture_dir,
        probe_capture_every: args.probe_capture_every,
        response_transcript: args.response_transcript,
        response_transcript_session: args.response_transcript_session,
        commit_policy: args.commit_policy,
        action_probe,
    };
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(server::run(cfg))
}

fn stock_vector_capture_config(
    args: &Args,
    outer_lr_by_fragment: Option<&[f32]>,
) -> anyhow::Result<Option<stock_vector_tape::StockVectorTapeConfig>> {
    let fields_present = [
        args.stock_vector_capture_dir.is_some(),
        args.stock_vector_capture_session.is_some(),
        args.stock_vector_capture_run_config_sha256.is_some(),
        args.stock_vector_capture_expected_records != 0,
    ];
    if fields_present.iter().all(|present| !present)
        && args.stock_vector_capture_layout_sha256.is_none()
    {
        return Ok(None);
    }
    if !fields_present.iter().all(|present| *present) {
        anyhow::bail!(
            "stock-vector capture requires dir, session, run-config SHA-256, and expected-records together"
        );
    }
    validate_stock_shadow_identity(args, outer_lr_by_fragment)?;
    if args.stock_vector_capture_expected_records != args.total_steps {
        anyhow::bail!("stock-vector capture requires expected-records equal to total-steps");
    }
    Ok(Some(stock_vector_tape::StockVectorTapeConfig {
        root: args.stock_vector_capture_dir.clone().unwrap(),
        capture_session_uuid: args.stock_vector_capture_session.clone().unwrap(),
        expected_layout_sha256: args.stock_vector_capture_layout_sha256.clone(),
        run_config_sha256: args.stock_vector_capture_run_config_sha256.clone().unwrap(),
        expected_records: args.stock_vector_capture_expected_records,
    }))
}

fn validate_stock_shadow_identity(
    args: &Args,
    outer_lr_by_fragment: Option<&[f32]>,
) -> anyhow::Result<()> {
    if args.outer_optimizer != merge::OuterOptimizer::Nesterov
        || args.outer_lr.to_bits() != CPLG_TREATMENT_LR.to_bits()
        || args.outer_momentum.to_bits() != 0x0000_0000
        || outer_lr_by_fragment.is_some()
        || args.delta_norm_ref.to_bits() != 0x0000_0000
    {
        anyhow::bail!(
            "stock shadow requires exact memoryless stock SGD-0.28, positive-zero momentum, no per-fragment LR, and positive-zero delta-norm-ref"
        );
    }
    if !args.deterministic_commit_order
        || args.commit_policy != action_probe::CommitPolicy::TokenWeighted
        || args.resume
    {
        anyhow::bail!(
            "stock shadow requires a fresh non-resumed run with deterministic token-weighted commits"
        );
    }
    Ok(())
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
    if matches!(
        optimizer,
        merge::OuterOptimizer::PtiSgd | merge::OuterOptimizer::CplgSgd
    ) && outer_momentum.to_bits() != 0x0000_0000
    {
        anyhow::bail!(
            "--outer-optimizer {optimizer} requires exact positive-zero --outer-momentum (bits 0x00000000) so its selected direction feeds the frozen stock SGD kernel; got {outer_momentum:?}"
        );
    }
    Ok(())
}

const CPLG_TREATMENT_LR: f32 = f32::from_bits(0x3e8f_5c29); // exact f32 0.28

fn validate_cplg_treatment(
    optimizer: merge::OuterOptimizer,
    outer_lr: f32,
    outer_lr_by_fragment: Option<&[f32]>,
) -> anyhow::Result<()> {
    if optimizer != merge::OuterOptimizer::CplgSgd {
        return Ok(());
    }
    if outer_lr_by_fragment.is_some() {
        anyhow::bail!(
            "--outer-optimizer cplg-sgd forbids --outer-lr-by-fragment so every fragment has the frozen SGD-0.28 treatment identity"
        );
    }
    if outer_lr.to_bits() != CPLG_TREATMENT_LR.to_bits() {
        anyhow::bail!(
            "--outer-optimizer cplg-sgd requires exact f32 --outer-lr 0.28 (bits 0x3e8f5c29); got {outer_lr:?}"
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
        assert_eq!(args.delta_norm_ref, 0.0);
        assert!(!args.deterministic_commit_order);
        assert_eq!(
            args.commit_policy,
            action_probe::CommitPolicy::TokenWeighted
        );
        assert!(action_probe_config(&args).unwrap().is_none());
    }

    #[test]
    fn parses_deterministic_commit_order_as_opt_in() {
        let args = Args::try_parse_from([
            "yeto-syncer",
            "--learners",
            "1",
            "--total-steps",
            "4",
            "--deterministic-commit-order",
        ])
        .unwrap();
        assert!(args.deterministic_commit_order);
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
            ("normalized-ema", merge::OuterOptimizer::NormalizedEma),
            ("restarted-ema", merge::OuterOptimizer::RestartedEma),
            ("capped-nesterov", merge::OuterOptimizer::CappedNesterov),
            (
                "capped-nesterov-gc",
                merge::OuterOptimizer::CappedNesterovGc,
            ),
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
            ("pti-sgd", merge::OuterOptimizer::PtiSgd),
            ("cplg-sgd", merge::OuterOptimizer::CplgSgd),
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
        assert!(validate_outer_optimizer(merge::OuterOptimizer::PtiSgd, 0.0).is_ok());
        assert!(validate_outer_optimizer(merge::OuterOptimizer::PtiSgd, -0.0).is_err());
        assert!(validate_outer_optimizer(merge::OuterOptimizer::PtiSgd, 0.9).is_err());
        assert!(validate_outer_optimizer(merge::OuterOptimizer::CplgSgd, 0.0).is_ok());
        assert!(validate_outer_optimizer(merge::OuterOptimizer::CplgSgd, -0.0).is_err());
        assert!(validate_outer_optimizer(merge::OuterOptimizer::CplgSgd, 0.9).is_err());
        assert!(validate_cplg_treatment(
            merge::OuterOptimizer::CplgSgd,
            f32::from_bits(0x3e8f_5c29),
            None,
        )
        .is_ok());
        assert!(
            validate_cplg_treatment(merge::OuterOptimizer::CplgSgd, 0.280_000_03, None).is_err()
        );
        assert!(validate_cplg_treatment(
            merge::OuterOptimizer::CplgSgd,
            f32::from_bits(0x3e8f_5c29),
            Some(&[f32::from_bits(0x3e8f_5c29)]),
        )
        .is_err());
        assert!(validate_cplg_treatment(merge::OuterOptimizer::PtiSgd, 0.7, None).is_ok());
        assert!(validate_outer_optimizer(merge::OuterOptimizer::CappedNesterovR, 0.9).is_ok());
        assert!(
            validate_outer_optimizer(merge::OuterOptimizer::CappedNesterovR, f32::NAN).is_err()
        );
        assert_eq!(parse_cosine_threshold("-0.25").unwrap(), -0.25);
        assert!(parse_cosine_threshold("1.1").is_err());
        assert!(parse_cosine_threshold("NaN").is_err());
    }

    #[test]
    fn stock_vector_capture_requires_frozen_stock_identity() {
        let parse = || {
            let layout_sha256 = "1".repeat(64);
            let config_sha256 = "2".repeat(64);
            Args::try_parse_from([
                "yeto-syncer",
                "--learners",
                "1",
                "--total-steps",
                "32",
                "--deterministic-commit-order",
                "--outer-optimizer",
                "nesterov",
                "--outer-lr",
                "0.28",
                "--outer-momentum",
                "0",
                "--stock-vector-capture-dir",
                "/tmp/yeto-cplg-stock-test",
                "--stock-vector-capture-session",
                "123e4567-e89b-12d3-a456-426614174000",
                "--stock-vector-capture-layout-sha256",
                &layout_sha256,
                "--stock-vector-capture-run-config-sha256",
                &config_sha256,
                "--stock-vector-capture-expected-records",
                "32",
            ])
            .unwrap()
        };
        let args = parse();
        let config = stock_vector_capture_config(&args, None).unwrap().unwrap();
        assert_eq!(config.expected_records, 32);
        assert_eq!(config.expected_layout_sha256, Some("1".repeat(64)));

        let mut negative_zero = parse();
        negative_zero.outer_momentum = -0.0;
        assert!(stock_vector_capture_config(&negative_zero, None).is_err());
        let mut wrong_lr = parse();
        wrong_lr.outer_lr = 0.280_000_03;
        assert!(stock_vector_capture_config(&wrong_lr, None).is_err());
        let mut resume = parse();
        resume.resume = true;
        assert!(stock_vector_capture_config(&resume, None).is_err());
        let mut missing_identity = parse();
        missing_identity.stock_vector_capture_run_config_sha256 = None;
        assert!(stock_vector_capture_config(&missing_identity, None).is_err());
        let mut derived_layout = parse();
        derived_layout.stock_vector_capture_layout_sha256 = None;
        assert_eq!(
            stock_vector_capture_config(&derived_layout, None)
                .unwrap()
                .unwrap()
                .expected_layout_sha256,
            None
        );
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
