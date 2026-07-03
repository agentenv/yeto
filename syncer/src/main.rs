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
    /// Grace window after quorum is reached, in milliseconds.
    #[arg(long, default_value_t = 1000)]
    grace_ms: u64,
    /// Give up waiting for quorum and re-send the pull after this long.
    #[arg(long, default_value_t = 900)]
    quorum_timeout_s: u64,
    /// Total number of outer steps T (each syncs one fragment).
    #[arg(long)]
    total_steps: u64,
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
    /// JSONL event tape (one record per merge).
    #[arg(long)]
    event_tape: Option<std::path::PathBuf>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Args::parse();
    let cfg = server::Config {
        port: args.port,
        learners: args.learners,
        quorum: args.quorum,
        grace_ms: args.grace_ms,
        quorum_timeout_s: args.quorum_timeout_s,
        total_steps: args.total_steps,
        outer_lr: args.outer_lr,
        outer_momentum: args.outer_momentum,
        final_state: args.final_state,
        checkpoint_path: args.checkpoint_path,
        checkpoint_every: args.checkpoint_every,
        resume: args.resume,
        event_tape: args.event_tape,
    };
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(server::run(cfg))
}
