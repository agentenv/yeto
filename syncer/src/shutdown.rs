//! Graceful stop on SIGTERM / SIGINT.
//!
//! The first signal asks the scheduler to stop: no new round is launched,
//! the rounds already gathering get `--shutdown-timeout-s` to close, every
//! computed round commits, and the consistent cut is written synchronously
//! to `--checkpoint-path` before the process exits 0, so `--resume` starts
//! at the last outer step on the event tape. A second signal exits at once.

use std::time::{Duration, Instant};

use tokio::sync::watch;
use tracing::warn;

/// Exit status of a second signal while the first one is being served.
pub const FORCED_EXIT_CODE: i32 = 143;

pub struct Shutdown {
    tx: watch::Sender<Option<Instant>>,
    timeout: Duration,
}

impl Shutdown {
    pub fn new(timeout: Duration) -> Self {
        Self {
            tx: watch::channel(None).0,
            timeout,
        }
    }

    /// Ask for a stop; returns false if one was already requested.
    pub fn request(&self) -> bool {
        let mut first = false;
        self.tx.send_if_modified(|at| {
            first = at.is_none();
            if first {
                *at = Some(Instant::now());
            }
            first
        });
        first
    }

    pub fn requested_at(&self) -> Option<Instant> {
        *self.tx.borrow()
    }

    /// When the rounds still gathering are abandoned.
    pub fn deadline(&self) -> Option<Instant> {
        self.requested_at().map(|at| at + self.timeout)
    }

    /// Resolves once a stop is requested.
    pub async fn requested(&self) {
        let mut rx = self.tx.subscribe();
        let _ = rx.wait_for(Option::is_some).await;
    }

    /// Route SIGTERM and SIGINT here (inside a tokio runtime).
    #[cfg(unix)]
    pub fn install_signal_handlers(self: &std::sync::Arc<Self>) -> anyhow::Result<()> {
        use tokio::signal::unix::{signal, SignalKind};

        let mut term = signal(SignalKind::terminate())?;
        let mut int = signal(SignalKind::interrupt())?;
        let this = self.clone();
        tokio::spawn(async move {
            loop {
                let name = tokio::select! {
                    _ = term.recv() => "SIGTERM",
                    _ = int.recv() => "SIGINT",
                };
                if this.request() {
                    warn!(
                        signal = name,
                        timeout_s = this.timeout.as_secs(),
                        "stop requested: no new rounds; draining in-flight rounds, then checkpoint and exit \
                         (a second signal exits at once)"
                    );
                } else {
                    warn!(signal = name, "second stop signal: exiting without a checkpoint");
                    std::process::exit(FORCED_EXIT_CODE);
                }
            }
        });
        Ok(())
    }

    #[cfg(not(unix))]
    pub fn install_signal_handlers(self: &std::sync::Arc<Self>) -> anyhow::Result<()> {
        let this = self.clone();
        tokio::spawn(async move {
            while tokio::signal::ctrl_c().await.is_ok() {
                if !this.request() {
                    std::process::exit(FORCED_EXIT_CODE);
                }
            }
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn first_request_wins_and_wakes_waiters() {
        let stop = std::sync::Arc::new(Shutdown::new(Duration::from_secs(5)));
        assert!(stop.requested_at().is_none() && stop.deadline().is_none());
        let waiter = {
            let stop = stop.clone();
            tokio::spawn(async move { stop.requested().await })
        };
        tokio::task::yield_now().await;
        assert!(stop.request());
        let at = stop.requested_at().unwrap();
        assert!(!stop.request());
        assert_eq!(stop.requested_at(), Some(at));
        assert_eq!(stop.deadline(), Some(at + Duration::from_secs(5)));
        tokio::time::timeout(Duration::from_secs(1), waiter).await.unwrap().unwrap();
        // A waiter that starts after the request returns at once.
        tokio::time::timeout(Duration::from_secs(1), stop.requested()).await.unwrap();
    }
}
