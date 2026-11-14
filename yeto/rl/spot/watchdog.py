"""Launch and keep one Nebius rollout island VM alive.

One process owns one island (one sky cluster name at a time). It

* walks a candidate list (GPU kind x Nebius region x on-demand/spot): on-demand
  H100 first (the user's standing choice for the resident island), then every
  spot candidate (H100 and H200, all regions) cheapest first, and on-demand H200
  last if ``--allow-ondemand-h200``;
* never ``stop``s or ``start``s a VM (a restarted spot VM once sat in
  STARTING for 30 min and then failed on capacity): a VM that is stopped,
  preempted, missing, or stuck in STARTING/launch for more than
  ``--launch-timeout-s`` (30 min) is deleted and the next candidate is created;
* only ever deletes clusters whose name matches ``--own-name-re`` (the cluster
  prefix must match it too);
* stops when the manual stop marker exists (an operator who cuts losses
  touches ``--stop-file`` first; the watchdog then leaves the VM as it is and
  exits) or when the spend estimate reaches ``--budget-usd`` (then it deletes
  its VM);
* after the VM has been gone for ``--orphan-sweep-min`` minutes without a new
  one up, deletes the island's boat sandboxes (``boat_labels sweep``);
* appends every event (launch phases, preemption time, cold start, spend) to a
  JSONL log; the spend total survives a watchdog restart (``--state``).

The external calls (sky, nebius, boat sweep, ready probe) sit behind
:class:`Ops`; tests inject a fake. ``--via-provider`` puts the machine calls behind the
rollout :class:`~yeto.launch.nc.sky_provider.SkyProvider` instead (:class:`ProviderOps`);
the candidate list is the shared one of :mod:`yeto.launch.rollout.offers`.

Deprecated for new runs: a sky rollout pool (``"provider": {"kind": "sky", "candidates":
{...}}``, ``yeto rollout scale --provider sky``) runs the same candidate rotation under the
shared pool policy (several islands, fencing, scale, one budget).

    python -m yeto.rl.spot.watchdog --yaml SKY_TASK.yaml --cluster-prefix spot-h100 --run-id RUN \\
        --island spot-0 --budget-usd 120 --env CARD=H100 --env TP=8 --env BOAT_STARTS_PER_HOUR=60 \\
        --secret MODAL_TOKEN_ID --secret MODAL_TOKEN_SECRET --secret BOAT_API_KEY

Environment: YETO_SPOT_SKY (sky CLI, default ``sky``), YETO_SPOT_NEBIUS (nebius CLI,
default ``nebius``), YETO_SPOT_SKY_CONFIG (default ``~/.sky/config.yaml``: the Nebius
project per region), YETO_SPOT_SKY_YAML (default of ``--yaml``), YETO_SPOT_OWN_NAME_RE
(default of ``--own-name-re``), YETO_SPOT_PRICES (default of ``--prices``).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from yeto.launch.rollout.offers import PRICES_8GPU, Candidate, candidates, load_prices  # noqa: F401 - re-exported

REPO = Path(__file__).resolve().parents[3]
SKY = os.path.expanduser(os.environ.get("YETO_SPOT_SKY") or "sky")
NEBIUS = os.path.expanduser(os.environ.get("YETO_SPOT_NEBIUS") or "nebius")
SKY_CONFIG = os.path.expanduser(os.environ.get("YETO_SPOT_SKY_CONFIG") or "~/.sky/config.yaml")

DEFAULT_OWN_NAME = r"spot"  # only ever touch our own clusters (--own-name-re)

SWEEP_BACKOFF_S, SWEEP_BACKOFF_MAX_S = 60.0, 1800.0  # orphan sweep retry after a failure

# Watchdog.try_launch results
UP, FAILED, RECLAIMED, STOPPED = "up", "failed", "reclaimed", "stopped"
MAX_STATE_WRITE_FAILURES = 3  # in a row: exit (a restart would roll spend and seq back)


class LaunchConfigError(RuntimeError):
    """The launcher cannot run at all (no sky, not executable): exit, do not try candidates."""


class LaunchNotStarted(RuntimeError):
    """sky was never run (e.g. fork failed): no cluster exists, nothing to delete or bill."""


class Ops:
    """The real external calls. Every method is synchronous and returns plain data."""

    def __init__(self, *, yaml: str, envs: list[str], secrets: list[str], extra: list[str] | None = None,
                 own_name: str = DEFAULT_OWN_NAME):
        self.yaml, self.envs, self.secrets, self.extra = yaml, envs, secrets, extra or []
        self.own_name = re.compile(own_name)

    def launch(self, cluster: str, cand: Candidate, timeout_s: float) -> tuple[bool, str]:
        cmd = [SKY, "launch", "-c", cluster, self.yaml, "--infra", f"nebius/{cand.region}",
               "--gpus", f"{cand.gpu}:{cand.count}", "--use-spot" if cand.spot else "--no-use-spot", "-d", "-y"]
        for e in self.envs:
            cmd += ["--env", e]
        for s in self.secrets:
            cmd += ["--secret", s]
        cmd += self.extra
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=REPO)
        except subprocess.TimeoutExpired:
            return False, f"launch timeout after {timeout_s:.0f}s"
        except (FileNotFoundError, PermissionError) as error:  # a configuration error, not a launch
            raise LaunchConfigError(f"cannot run {SKY}: {error}") from error
        except OSError as error:  # fork/exec failed: sky never ran, no cluster was made
            raise LaunchNotStarted(f"could not start {SKY}: {error}") from error
        out = p.stdout + p.stderr
        if p.returncode == 0 and "Job submitted" in out:
            return True, ""
        errs = [ln for ln in out.splitlines() if re.search(r"Error|Failed|exhausted|capacity", ln, re.I)]
        tail = errs or out.strip().splitlines() or [f"rc={p.returncode}"]
        return False, tail[-1][:300]

    def preflight(self) -> None:
        """Raise :class:`LaunchConfigError` when sky cannot be run at all."""
        if shutil.which(SKY) is None:
            raise LaunchConfigError(f"sky executable {SKY!r} not found or not executable")

    def ip(self, cluster: str) -> str:
        """The head's IP, "" when sky does not answer (the IP is only logged)."""
        try:
            p = subprocess.run([SKY, "status", "--ip", cluster], capture_output=True, text=True, timeout=120)
        except (subprocess.TimeoutExpired, OSError):
            return ""
        return (p.stdout.strip().splitlines() or [""])[-1]

    def state(self, cluster: str, region: str) -> str | None:
        """The Nebius instance state of the cluster's head (RUNNING, STARTING, STOPPED, ...),
        None if no instance carries the name (deleted / reclaimed)."""
        project = project_ids().get(region)
        if not project:
            return None
        p = subprocess.run([NEBIUS, "compute", "instance", "list", "--parent-id", project, "--format", "json"],
                           capture_output=True, text=True, timeout=120)
        if p.returncode != 0:
            raise RuntimeError(p.stderr.strip()[:200])
        items = json.loads(p.stdout or "{}").get("items", [])
        states = [i["status"].get("state") for i in items if cluster in i["metadata"].get("name", "")]
        return states[0] if states else None

    def down(self, cluster: str) -> bool:
        """True once sky reports the cluster deleted; False on an error or a timeout
        (the VM may still be up and billing)."""
        assert self.own_name.search(cluster), cluster
        try:
            p = subprocess.run([SKY, "down", "-y", "--purge", cluster], capture_output=True, text=True,
                               timeout=900)
        except (subprocess.TimeoutExpired, OSError):
            return False
        return p.returncode == 0

    def sweep(self, run: str, island: str) -> str:
        try:
            p = subprocess.run([sys.executable, "-m", "yeto.rl.harness.harbor.boat_labels", "sweep", "--run", run,
                                "--island", island], capture_output=True, text=True, timeout=600, cwd=REPO,
                               env={**os.environ, "PYTHONPATH": str(REPO)})
        except (subprocess.TimeoutExpired, OSError) as error:  # a failed sweep: retried with backoff
            raise RuntimeError(f"{type(error).__name__}: {error}"[:300]) from error
        return (p.stdout + p.stderr).strip()[-300:]

    def ready(self, cluster: str, cmd: str) -> bool:
        if not cmd:
            return True
        try:
            p = subprocess.run(["ssh", "-o", "ConnectTimeout=15", cluster, cmd], capture_output=True,
                               timeout=120)
        except (subprocess.TimeoutExpired, OSError):
            return False  # a hung ssh is "not ready yet", not a reason to stop watching
        return p.returncode == 0


class ProviderOps:
    """:class:`Ops` over :class:`yeto.launch.nc.sky_provider.SkyProvider` (``--via-provider``):
    the watchdog keeps its loop (candidates, budget, stop marker, orphan sweep, ready
    probe) and the machine calls go through the rollout provider (sky SDK launch / status /
    down, its record and resource ledger, the own-name guard). The rollout pool
    (``yeto rollout scale --provider sky``) uses the same provider."""

    STATES = {"UP": "RUNNING", "INIT": "STARTING", "STOPPED": "STOPPED"}

    def __init__(self, provider: Any, *, own_name: str = DEFAULT_OWN_NAME, sweep: Callable[[str, str], str] | None = None,
                 ready: Callable[[str, str], bool] | None = None):
        self.provider, self.own_name = provider, re.compile(own_name)
        legacy = Ops(yaml="", envs=[], secrets=[], own_name=own_name)
        self._sweep, self._ready = sweep or legacy.sweep, ready or legacy.ready

    def launch(self, cluster: str, cand: Candidate, timeout_s: float) -> tuple[bool, str]:
        assert self.own_name.search(cluster), cluster
        self.provider.envs = {**self.provider.envs, "CARD": cand.gpu}  # CARD follows the candidate's GPU kind
        try:
            self.provider.launch_cluster(cluster, cand.offer(), slot=1)
        except Exception as error:  # noqa: BLE001 - a failed launch: the next candidate
            return False, f"{type(error).__name__}: {error}"[:300]
        return True, ""

    def ip(self, cluster: str) -> str:
        try:
            return self.provider.node(1, cluster).ip
        except Exception:  # noqa: BLE001 - the IP is only logged
            return ""

    def state(self, cluster: str, region: str) -> str | None:
        record = self.provider._record(cluster)  # raises on an API error: not a preemption
        if record is None:
            return None
        from yeto.launch.nc.sky_provider import _status_name  # noqa: PLC0415

        name = _status_name(record.get("status"))
        return self.STATES.get(name, name)

    def down(self, cluster: str) -> bool:
        assert self.own_name.search(cluster), cluster
        try:
            self.provider.run_tickets()  # a cluster of an earlier watchdog process is recorded first
            self.provider.release(1, cluster)
        except Exception:  # noqa: BLE001 - the VM may still run: the watchdog retries
            return False
        return True

    def sweep(self, run: str, island: str) -> str:
        return self._sweep(run, island)

    def ready(self, cluster: str, cmd: str) -> bool:
        return self._ready(cluster, cmd)


def project_ids() -> dict[str, str]:
    try:
        import yaml  # noqa: PLC0415
        cfg = yaml.safe_load(open(SKY_CONFIG))
    except Exception:  # noqa: BLE001
        return {}
    return {r: v["project_id"] for r, v in (cfg.get("nebius", {}).get("region_configs") or {}).items()}


@dataclass
class Config:
    cluster_prefix: str
    run_id: str
    island: str
    budget_usd: float
    log: str
    state: str
    stop_file: str
    launch_timeout_s: float = 1800.0  # an on-demand 8 x H100 took 15 min to come up
    starting_timeout_s: float = 600.0
    ready_timeout_s: float = 3600.0
    ready_cmd: str = ""
    poll_s: float = 60.0
    round_sleep_s: float = 300.0
    orphan_sweep_min: float = 15.0
    max_rounds: int = 0  # 0 = forever
    own_name: str = DEFAULT_OWN_NAME  # regex every cluster name we create or delete matches


@dataclass
class Watchdog:
    cfg: Config
    cands: list[Candidate]
    ops: Any
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    spent: float = 0.0
    seq: int = 0
    cluster: str | None = None
    cand: Candidate | None = None
    billed_at: float = 0.0
    gone_since: float | None = None
    swept: bool = False
    # a failed orphan sweep is retried no earlier than sweep_retry_at (exponential backoff),
    # so a hung sweep cannot hold up every candidate's launch
    sweep_failures: int = 0
    sweep_retry_at: float = 0.0
    starting_since: float | None = None
    # clusters whose delete failed: retried every loop and billed until they are gone
    doomed: dict[str, dict[str, Any]] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    # event-log / state-file writes that failed (e.g. a full disk); the loop goes on
    log_write_errors: int = 0
    state_write_failures: int = 0  # in a row; while > 0 no new launch (spend/seq may not persist)

    def __post_init__(self) -> None:
        if not re.search(self.cfg.own_name, self.cfg.cluster_prefix):
            raise ValueError(f"cluster prefix {self.cfg.cluster_prefix!r} must match {self.cfg.own_name!r}")
        st = Path(self.cfg.state)
        if st.exists():
            d = json.loads(st.read_text())
            self.spent, self.seq = float(d.get("spent_usd", 0.0)), int(d.get("seq", 0))
            if d.get("cluster") and d.get("cand"):
                self.cluster, self.cand = d["cluster"], Candidate(**d["cand"])
                self.billed_at = self.clock()
            for name, cand in (d.get("doomed") or {}).items():
                self.doomed[name] = {"cand": Candidate(**cand), "billed_at": self.clock()}

    # --- bookkeeping
    def event(self, kind: str, **kw: Any) -> None:
        rec = {"t": round(self.clock(), 3), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.clock())),
               "event": kind, "cluster": self.cluster, "spent_usd": round(self.spent, 3), **kw}
        if self.log_write_errors:
            rec["log_write_errors"] = self.log_write_errors
        self.history.append(rec)
        try:
            with open(self.cfg.log, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as error:  # a full disk must not take the VM loop (and its deletes) down
            self._write_failed("event log", error)
        self._save()

    def _save(self) -> None:
        # written aside and renamed: a full disk leaves the previous state, never a truncated one
        tmp = Path(f"{self.cfg.state}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps({
                "spent_usd": self.spent, "seq": self.seq, "cluster": self.cluster,
                "cand": asdict(self.cand) if self.cand else None,
                "doomed": {name: asdict(d["cand"]) for name, d in self.doomed.items()}}))
            os.replace(tmp, self.cfg.state)
        except OSError as error:
            self.state_write_failures += 1
            self._write_failed("state", error)
            tmp.unlink(missing_ok=True)
            return
        self.state_write_failures = 0

    def _write_failed(self, what: str, error: OSError) -> None:
        self.log_write_errors += 1
        if self.log_write_errors == 1 or self.log_write_errors % 100 == 0:
            print(f"[watchdog] {what} write failed ({self.log_write_errors} so far): {error}",
                  file=sys.stderr, flush=True)

    def bill(self) -> None:
        now = self.clock()
        if self.cand is not None and self.billed_at:
            self.spent += self.cand.price_hr * max(0.0, now - self.billed_at) / 3600.0
        self.billed_at = now if self.cand is not None else 0.0
        for d in self.doomed.values():
            self.spent += d["cand"].price_hr * max(0.0, now - d["billed_at"]) / 3600.0
            d["billed_at"] = now

    def stopped_by_hand(self) -> bool:
        return os.path.exists(self.cfg.stop_file)

    def over_budget(self, cand: Candidate | None = None) -> bool:
        nxt = (cand or self.cand)
        step = nxt.price_hr * self.cfg.poll_s / 3600.0 if nxt else 0.0
        step += sum(d["cand"].price_hr for d in self.doomed.values()) * self.cfg.poll_s / 3600.0
        return self.spent + step >= self.cfg.budget_usd

    # --- actions
    def delete(self, reason: str) -> None:
        self.bill()
        c, cand = self.cluster, self.cand
        self.event("delete", reason=reason)
        if c and not self.ops.down(c):
            # the VM may still run: keep billing it and retry the delete every loop
            self.doomed[c] = {"cand": cand, "billed_at": self.clock()}
            self.event("delete_failed", reason=reason, doomed=sorted(self.doomed))
        self.cluster, self.cand, self.billed_at, self.starting_since = None, None, 0.0, None
        self._save()

    def retry_deletes(self) -> None:
        """Delete again every cluster whose delete failed; billed until it succeeds."""
        if not self.doomed:
            return
        self.bill()
        for name in list(self.doomed):
            if self.ops.down(name):
                del self.doomed[name]
                self.event("delete_retry_ok", deleted=name, doomed=sorted(self.doomed))
        self._save()

    def try_launch(self, cand: Candidate) -> str:
        """One launch: :data:`UP` (the VM is up and ready), :data:`FAILED` (deleted),
        :data:`RECLAIMED` (reclaimed during the ready wait and deleted: try the next
        candidate at once) or :data:`STOPPED` (the stop marker appeared; the VM stays)."""
        self.seq += 1
        self.cluster, self.cand = f"{self.cfg.cluster_prefix}-{self.seq}", cand
        t0 = self.clock()
        self.billed_at = t0  # pessimistic: bill from the request
        self.event("launch_start", cand=cand.key, price_hr=cand.price_hr)
        try:
            ok, msg = self.ops.launch(self.cluster, cand, self.cfg.launch_timeout_s)
        except (LaunchConfigError, LaunchNotStarted) as error:
            # sky never ran: no cluster to delete, nothing to bill or keep as doomed
            self.event("launch_fail", cand=cand.key, msg=str(error)[:300], not_started=True)
            self.cluster, self.cand, self.billed_at = None, None, 0.0
            self._save()
            if isinstance(error, LaunchConfigError):
                raise
            return FAILED
        t1 = self.clock()
        if not ok or t1 - t0 > self.cfg.launch_timeout_s:
            self.event("launch_fail", cand=cand.key, launch_s=round(t1 - t0, 1), msg=msg)
            self.delete("launch failed")  # a half-made cluster must not linger
            return FAILED
        ip = self.ops.ip(self.cluster)
        self.event("launch_ok", cand=cand.key, launch_s=round(t1 - t0, 1), ip=ip)
        if self.cfg.ready_cmd:
            while not self.ops.ready(self.cluster, self.cfg.ready_cmd):
                if self.stopped_by_hand():  # the marker also ends a ready wait
                    return STOPPED  # the VM stays; the caller's next stop check exits
                if self.reclaimed_while_waiting():  # reclaimed during cold start
                    return RECLAIMED
                if self.clock() - t1 > self.cfg.ready_timeout_s:
                    self.event("ready_timeout", cand=cand.key)
                    self.delete("sampler not ready")
                    return FAILED
                self.sleep(self.cfg.poll_s)
                self.bill()
            self.event("ready", cand=cand.key, ready_s=round(self.clock() - t1, 1),
                       cold_start_s=round(self.clock() - t0, 1),
                       outage_s=round(t0 - self.gone_since, 1) if self.gone_since else None)
        self.gone_since, self.swept = None, False
        self.sweep_failures, self.sweep_retry_at = 0, 0.0
        return UP

    def reclaimed_while_waiting(self) -> bool:
        """True (and the VM deleted) when the instance left RUNNING/STARTING during the ready wait."""
        try:
            st = self.ops.state(self.cluster, self.cand.region)
        except Exception as e:  # noqa: BLE001 - an API hiccup is not a preemption
            self.event("state_error", err=str(e)[:200])
            return False
        if st in ("RUNNING", "STARTING", "CREATING", "UPDATING"):
            return False
        self.gone_since = self.clock()
        self.event("preempted", state=st, cand=self.cand.key, during="ready_wait")
        self.delete(f"state {st} during ready wait")
        return True

    def maybe_sweep(self) -> None:
        if self.gone_since is None or self.swept or self.clock() < self.sweep_retry_at:
            return
        if self.clock() - self.gone_since >= self.cfg.orphan_sweep_min * 60:
            try:
                out = self.ops.sweep(self.cfg.run_id, self.cfg.island)
            except Exception as e:  # noqa: BLE001 - the VM loop must go on; the sweep is retried later
                delay = min(SWEEP_BACKOFF_MAX_S, SWEEP_BACKOFF_S * 2 ** self.sweep_failures)
                self.sweep_failures += 1
                self.sweep_retry_at = self.clock() + delay
                self.event("orphan_sweep_error", err=f"{type(e).__name__}: {e}"[:200],
                           failures=self.sweep_failures, retry_in_s=delay)
                return
            self.swept, self.sweep_failures, self.sweep_retry_at = True, 0, 0.0
            self.event("orphan_sweep", run=self.cfg.run_id, island=self.cfg.island,
                       gone_s=round(self.clock() - self.gone_since, 1), out=out)

    def acquire(self) -> str | None:
        """One pass over the candidates; returns an exit reason or None (VM up / round over)."""
        for cand in self.cands:
            if self.stopped_by_hand():
                return "manual_stop"
            self.retry_deletes()
            if self.over_budget(cand):
                return "budget"
            self.maybe_sweep()
            if self.state_write_failures:  # spend/seq would not survive a restart: no new VM
                self.event("launch_held", reason="state not writable", failures=self.state_write_failures)
                return None
            if self.try_launch(cand) in (UP, STOPPED):
                return None
            # FAILED or RECLAIMED: the next candidate at once (no round sleep)
        self.event("round_failed", n=len(self.cands))
        return None

    def watch_once(self) -> str | None:
        """One poll of the live VM; deletes it if it is not healthy."""
        self.bill()
        try:
            st = self.ops.state(self.cluster, self.cand.region)
        except Exception as e:  # noqa: BLE001 - an API hiccup is not a preemption
            self.event("state_error", err=str(e)[:200])
            return None
        if st == "RUNNING":
            self.starting_since = None
            return None
        if st in ("STARTING", "CREATING", "UPDATING"):
            since = self.starting_since or self.clock()
            self.starting_since = since
            if self.clock() - since > self.cfg.starting_timeout_s:
                self.gone_since = self.gone_since or since
                self.event("starting_timeout", state=st)
                self.delete("STARTING > timeout")
            return None
        # STOPPED / STOPPING / ERROR / DELETING / gone: preempted (or stopped by someone) -> recreate
        self.starting_since = None
        self.gone_since = self.clock()
        self.event("preempted", state=st, cand=self.cand.key)
        self.delete(f"state {st}")
        return None

    def run(self) -> str:
        self.event("watchdog_start", cands=[c.key for c in self.cands], budget_usd=self.cfg.budget_usd)
        try:
            preflight = getattr(self.ops, "preflight", None)
            if preflight is not None:
                preflight()
            return self._loop()
        except LaunchConfigError as error:
            self.retry_deletes()
            self.event("exit", reason="config_error", error=str(error)[:300], doomed=sorted(self.doomed))
            return "config_error"

    def _state_unwritable(self) -> bool:
        return self.state_write_failures >= MAX_STATE_WRITE_FAILURES

    def _loop(self) -> str:
        rounds = 0
        while True:
            if self.state_write_failures:  # retry every pass: recover, or count toward the exit
                self.bill()
                self._save()
            if self._state_unwritable():
                # a restart would read an old spend and seq (overspend, a reused cluster name,
                # an orphan VM): delete what we run and stop
                if self.cluster:
                    self.delete("state not writable")
                self.retry_deletes()
                self.event("exit", reason="state_unwritable", doomed=sorted(self.doomed))
                return "state_unwritable"
            if self.stopped_by_hand():
                self.bill()
                self.event("exit", reason="manual_stop", note="VM left as is", doomed=sorted(self.doomed))
                return "manual_stop"
            self.retry_deletes()
            if self.cluster is None:
                reason = self.acquire()
                if reason:
                    if reason == "budget" and self.cluster:
                        self.delete("budget")
                    self.retry_deletes()
                    self.event("exit", reason=reason, doomed=sorted(self.doomed))
                    return reason
                if self.cluster is None:
                    rounds += 1
                    if self.cfg.max_rounds and rounds >= self.cfg.max_rounds:
                        self.event("exit", reason="max_rounds", doomed=sorted(self.doomed))
                        return "max_rounds"
                    self.sleep(self.cfg.round_sleep_s)
                    self.maybe_sweep()
                    continue
            self.sleep(self.cfg.poll_s)
            self.watch_once()
            if self.cluster is not None and self.over_budget():
                self.delete("budget")
                self.retry_deletes()
                self.event("exit", reason="budget", doomed=sorted(self.doomed))
                return "budget"


def provider_ops(a: argparse.Namespace, cands: list[Candidate], envs: list[str], *, sky: Any = None) -> ProviderOps:
    """The ``--via-provider`` ops: one SkyProvider slot, this watchdog's cluster prefix."""
    from yeto.launch.nc.sky_provider import SkyProvider  # noqa: PLC0415

    provider = SkyProvider(a.run_id, slots=[1], offers=[c.offer() for c in cands], exact_prefix=a.cluster_prefix,
                           task_yaml=a.yaml, envs=dict(e.split("=", 1) for e in envs),
                           secrets={s: s for s in a.secret}, start_timeout_s=a.starting_timeout_s, sky=sky,
                           record=Path(a.dir) / f"{a.cluster_prefix}.clusters.json")
    return ProviderOps(provider, own_name=a.own_name_re)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cluster-prefix", required=True, help="must match --own-name-re; clusters are PREFIX-<n>")
    p.add_argument("--own-name-re", default=os.environ.get("YETO_SPOT_OWN_NAME_RE") or DEFAULT_OWN_NAME,
                   help="regex our cluster names match; nothing else is ever deleted "
                        "(default $YETO_SPOT_OWN_NAME_RE or %(default)s)")
    p.add_argument("--run-id", required=True, help="boat label run (RUN_ID on the VM)")
    p.add_argument("--island", default="spot-0", help="boat label island")
    p.add_argument("--budget-usd", type=float, required=True)
    p.add_argument("--gpus", default="H100,H200")
    p.add_argument("--gpu-count", type=int, default=8)
    p.add_argument("--regions", default="", help="comma list; default every region with a price")
    p.add_argument("--prices", default=os.environ.get("YETO_SPOT_PRICES") or "",
                   help="JSON file [[gpu, region, on_demand_usd_hr, spot_usd_hr], ...] for 8 GPUs; "
                        "default $YETO_SPOT_PRICES or the built-in table")
    p.add_argument("--spot-only", action="store_true")
    p.add_argument("--allow-ondemand-h200", action="store_true")
    p.add_argument("--yaml", default=os.environ.get("YETO_SPOT_SKY_YAML") or None,
                   help="sky task of the island VM (default $YETO_SPOT_SKY_YAML); required unless --dry-run")
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE passed to sky launch (CARD is set per candidate)")
    p.add_argument("--secret", action="append", default=[])
    p.add_argument("--launch-timeout-s", type=float, default=1800)
    p.add_argument("--starting-timeout-s", type=float, default=600)
    p.add_argument("--ready-cmd", default="", help="ssh <cluster> CMD exit 0 = sampler ready (cold-start end)")
    p.add_argument("--ready-timeout-s", type=float, default=3600)
    p.add_argument("--poll-s", type=float, default=60)
    p.add_argument("--round-sleep-s", type=float, default=300)
    p.add_argument("--orphan-sweep-min", type=float, default=15)
    p.add_argument("--max-rounds", type=int, default=0)
    p.add_argument("--dir", default="/tmp/vm-watchdog")
    p.add_argument("--dry-run", action="store_true", help="print the candidate order and exit")
    p.add_argument("--via-provider", action="store_true",
                   help="machine calls through the rollout SkyProvider (sky SDK, its record and resource "
                        "ledger) instead of the sky / nebius CLIs")
    a = p.parse_args(argv)
    cands = candidates(gpus=a.gpus.split(","), regions=[r for r in a.regions.split(",") if r] or None,
                       count=a.gpu_count, prices=load_prices(a.prices) if a.prices else None,
                       allow_ondemand_h200=a.allow_ondemand_h200, spot_only=a.spot_only)
    if a.dry_run:
        for c in cands:
            print(f"{c.key:32s} ${c.price_hr:.2f}/h")
        return 0
    if not a.yaml:
        p.error("--yaml (or YETO_SPOT_SKY_YAML) is required")
    os.makedirs(a.dir, exist_ok=True)
    cfg = Config(cluster_prefix=a.cluster_prefix, run_id=a.run_id, island=a.island, budget_usd=a.budget_usd,
                 log=f"{a.dir}/{a.cluster_prefix}.jsonl", state=f"{a.dir}/{a.cluster_prefix}.state.json",
                 stop_file=f"{a.dir}/{a.cluster_prefix}.stop", launch_timeout_s=a.launch_timeout_s,
                 starting_timeout_s=a.starting_timeout_s, ready_cmd=a.ready_cmd, ready_timeout_s=a.ready_timeout_s,
                 poll_s=a.poll_s, round_sleep_s=a.round_sleep_s, orphan_sweep_min=a.orphan_sweep_min,
                 max_rounds=a.max_rounds, own_name=a.own_name_re)
    envs = [e for e in a.env if not e.startswith("RUN_ID=")] + [f"RUN_ID={a.run_id}"]

    class _Ops(Ops):  # CARD follows the candidate's GPU kind
        def launch(self, cluster: str, cand: Candidate, timeout_s: float) -> tuple[bool, str]:
            self.envs = [e for e in envs if not e.startswith("CARD=")] + [f"CARD={cand.gpu}"]
            return super().launch(cluster, cand, timeout_s)

    ops = provider_ops(a, cands, envs) if a.via_provider else \
        _Ops(yaml=a.yaml, envs=envs, secrets=a.secret, own_name=a.own_name_re)
    wd = Watchdog(cfg, cands, ops)
    print(f"log {cfg.log}; stop marker: touch {cfg.stop_file}", flush=True)
    reason = wd.run()
    print(reason)
    return 2 if reason in ("config_error", "state_unwritable") else 0


if __name__ == "__main__":
    raise SystemExit(main())
