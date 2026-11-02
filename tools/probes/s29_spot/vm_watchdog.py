"""S29-next P0-c3 / P1-1 / P1-7: launch and keep one Nebius rollout island VM alive.

One process owns one island (one sky cluster name at a time). It

* walks a candidate list (GPU kind x Nebius region x on-demand/spot): on-demand
  H100 first (the user's standing choice for the resident island), then every
  spot candidate (H100 and H200, all regions) cheapest first, and on-demand H200
  last if ``--allow-ondemand-h200``;
* never ``stop``s or ``start``s a VM (S29 D23: a restarted spot VM sat in
  STARTING for 30 min and then failed on capacity): a VM that is stopped,
  preempted, missing, or stuck in STARTING/launch for more than
  ``--launch-timeout-s`` (30 min) is deleted and the next candidate is created;
* stops when the manual stop marker exists (D23: an operator who cuts losses
  touches ``--stop-file`` first; the watchdog then leaves the VM as it is and
  exits) or when the spend estimate reaches ``--budget-usd`` (then it deletes
  its VM);
* after the VM has been gone for ``--orphan-sweep-min`` minutes without a new
  one up, deletes the island's boat sandboxes (P1-7, ``boat_labels sweep``);
* appends every event (launch phases, preemption time, cold start, spend) to a
  JSONL log; the spend total survives a watchdog restart (``--state``).

The external calls (sky, nebius, boat sweep, ready probe) sit behind
:class:`Ops`; tests inject a fake.

    python tools/probes/s29_spot/vm_watchdog.py --cluster-prefix s30r-h100 --run-id s30r1 \\
        --island spot-0 --budget-usd 120 --env CARD=H100 --env TP=8 --env BOAT_STARTS_PER_HOUR=60 \\
        --secret BOAT_API_KEY --secret MODAL_TOKEN_ID --secret MODAL_TOKEN_SECRET
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SKY = os.path.expanduser("~/work/venvs/skypilot/bin/sky")
NEBIUS = os.path.expanduser("~/.nebius/bin/nebius")
SKY_CONFIG = os.path.expanduser("~/.sky/config.yaml")
TENANT = "tenant-e00fey51fzn151mwsb"

# 8-GPU hourly prices (USD) from `sky gpus list H100:8|H200:8 --infra nebius --all-regions`
# on 2026-11-02. H200 is NOT the same price as H100 (spot H100 eu-north1 $6.32, cheapest
# spot H200 $11.20). A region missing here has no such preset in the sky catalog; Nebius
# quotas: compute.instance.gpu.h100 only in eu-north1, gpu.h200 in eu-north1/eu-west1/us-central1.
PRICES_8GPU: dict[tuple[str, str], tuple[float, float]] = {  # (gpu, region) -> (on-demand, spot)
    ("H100", "eu-north1"): (36.00, 6.32),
    ("H200", "eu-west1"): (43.20, 11.20),
    ("H200", "eu-north1"): (43.20, 16.00),
    ("H200", "us-central1"): (43.20, 39.92),
}
OWN_NAME = re.compile(r"s29|s30")  # only ever touch our own clusters


@dataclass(frozen=True)
class Candidate:
    gpu: str
    region: str
    spot: bool
    price_hr: float  # for the whole VM (``count`` GPUs)
    count: int = 8

    @property
    def key(self) -> str:
        return f"{self.gpu}:{self.count}@{self.region}/{'spot' if self.spot else 'od'}"


def candidates(*, gpus: Iterable[str] = ("H100", "H200"), regions: Iterable[str] | None = None,
               count: int = 8, prices: dict[tuple[str, str], tuple[float, float]] | None = None,
               allow_ondemand_h200: bool = False, spot_only: bool = False,
               availability: dict[tuple[str, str], int] | None = None) -> list[Candidate]:
    """On-demand H100 first, then spot cheapest first (better availability first
    at equal price), then on-demand H200 if allowed."""
    prices = PRICES_8GPU if prices is None else prices
    gpus = [g.upper() for g in gpus]
    regs = set(regions) if regions is not None else None
    scale = count / 8.0
    avail = availability or {}
    pool = [(g, r, od * scale, sp * scale) for (g, r), (od, sp) in prices.items()
            if g in gpus and (regs is None or r in regs)]
    out: list[Candidate] = []
    if not spot_only:
        out += [Candidate(g, r, False, od, count) for g, r, od, _ in sorted(pool, key=lambda x: x[2]) if g == "H100"]
    spot = sorted(pool, key=lambda x: (x[3], -avail.get((x[0], x[1]), 0)))
    out += [Candidate(g, r, True, sp, count) for g, r, _, sp in spot]
    if allow_ondemand_h200 and not spot_only:
        out += [Candidate(g, r, False, od, count) for g, r, od, _ in sorted(pool, key=lambda x: x[2]) if g == "H200"]
    return out


class Ops:
    """The real external calls. Every method is synchronous and returns plain data."""

    def __init__(self, *, yaml: str, envs: list[str], secrets: list[str], extra: list[str] | None = None):
        self.yaml, self.envs, self.secrets, self.extra = yaml, envs, secrets, extra or []

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
        out = p.stdout + p.stderr
        if p.returncode == 0 and "Job submitted" in out:
            return True, ""
        errs = [ln for ln in out.splitlines() if re.search(r"Error|Failed|exhausted|capacity", ln, re.I)]
        tail = errs or out.strip().splitlines() or [f"rc={p.returncode}"]
        return False, tail[-1][:300]

    def ip(self, cluster: str) -> str:
        p = subprocess.run([SKY, "status", "--ip", cluster], capture_output=True, text=True, timeout=120)
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

    def down(self, cluster: str) -> None:
        assert OWN_NAME.search(cluster), cluster
        subprocess.run([SKY, "down", "-y", "--purge", cluster], capture_output=True, text=True, timeout=900)

    def sweep(self, run: str, island: str) -> str:
        p = subprocess.run([sys.executable, "-m", "yeto.rl.harness.harbor.boat_labels", "sweep", "--run", run,
                            "--island", island], capture_output=True, text=True, timeout=600, cwd=REPO,
                           env={**os.environ, "PYTHONPATH": str(REPO)})
        return (p.stdout + p.stderr).strip()[-300:]

    def ready(self, cluster: str, cmd: str) -> bool:
        if not cmd:
            return True
        p = subprocess.run(["ssh", "-o", "ConnectTimeout=15", cluster, cmd], capture_output=True, timeout=120)
        return p.returncode == 0


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
    launch_timeout_s: float = 1800.0  # S30 D33: an on-demand 8 x H100 took 15 min to come up
    starting_timeout_s: float = 600.0
    ready_timeout_s: float = 3600.0
    ready_cmd: str = ""
    poll_s: float = 60.0
    round_sleep_s: float = 300.0
    orphan_sweep_min: float = 15.0
    max_rounds: int = 0  # 0 = forever


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
    starting_since: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not OWN_NAME.search(self.cfg.cluster_prefix):
            raise ValueError(f"cluster prefix {self.cfg.cluster_prefix!r} must contain s29 or s30")
        st = Path(self.cfg.state)
        if st.exists():
            d = json.loads(st.read_text())
            self.spent, self.seq = float(d.get("spent_usd", 0.0)), int(d.get("seq", 0))
            if d.get("cluster") and d.get("cand"):
                self.cluster, self.cand = d["cluster"], Candidate(**d["cand"])
                self.billed_at = self.clock()

    # --- bookkeeping
    def event(self, kind: str, **kw: Any) -> None:
        rec = {"t": round(self.clock(), 3), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.clock())),
               "event": kind, "cluster": self.cluster, "spent_usd": round(self.spent, 3), **kw}
        self.history.append(rec)
        with open(self.cfg.log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self._save()

    def _save(self) -> None:
        Path(self.cfg.state).write_text(json.dumps({
            "spent_usd": self.spent, "seq": self.seq, "cluster": self.cluster,
            "cand": asdict(self.cand) if self.cand else None}))

    def bill(self) -> None:
        now = self.clock()
        if self.cand is not None and self.billed_at:
            self.spent += self.cand.price_hr * max(0.0, now - self.billed_at) / 3600.0
        self.billed_at = now if self.cand is not None else 0.0

    def stopped_by_hand(self) -> bool:
        return os.path.exists(self.cfg.stop_file)

    def over_budget(self, cand: Candidate | None = None) -> bool:
        nxt = (cand or self.cand)
        step = nxt.price_hr * self.cfg.poll_s / 3600.0 if nxt else 0.0
        return self.spent + step >= self.cfg.budget_usd

    # --- actions
    def delete(self, reason: str) -> None:
        self.bill()
        c = self.cluster
        self.event("delete", reason=reason)
        if c:
            self.ops.down(c)
        self.cluster, self.cand, self.billed_at, self.starting_since = None, None, 0.0, None
        self._save()

    def try_launch(self, cand: Candidate) -> bool:
        self.seq += 1
        self.cluster, self.cand = f"{self.cfg.cluster_prefix}-{self.seq}", cand
        t0 = self.clock()
        self.billed_at = t0  # pessimistic: bill from the request
        self.event("launch_start", cand=cand.key, price_hr=cand.price_hr)
        ok, msg = self.ops.launch(self.cluster, cand, self.cfg.launch_timeout_s)
        t1 = self.clock()
        if not ok or t1 - t0 > self.cfg.launch_timeout_s:
            self.event("launch_fail", cand=cand.key, launch_s=round(t1 - t0, 1), msg=msg)
            self.delete("launch failed")  # a half-made cluster must not linger
            return False
        ip = self.ops.ip(self.cluster)
        self.event("launch_ok", cand=cand.key, launch_s=round(t1 - t0, 1), ip=ip)
        if self.cfg.ready_cmd:
            while not self.ops.ready(self.cluster, self.cfg.ready_cmd):
                if self.stopped_by_hand():  # S30 D35: the marker also ends a ready wait
                    return True  # the VM stays; the caller's next stop check exits
                if self.clock() - t1 > self.cfg.ready_timeout_s:
                    self.event("ready_timeout", cand=cand.key)
                    self.delete("sampler not ready")
                    return False
                self.sleep(self.cfg.poll_s)
                self.bill()
            self.event("ready", cand=cand.key, ready_s=round(self.clock() - t1, 1),
                       cold_start_s=round(self.clock() - t0, 1),
                       outage_s=round(t0 - self.gone_since, 1) if self.gone_since else None)
        self.gone_since, self.swept = None, False
        return True

    def maybe_sweep(self) -> None:
        if self.gone_since is None or self.swept:
            return
        if self.clock() - self.gone_since >= self.cfg.orphan_sweep_min * 60:
            out = self.ops.sweep(self.cfg.run_id, self.cfg.island)
            self.swept = True
            self.event("orphan_sweep", run=self.cfg.run_id, island=self.cfg.island,
                       gone_s=round(self.clock() - self.gone_since, 1), out=out)

    def acquire(self) -> str | None:
        """One pass over the candidates; returns an exit reason or None (VM up / round over)."""
        for cand in self.cands:
            if self.stopped_by_hand():
                return "manual_stop"
            if self.over_budget(cand):
                return "budget"
            self.maybe_sweep()
            if self.try_launch(cand):
                return None
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
        rounds = 0
        self.event("watchdog_start", cands=[c.key for c in self.cands], budget_usd=self.cfg.budget_usd)
        while True:
            if self.stopped_by_hand():
                self.bill()
                self.event("exit", reason="manual_stop", note="VM left as is")
                return "manual_stop"
            if self.cluster is None:
                reason = self.acquire()
                if reason:
                    if reason == "budget" and self.cluster:
                        self.delete("budget")
                    self.event("exit", reason=reason)
                    return reason
                if self.cluster is None:
                    rounds += 1
                    if self.cfg.max_rounds and rounds >= self.cfg.max_rounds:
                        self.event("exit", reason="max_rounds")
                        return "max_rounds"
                    self.sleep(self.cfg.round_sleep_s)
                    self.maybe_sweep()
                    continue
            self.sleep(self.cfg.poll_s)
            self.watch_once()
            if self.cluster is not None and self.over_budget():
                self.delete("budget")
                self.event("exit", reason="budget")
                return "budget"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cluster-prefix", required=True, help="must contain s29/s30; clusters are PREFIX-<n>")
    p.add_argument("--run-id", required=True, help="boat label run (RUN_ID on the VM)")
    p.add_argument("--island", default="spot-0", help="boat label island (vm120_run.sh: spot-0)")
    p.add_argument("--budget-usd", type=float, required=True)
    p.add_argument("--gpus", default="H100,H200")
    p.add_argument("--gpu-count", type=int, default=8)
    p.add_argument("--regions", default="", help="comma list; default every region with a price")
    p.add_argument("--spot-only", action="store_true")
    p.add_argument("--allow-ondemand-h200", action="store_true")
    p.add_argument("--yaml", default=str(HERE / "sky_spot120.yaml"))
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE passed to sky launch (CARD is set per candidate)")
    p.add_argument("--secret", action="append", default=[])
    p.add_argument("--launch-timeout-s", type=float, default=1800)  # S30 D33
    p.add_argument("--starting-timeout-s", type=float, default=600)
    p.add_argument("--ready-cmd", default="", help="ssh <cluster> CMD exit 0 = sampler ready (cold-start end)")
    p.add_argument("--ready-timeout-s", type=float, default=3600)
    p.add_argument("--poll-s", type=float, default=60)
    p.add_argument("--round-sleep-s", type=float, default=300)
    p.add_argument("--orphan-sweep-min", type=float, default=15)
    p.add_argument("--max-rounds", type=int, default=0)
    p.add_argument("--dir", default="/tmp/s30wd")
    p.add_argument("--dry-run", action="store_true", help="print the candidate order and exit")
    a = p.parse_args(argv)
    cands = candidates(gpus=a.gpus.split(","), regions=[r for r in a.regions.split(",") if r] or None,
                       count=a.gpu_count, allow_ondemand_h200=a.allow_ondemand_h200, spot_only=a.spot_only)
    if a.dry_run:
        for c in cands:
            print(f"{c.key:32s} ${c.price_hr:.2f}/h")
        return 0
    os.makedirs(a.dir, exist_ok=True)
    cfg = Config(cluster_prefix=a.cluster_prefix, run_id=a.run_id, island=a.island, budget_usd=a.budget_usd,
                 log=f"{a.dir}/{a.cluster_prefix}.jsonl", state=f"{a.dir}/{a.cluster_prefix}.state.json",
                 stop_file=f"{a.dir}/{a.cluster_prefix}.stop", launch_timeout_s=a.launch_timeout_s,
                 starting_timeout_s=a.starting_timeout_s, ready_cmd=a.ready_cmd, ready_timeout_s=a.ready_timeout_s,
                 poll_s=a.poll_s, round_sleep_s=a.round_sleep_s, orphan_sweep_min=a.orphan_sweep_min,
                 max_rounds=a.max_rounds)
    envs = [e for e in a.env if not e.startswith("RUN_ID=")] + [f"RUN_ID={a.run_id}"]

    class _Ops(Ops):  # CARD follows the candidate's GPU kind
        def launch(self, cluster: str, cand: Candidate, timeout_s: float) -> tuple[bool, str]:
            self.envs = [e for e in envs if not e.startswith("CARD=")] + [f"CARD={cand.gpu}"]
            return super().launch(cluster, cand, timeout_s)

    wd = Watchdog(cfg, cands, _Ops(yaml=a.yaml, envs=envs, secrets=a.secret))
    print(f"log {cfg.log}; stop marker: touch {cfg.stop_file}", flush=True)
    print(wd.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
