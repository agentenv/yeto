#!/usr/bin/env python
"""Measure boat sandbox start-up per phase, cold vs pooled (S26 E8).

Each round starts ``--per-round`` task containers at once (TB2 task images,
round-robin over the given tasks), records the seconds of every phase
(create, ready, pull, run, total) and how long until the whole round was up,
then stops them.  ``--mode cold`` creates and deletes a VM per container
(today's behaviour); ``--mode pool`` goes through BoatSandboxPool (VMs are
recycled between rounds).  ``--codex`` instead runs Harbor trials with the
codex agent and ``install_only`` (environment start + Codex install, no
model endpoint), so the Codex install time and failures are measured too.

One JSON line per container (``kind: sandbox``) or trial (``kind: codex``)
and one per round (``kind: round``) goes to ``--out``.

    python scripts/boat_startup_bench.py --tasks-dir ~/work/tb2-data \\
        --mode cold --rounds 3 --per-round 4 --out cold.jsonl fix-git regex-log
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import tomllib
from pathlib import Path
from typing import Any

from yeto.rl.harness.harbor.boat_client import BoatQuota, process_client
from yeto.rl.harness.harbor.boat_pool import BoatSandboxPool
from yeto.rl.harness.harbor.boat_sandbox import BoatSandbox, ContainerSpec, machine_type_for, pinned_codex_cache


def task_spec(tasks_dir: Path, task_id: str) -> ContainerSpec:
    config = tomllib.loads((tasks_dir / task_id / "task.toml").read_text())
    env = config.get("environment", {})
    return ContainerSpec(image=env["docker_image"], workdir=None, cpus=env.get("cpus"),
                         memory_mb=env.get("memory_mb"))


class Bench:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.client = process_client()
        self.quota = BoatQuota(args.max_active, args.starts_per_minute)
        self.sandboxes: list[BoatSandbox] = []  # every VM this run started (billing)
        self.extra_usage: list[float | None] = []  # usage reads of VMs started inside Harbor
        self.pool = BoatSandboxPool(self.client, self.quota, idle_ttl_s=args.idle_ttl_s,
                                    sandbox_factory=self.new_sandbox) if args.mode == "pool" else None
        self.out = args.out.open("a")
        self.cache_kwargs = {"codex_cache": pinned_codex_cache()} if args.codex_cache else {}

    def new_sandbox(self, *args: Any, **kwargs: Any) -> BoatSandbox:
        sandbox = BoatSandbox(*args, **kwargs)
        self.sandboxes.append(sandbox)
        return sandbox

    def emit(self, row: dict[str, Any]) -> None:
        row = {"mode": self.args.mode, "ts": round(time.time(), 3), **row}
        self.out.write(json.dumps(row) + "\n")
        self.out.flush()
        logging.info("ROW %s", json.dumps(row))

    async def one_sandbox(self, round_no: int, slot: int, task_id: str, t0: float) -> BoatSandbox | None:
        spec = task_spec(self.args.tasks_dir, task_id)
        machine = machine_type_for(spec.memory_mb, spec.cpus)
        name = f"e8-r{round_no}-s{slot}"
        started = time.monotonic()
        sandbox = None
        error = None
        codex_version = None
        try:
            if self.pool is not None:
                sandbox = await self.pool.acquire(spec, name=name, machine_type=machine, **self.cache_kwargs)
            else:
                sandbox = self.new_sandbox(self.client, self.quota, name=name, machine_type=machine,
                                           **self.cache_kwargs)
                await sandbox.start(spec)
            check = await sandbox.exec("echo ok", timeout_s=60)
            if check.stdout.strip() != "ok":
                raise RuntimeError(f"exec check failed: {check.stdout!r} {check.stderr!r}")
            if self.args.codex_cache:  # the task's default user must see the pinned build as `codex`
                version = await sandbox.exec("codex --version", timeout_s=60)
                codex_version = (version.stdout or version.stderr).strip()[-200:]
        except Exception as exc:  # measured, not raised: the failure rate is a result
            error = f"{type(exc).__name__}: {str(exc)[-300:]}"
            if sandbox is not None:
                await sandbox.close()
            sandbox = None
        done = time.monotonic()
        self.emit({"kind": "sandbox", "round": round_no, "slot": slot, "task_id": task_id,
                   "machine_type": machine, "ok": error is None, "error": error, "codex_version": codex_version,
                   "start_s": round(done - started, 3), "ready_after_round_start_s": round(done - t0, 3),
                   "reused": bool(sandbox is not None and sandbox.timings.get("create_s") == 0.0),
                   "timings": {k: round(v, 3) for k, v in (sandbox.timings if sandbox else {}).items()}})
        return sandbox

    async def stop(self, sandbox: BoatSandbox) -> None:
        if self.pool is not None:
            await self.pool.release(sandbox)
        else:
            await sandbox.close()

    async def sandbox_rounds(self) -> None:
        tasks = self.args.tasks
        for round_no in range(self.args.rounds):
            t0 = time.monotonic()
            boxes = await asyncio.gather(*(self.one_sandbox(round_no, i, tasks[i % len(tasks)], t0)
                                           for i in range(self.args.per_round)))
            up = time.monotonic() - t0
            ok = [b for b in boxes if b is not None]
            self.emit({"kind": "round", "round": round_no, "per_round": self.args.per_round,
                       "ok": len(ok), "round_up_s": round(up, 3)})
            await asyncio.gather(*(self.stop(b) for b in ok))

    async def codex_trial(self, round_no: int, slot: int, task_id: str) -> None:
        from yeto.rl.agentic import AgentTask
        from yeto.rl.harness.harbor import HarborSettings, build_trial_config
        from yeto.rl.harness.harbor.trial import HarborHarness

        # Provider kwargs must be JSON (Harbor writes them to the trial lock), so the
        # environment finds this run's client and pool as the process ones (see install_process_hooks).
        settings = HarborSettings(tasks_dir=self.args.tasks_dir, trials_dir=self.args.trials_dir,
                                  provider_kwargs={"pool": self.pool is not None,
                                                   "codex_cache": self.args.codex_cache})
        task = AgentTask.from_mapping({"task_id": task_id, "agent": "codex", "provider": "boat",
                                       "model_name": "none", "wall_clock_s": 3600.0})

        def build(t: Any, endpoint: str) -> Any:
            return build_trial_config(t, endpoint, settings).model_copy(update={"install_only": True})

        harness = HarborHarness(build, setup_retries=self.args.setup_retries)
        started = time.monotonic()
        verdict = await harness.run(task, endpoint="http://127.0.0.1:9/v1")
        self.emit({"kind": "codex", "round": round_no, "slot": slot, "task_id": task_id,
                   "ok": verdict.exception_type is None, "exception_type": verdict.exception_type,
                   "exception_message": verdict.exception_message[-500:],
                   "wall_s": round(time.monotonic() - started, 3),
                   "timings_s": verdict.timings_s, "agent_metrics": dict(verdict.agent_metrics)})

    async def codex_rounds(self) -> None:
        tasks = self.args.tasks
        for round_no in range(self.args.rounds):
            t0 = time.monotonic()
            await asyncio.gather(*(self.codex_trial(round_no, i, tasks[i % len(tasks)])
                                   for i in range(self.args.per_round)))
            self.emit({"kind": "round", "round": round_no, "per_round": self.args.per_round,
                       "round_done_s": round(time.monotonic() - t0, 3)})

    def install_process_hooks(self) -> None:
        """Route Harbor-built environments to this run's client (usage reads) and pool."""
        from yeto.rl.harness.harbor import boat_client, boat_env

        boat_client._process_client = _RecordingClient(self)  # noqa: SLF001 - bench-only hook
        if self.pool is not None:
            pool = self.pool
            boat_env.process_pool = lambda: pool  # type: ignore[assignment]

    async def run(self) -> None:
        if self.args.codex:
            self.install_process_hooks()
        try:
            await (self.codex_rounds() if self.args.codex else self.sandbox_rounds())
        finally:
            if self.pool is not None:
                await self.pool.close()
                self.emit({"kind": "pool", "stats": vars(self.pool.stats)})
            await asyncio.gather(*(b.close() for b in self.sandboxes))  # no-op for deleted VMs
            usage = [b.usage_seconds for b in self.sandboxes] + self.extra_usage
            self.emit({"kind": "billing", "vms": len(usage),
                       "billable_s": round(sum(u or 0.0 for u in usage), 1),
                       "vms_without_usage": sum(u is None for u in usage)})
            self.out.close()


class _RecordingClient:
    """The process boat client; keeps the usage reads of the cold codex path (billing)."""

    def __init__(self, bench: Bench) -> None:
        self._bench = bench

    def usage_seconds(self, sandbox_id: str) -> float | None:
        value = self._bench.client.usage_seconds(sandbox_id)
        self._bench.extra_usage.append(value)
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bench.client, name)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tasks", nargs="+")
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--trials-dir", type=Path, default=Path("/tmp/e8_boat_trials"))
    parser.add_argument("--mode", choices=("cold", "pool"), default="cold")
    parser.add_argument("--codex", action="store_true")
    parser.add_argument("--codex-cache", action="store_true", help="mount the pinned Codex build from the VM cache")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--per-round", type=int, default=4)
    parser.add_argument("--max-active", type=int, default=20)
    parser.add_argument("--starts-per-minute", type=int, default=12)
    parser.add_argument("--idle-ttl-s", type=float, default=120.0)
    parser.add_argument("--setup-retries", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    await Bench(args).run()


if __name__ == "__main__":
    asyncio.run(main())
