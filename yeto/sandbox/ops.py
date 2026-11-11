"""``yeto sandbox`` operations: prepare, status, validate, teardown (see :mod:`.envs`).

* ``prepare`` (idempotent, per host in parallel): probe the host (bridge
  gateway, docker data-root, published ports); make the agent base image
  present; pull or build each task image (``pull_parallel`` workers, at most
  ``build_parallel`` builds at once), baking the agent runtime when the
  taskset asks; pull prefetch images; start each sidecar (labelled with the
  run) or adopt an existing container of its declared name; health-check;
  render the task directory for the host into the stage directory and sync it
  (``sync_to``). A task that fails does not stop the others; it is recorded
  with its error. Present images and matching sidecars are left alone.
* ``validate``: oracle (must score 1) and nop (must score 0) per prepared task.
* ``teardown``: removes containers labelled ``yeto.sandbox.run=<run>`` only
  (each one's label is checked again before removal); adopted and foreign
  containers are never touched.

Results go to the readiness manifest (:mod:`.ready`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import ready as ready_mod
from .docker import LABEL_RUN, LABEL_SIDECAR, LABEL_SPEC, LABEL_TASK, DockerError, DockerHost
from .envs import EnvsConfig, HostConfig, Sidecar, TaskEnv, all_tasks, runtime_layer

logger = logging.getLogger(__name__)

DockerFactory = Callable[[HostConfig], DockerHost]
TrialFn = Callable[[TaskEnv, HostConfig, Path, str], "tuple[float | None, str | None]"]
PLACEHOLDER = re.compile(r"\{\{\s*yeto\.(sidecar\.([A-Za-z0-9_.-]+?)\.(url|host|port)|host\.bridge_ip)\s*\}\}")
HEALTH_TRIES = 15
HEALTH_SLEEP_S = 2.0


def default_factory(host: HostConfig) -> DockerHost:
    return DockerHost(host.docker_host, ssh=host.ssh)


def container_name(run: str, task: TaskEnv, sidecar: Sidecar) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", f"yeto-{run}-{task.task_id}-{sidecar.name}")[:120]


# ------------------------------------------------------------------ probe


def probe(dk: DockerHost) -> dict[str, Any]:
    info = dk.info()
    return {"docker_host": dk.docker_host, "bridge_ip": dk.bridge_ip(),
            "data_root": info.get("DockerRootDir"), "server_version": info.get("ServerVersion"),
            "rootless": "rootless" in json.dumps(info.get("SecurityOptions") or []), "probed_at": time.time()}


class _Ports:
    def __init__(self, port_range: tuple[int, int], used: set[int]) -> None:
        self.lo, self.hi = port_range
        self.used = set(used)
        self.lock = threading.Lock()

    def take(self, want: int | None = None) -> int:
        with self.lock:
            if want is not None:
                self.used.add(want)
                return want
            for p in range(self.lo, self.hi + 1):
                if p not in self.used:
                    self.used.add(p)
                    return p
        raise DockerError(f"no free port in {self.lo}-{self.hi}")


# ---------------------------------------------------------------- prepare


class _HostPrep:
    def __init__(self, cfg: EnvsConfig, host: HostConfig, dk: DockerHost, previous: Mapping[str, Any] | None,
                 stage: Path, sleep: Callable[[float], None]) -> None:
        self.cfg, self.host, self.dk, self.stage, self.sleep = cfg, host, dk, stage, sleep
        self.previous = previous or {}
        self.builds = threading.Semaphore(cfg.build_parallel)
        self.image_locks: dict[str, threading.Lock] = {}
        self.lock = threading.Lock()
        self.labels = {LABEL_RUN: cfg.run}

    def _image_lock(self, ref: str) -> threading.Lock:
        with self.lock:
            return self.image_locks.setdefault(ref, threading.Lock())

    def ensure_image(self, ref: str, *, context: str | None = None, mode: str = "auto",
                     dockerfile: str | None = None) -> str:
        """The image id of ``ref`` after pulling or building it (once per ref)."""
        with self._image_lock(ref):
            got = self.dk.image_id(ref)
            if got:
                return got
            pull_error = None
            if mode in ("auto", "pull") and dockerfile is None:
                try:
                    self.dk.pull(ref)
                    got = self.dk.image_id(ref)
                    if got:
                        return got
                except DockerError as exc:
                    pull_error = exc
                if mode == "pull" or context is None:
                    raise DockerError(f"pull {ref} failed: {pull_error}")
            if context is None and dockerfile is None:
                raise DockerError(f"no build context for {ref}")
            with self.builds:
                self.dk.build(ref, context, dockerfile=dockerfile, labels=self.labels)
            got = self.dk.image_id(ref)
            if not got:
                raise DockerError(f"build of {ref} left no image")
            return got

    def base(self) -> None:
        for ts in self.cfg.tasksets:
            if ts.agent_base and ts.agent_base.get("image"):
                ctx = ts.agent_base.get("context")
                self.ensure_image(str(ts.agent_base["image"]), context=str(Path(ctx).expanduser()) if ctx else None)

    def task_image(self, task: TaskEnv, mode: str) -> str:
        ctx = str(task.build_context) if task.build_context else None
        source = task.source_image or task.image
        src_id = self.ensure_image(source, context=ctx, mode=mode)
        if not task.bake_runtime:
            return src_id
        return self.ensure_image(task.image, dockerfile=f"FROM {source}\n{runtime_layer(task.bake_runtime)}")

    def sidecar(self, task: TaskEnv, s: Sidecar, ports: _Ports, bridge: str) -> dict[str, Any]:
        name = s.container_name or container_name(self.cfg.run, task, s)
        current = self.dk.inspect(name)
        labels = (current or {}).get("Config", {}).get("Labels") or {}
        running = bool((current or {}).get("State", {}).get("Running"))
        ours = labels.get(LABEL_RUN) == self.cfg.run
        if current is not None and not ours:
            image = (current.get("Config") or {}).get("Image")
            if running and image == s.image:  # adopt: started by someone else, left exactly as it is
                port = _host_port(current, s.port) or s.host_port
                return self._health(s, name, bridge, port, adopted=True)
            raise DockerError(f"container {name} exists and is not this run's (image {image}, running={running}); "
                              "it is left alone: free the name or declare another container_name")
        if current is not None and ours and running and labels.get(LABEL_SPEC) == s.spec_hash():
            return self._health(s, name, bridge, _host_port(current, s.port) or s.host_port, adopted=False)
        if current is not None:  # ours, stale or stopped
            self.dk.remove([current.get("Id") or name])
        self.ensure_image(s.image, context=s.build, mode="auto" if s.build else "pull")
        prev = ((((self.previous.get(task.task_id) or {}).get("hosts") or {}).get(self.host.name) or {})
                .get("sidecars") or {}).get(s.name) or {}
        port = ports.take(s.host_port or prev.get("host_port"))
        self.dk.run_detached(name, s.image, publish=f"{bridge}:{port}:{s.port}", env=s.env,
                             labels={**self.labels, LABEL_TASK: task.task_id, LABEL_SIDECAR: s.name,
                                     LABEL_SPEC: s.spec_hash()})
        return self._health(s, name, bridge, port, adopted=False)

    def _health(self, s: Sidecar, name: str, bridge: str, port: int | None, *, adopted: bool) -> dict[str, Any]:
        out = {"container": name, "host_port": port, "adopted": adopted,
               "url": f"http://{bridge}:{port}" if port else None, "healthy": None}
        if s.health and port:
            for i in range(HEALTH_TRIES):
                ok = self.dk.http_ok(f"http://{bridge}:{port}{s.health}")
                if ok is None:
                    break
                if ok:
                    out["healthy"] = True
                    return out
                if i + 1 < HEALTH_TRIES:
                    self.sleep(HEALTH_SLEEP_S)
            else:
                raise DockerError(f"sidecar {name} unhealthy at http://{bridge}:{port}{s.health}")
        state = (self.dk.inspect(name) or {}).get("State") or {}
        if not state.get("Running"):
            raise DockerError(f"sidecar {name} is not running")
        out["healthy"] = True if out["healthy"] is None and not s.health else out["healthy"]
        return out

    def render(self, task: TaskEnv, facts: Mapping[str, Any], sidecars: Mapping[str, Any]) -> Path:
        dst = self.stage / self.host.name / task.task_id
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(task.dir, dst, symlinks=True)
        bridge = str(facts.get("bridge_ip") or "")

        def sub(m: re.Match[str]) -> str:
            if m.group(1) == "host.bridge_ip":
                return bridge
            sc = sidecars.get(m.group(2))
            if sc is None:
                raise DockerError(f"task {task.task_id} names unknown sidecar {m.group(2)!r}")
            return {"url": str(sc["url"]), "host": bridge, "port": str(sc["host_port"])}[m.group(3)]

        for path in dst.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                text = path.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            new = PLACEHOLDER.sub(sub, text)
            if path == dst / "task.toml" and task.image != task.source_image:
                new = _set_docker_image(new, task.image)
            if new != text:
                path.write_text(new)
        return dst

    def task(self, task: TaskEnv, ts_mode: str, ports: _Ports, facts: Mapping[str, Any]) -> dict[str, Any]:
        entry: dict[str, Any] = {"image": task.image, "prepared": False, "error": None, "sidecars": {}}
        try:
            for ref in task.prefetch:
                self.ensure_image(ref, mode="pull")
            entry["digest"] = self.task_image(task, ts_mode)
            for s in task.sidecars:
                entry["sidecars"][s.name] = self.sidecar(task, s, ports, str(facts.get("bridge_ip") or ""))
            entry["staged"] = str(self.render(task, facts, entry["sidecars"]))
            entry["prepared"] = True
        except Exception as exc:  # noqa: BLE001 - one task's failure is recorded, the rest go on
            entry["error"] = f"{type(exc).__name__}: {exc}"[:2000]
            logger.warning("sandbox prepare %s/%s: %s", self.host.name, task.task_id, entry["error"])
        entry["prepared_at"] = time.time()
        return entry


def _set_docker_image(toml_text: str, image: str) -> str:
    line = f"docker_image = {json.dumps(image)}"
    if re.search(r"(?m)^docker_image\s*=.*$", toml_text):
        return re.sub(r"(?m)^docker_image\s*=.*$", line, toml_text, count=1)
    if re.search(r"(?m)^\[environment\]\s*$", toml_text):
        return re.sub(r"(?m)^\[environment\]\s*$", "[environment]\n" + line, toml_text, count=1)
    return toml_text.rstrip("\n") + f"\n\n[environment]\n{line}\n"


def _host_port(container: Mapping[str, Any], cport: int) -> int | None:
    bindings = ((container.get("HostConfig") or {}).get("PortBindings") or {}).get(f"{cport}/tcp") or []
    for b in bindings:
        if str(b.get("HostPort") or "").isdigit():
            return int(b["HostPort"])
    return None


def _modes(cfg: EnvsConfig) -> dict[str, str]:
    return {ts.name: ts.image_source for ts in cfg.tasksets}


def _hosts(cfg: EnvsConfig, names: Sequence[str] | None) -> list[HostConfig]:
    return [cfg.host(n) for n in names] if names else list(cfg.hosts)


def prepare(cfg: EnvsConfig, objects: Any, *, hosts: Sequence[str] | None = None, ids: Sequence[str] | None = None,
            stage_dir: str | Path, factory: DockerFactory = default_factory,
            sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    tasks = all_tasks(cfg, ids)
    previous = (ready_mod.read_manifest(objects, cfg.ready_key) or {}).get("tasks") or {}
    stage = Path(stage_dir).expanduser()
    modes = _modes(cfg)
    host_facts: dict[str, Any] = {}
    results: dict[str, dict[str, Any]] = {t.task_id: {"portable": t.portable, "taskset": t.taskset, "hosts": {}}
                                          for t in tasks}
    lock = threading.Lock()

    def one_host(host: HostConfig) -> None:
        dk = factory(host)
        try:
            facts = probe(dk)
            used = dk.published_ports()
        except Exception as exc:  # noqa: BLE001 - an unreachable host fails its tasks only
            err = f"probe failed: {type(exc).__name__}: {exc}"[:2000]
            with lock:
                host_facts[host.name] = {"docker_host": host.docker_host, "error": err, "probed_at": time.time()}
                for t in tasks:
                    results[t.task_id]["hosts"][host.name] = {"prepared": False, "error": err}
            return
        prep = _HostPrep(cfg, host, dk, previous, stage, sleep)
        try:
            prep.base()
        except Exception as exc:  # noqa: BLE001
            facts["base_error"] = f"{type(exc).__name__}: {exc}"[:2000]
        ports = _Ports(host.port_range, used)
        with ThreadPoolExecutor(max_workers=cfg.pull_parallel) as pool:
            done = dict(zip([t.task_id for t in tasks],
                            pool.map(lambda t: prep.task(t, modes.get(t.taskset, "auto"), ports, facts), tasks)))
        if host.sync_to and host.ssh:
            rc, _, err = dk.runner(["rsync", "-az", "--delete", f"{stage / host.name}/", f"{host.ssh}:{host.sync_to}/"],
                                   dk.env, None)
            facts["synced"] = rc == 0
            if rc != 0:
                facts["sync_error"] = err.strip()[-500:]
        with lock:
            host_facts[host.name] = facts
            for tid, entry in done.items():
                if "validation" not in entry:
                    old = ((previous.get(tid) or {}).get("hosts") or {}).get(host.name) or {}
                    if old.get("digest") != entry.get("digest"):
                        entry["validation"] = None  # a new image needs a new check
                results[tid]["hosts"][host.name] = entry

    selected = _hosts(cfg, hosts)
    with ThreadPoolExecutor(max_workers=max(1, len(selected))) as pool:
        list(pool.map(one_host, selected))
    ready_mod.merge_write(objects, cfg.ready_key, cfg.run, hosts=host_facts, tasks=results)
    return summarize(results, host_facts)


def summarize(results: Mapping[str, Any], host_facts: Mapping[str, Any]) -> dict[str, Any]:
    per_host: dict[str, dict[str, int]] = {}
    failed = []
    for tid, entry in results.items():
        for h, e in (entry.get("hosts") or {}).items():
            c = per_host.setdefault(h, {"prepared": 0, "failed": 0})
            c["prepared" if e.get("prepared") else "failed"] += 1
            if not e.get("prepared"):
                failed.append({"task_id": tid, "host": h, "error": e.get("error")})
    return {"hosts": per_host, "failed": failed, "host_facts": dict(host_facts),
            "ok": not failed and not any(f.get("error") for f in host_facts.values())}


# ----------------------------------------------------------------- status


def status(cfg: EnvsConfig, objects: Any, *, hosts: Sequence[str] | None = None,
           factory: DockerFactory = default_factory) -> dict[str, Any]:
    tasks = all_tasks(cfg)
    doc = ready_mod.read_manifest(objects, cfg.ready_key) or ready_mod.empty(cfg.run)
    out: dict[str, Any] = {"run": cfg.run, "hosts": {}}
    for host in _hosts(cfg, hosts):
        dk = factory(host)
        row: dict[str, Any] = {"docker_host": host.docker_host, "max_sandboxes": host.max_sandboxes}
        try:
            row["containers_running"] = dk.count_running()
            row["containers_ours"] = len(dk.containers(f"label={LABEL_RUN}={cfg.run}"))
            facts = doc["hosts"].get(host.name) or {}
            root = facts.get("data_root") or dk.info().get("DockerRootDir")
            row["data_root"] = root
            row["disk"] = dk.disk(root) if root else None
            present = sum(1 for t in tasks if dk.image_id(t.image))
            row["images_ready"] = f"{present}/{len(tasks)}"
            sidecars = {}
            for t in tasks:
                entry = ((doc["tasks"].get(t.task_id) or {}).get("hosts") or {}).get(host.name) or {}
                for s in t.sidecars:
                    rec = (entry.get("sidecars") or {}).get(s.name) or {}
                    name = rec.get("container") or s.container_name or container_name(cfg.run, t, s)
                    state = ((dk.inspect(name) or {}).get("State") or {})
                    ok = None
                    if state.get("Running") and s.health and rec.get("url"):
                        ok = dk.http_ok(f"{rec['url']}{s.health}")
                    sidecars[f"{t.task_id}/{s.name}"] = {"running": bool(state.get("Running")), "healthy": ok}
            row["sidecars_running"] = f"{sum(1 for v in sidecars.values() if v['running'])}/{len(sidecars)}"
            row["sidecars_unhealthy"] = sorted(k for k, v in sidecars.items() if not v["running"] or v["healthy"] is False)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        view = ready_mod.ReadyView(doc, host=host.name, require_validation=cfg.require_validation,
                                   affinity=cfg.affinity)
        counts = {"ready": 0, "pending": 0, "failed": 0}
        for t in tasks:
            counts[view.classify(t.task_id)[0]] += 1
        row["dispense"] = counts
        out["hosts"][host.name] = row
    return out


# --------------------------------------------------------------- validate


def harbor_trial(trials_dir: str | Path) -> TrialFn:
    """Oracle / nop trials through yeto's Harbor path on the host's daemon (remote-docker)."""

    def run(task: TaskEnv, host: HostConfig, tasks_dir: Path, agent: str) -> tuple[float | None, str | None]:
        from yeto.rl.agentic import AgentTask  # noqa: PLC0415
        from yeto.rl.harness.harbor import HarborHarness, HarborSettings, build_trial_config  # noqa: PLC0415
        from yeto.rl.harness.harbor.capacity import SandboxLimits  # noqa: PLC0415

        settings = HarborSettings(tasks_dir=tasks_dir, trials_dir=Path(trials_dir).expanduser(),
                                  provider_kwargs={"docker_host": host.docker_host})
        harness = HarborHarness(lambda t, ep: build_trial_config(t, ep, settings),
                                limits=SandboxLimits({"remote-docker": 1}))
        item = AgentTask.from_mapping({"task_id": task.task_id, "agent": agent, "provider": "remote-docker",
                                       "model_name": "none", "wall_clock_s": 3600.0})
        verdict = asyncio.run(harness.run(item, endpoint="http://127.0.0.1:9/v1"))
        if verdict.rewards is None:
            return None, f"{verdict.exception_type}: {verdict.exception_message[-500:]}"
        values = list(verdict.rewards.values())
        return float(verdict.rewards.get("reward", values[0] if values else 0.0)), None

    return run


def validate(cfg: EnvsConfig, objects: Any, *, trial: TrialFn, hosts: Sequence[str] | None = None,
             ids: Sequence[str] | None = None, force: bool = False) -> dict[str, Any]:
    tasks = all_tasks(cfg, ids)
    doc = ready_mod.read_manifest(objects, cfg.ready_key) or ready_mod.empty(cfg.run)
    jobs = []
    for host in _hosts(cfg, hosts):
        for t in tasks:
            entry = ((doc["tasks"].get(t.task_id) or {}).get("hosts") or {}).get(host.name) or {}
            if not entry.get("prepared") or entry.get("error"):
                continue
            if not force and (entry.get("validation") or {}).get("passed") in (True, False):
                continue
            jobs.append((t, host, Path(entry.get("staged") or t.dir).parent))

    def check(job: tuple[TaskEnv, HostConfig, Path]) -> tuple[str, str, dict[str, Any]]:
        t, host, root = job
        res: dict[str, Any] = {"at": time.time()}
        try:
            oracle, err_o = trial(t, host, root, "oracle")
            nop, err_n = trial(t, host, root, "nop")
        except Exception as exc:  # noqa: BLE001
            oracle = nop = None
            err_o, err_n = f"{type(exc).__name__}: {exc}", None
        res.update(oracle=oracle, nop=nop)
        if err_o or err_n or oracle is None or nop is None:
            res.update(passed=None, reason=f"infra: {err_o or err_n}"[:1000])
        elif oracle < 1.0:
            res.update(passed=False, reason=f"oracle_reward={oracle}")
        elif nop > 0.0:
            res.update(passed=False, reason=f"nop_reward={nop}")
        else:
            res.update(passed=True, reason=None)
        return t.task_id, host.name, res

    with ThreadPoolExecutor(max_workers=cfg.validate_parallel) as pool:
        out = list(pool.map(check, jobs))
    updates: dict[str, dict[str, Any]] = {}
    for tid, h, res in out:
        updates.setdefault(tid, {"hosts": {}})["hosts"][h] = {"validation": res}
    doc = ready_mod.merge_write(objects, cfg.ready_key, cfg.run, tasks=updates)
    passed = sorted(tid for tid, e in doc["tasks"].items()
                    if any((v.get("validation") or {}).get("passed") is True for v in (e.get("hosts") or {}).values()))
    return {"checked": len(out), "results": [{"task_id": t, "host": h, **r} for t, h, r in out], "passed": passed}


# --------------------------------------------------------------- teardown


def teardown(cfg: EnvsConfig, objects: Any, *, hosts: Sequence[str] | None = None, images: bool = False,
             factory: DockerFactory = default_factory, dry_run: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {}
    task_updates: dict[str, dict[str, Any]] = {}
    for host in _hosts(cfg, hosts):
        dk = factory(host)
        mine = []
        for c in dk.containers(f"label={LABEL_RUN}={cfg.run}"):
            cid = str(c.get("ID") or "")
            detail = dk.inspect(cid) if cid else None
            labels = ((detail or {}).get("Config") or {}).get("Labels") or {}
            if labels.get(LABEL_RUN) == cfg.run:  # the filter's answer is checked again, per container
                mine.append((cid, labels.get(LABEL_TASK)))
        if not dry_run:
            dk.remove([cid for cid, _ in mine])
        removed_images = []
        if images:
            _, lines, _ = dk.docker("images", "-q", "--filter", f"label={LABEL_RUN}={cfg.run}")
            removed_images = sorted({i for i in lines.split() if i})
            if removed_images and not dry_run:
                dk.docker("rmi", "-f", *removed_images, check=False)
        for _, tid in mine:
            if tid and not dry_run:
                task_updates.setdefault(tid, {"hosts": {}})["hosts"][host.name] = {
                    "prepared": False, "error": "torn down"}
        out[host.name] = {"containers": [cid for cid, _ in mine], "images": removed_images, "dry_run": dry_run}
    if task_updates:
        ready_mod.merge_write(objects, cfg.ready_key, cfg.run, tasks=task_updates)
    return out
