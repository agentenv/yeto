"""Modal islands: container config, tape pulls, app-stop checks and the
container-set poller.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from ..gpu_spec import ClusterSpec
from . import REPO_ROOT
from .code_pack import code_workdir
from .remote_setup import HF_TOKEN_PATH
from .syncer import critic_syncer_address, island_hmac_secret
from .credentials import SECRET_ENV_NAMES, _secret_value, registry_login_for
from .storage import (
    MODAL_CHECKPOINT_STORE_MOUNT, MODEL_STORE_MOUNT, _rl_checkpoint_mount,
    _rl_checkpoint_storage_name, droppable_rejoins_in_new_container, modal_checkpoint_store,
    resumes_in_new_container,
)
from .codex import CODEX_CONTAINER_DIR, modal_sandbox_secret_name
from .fleet import TEARDOWN_UNVERIFIED_EXIT


def verl_needs_nccl_shm_off(args, spec: ClusterSpec) -> bool:
    """A verl island with several cards in one Modal container runs NCCL with
    its shared-memory transport off.

    S26 E9 e9-g2-a: on a 2-card Modal container verl's NCCL checkpoint engine
    failed attaching ``/dev/shm/nccl-*`` ("No such file or directory", NCCL
    system error); the same run with ``NCCL_SHM_DISABLE=1`` passed (e9-g2-d/e).
    NCCL then uses P2P/sockets between the two local ranks. Miles islands are
    not changed (no failure seen there)."""
    backend = getattr(args, "rl_backend", None) or "miles"
    return backend == "verl" and int(spec.gpus_per_node or 1) > 1


def build_modal_island_config(args, spec: ClusterSpec, learner_id: int, task, syncer_addr: str):
    """Turn the sky task the factory built for this island into the Modal
    island config: same run script, same envs (syncer address swapped for
    the Modal-reachable one), image/volume per training mode."""
    from ..modal_runner import (
        ModalIslandConfig,
        image_ref_from_rl_image,
        modal_app_name,
    )

    rl = getattr(args, "training_mode", "sft") == "rl"
    registry_login = registry_login_for(args) if rl else None
    envs = dict(getattr(task, "envs", None) or {})
    # Modal ships cfg.envs as a modal.Secret, so the task's secret envs
    # (split_secret_envs) join them here; registry logins stay out.
    # sky.Task wraps secret values in pydantic SecretStr (str() is "**********"):
    # unwrap, or the island gets the mask (s19-compaction-g1-20261010a: TBENCH
    # reward HMAC key "outside its size bound").
    envs.update({k: _secret_value(v) for k, v in (getattr(task, "secrets", None) or {}).items()
                 if k in SECRET_ENV_NAMES})
    envs["SYNCER_ADDR"] = syncer_addr
    if "CRITIC_SYNCER_ADDR" in envs and syncer_addr != "none":
        # The critic syncer shares the actor syncer's host: swap it too, or a
        # Modal island behind --syncer-public-addr dials the private address.
        envs["CRITIC_SYNCER_ADDR"] = critic_syncer_address(syncer_addr)
    if rl and verl_needs_nccl_shm_off(args, spec):
        envs.setdefault("NCCL_SHM_DISABLE", "1")  # --modal-env below still overrides it
    for item in getattr(args, "modal_env", None) or []:
        key, sep, value = str(item).partition("=")
        if not sep or not key:
            raise ValueError(f"--modal-env takes KEY=VALUE, got {item!r}")
        envs[key] = value
    if rl:  # 0.20: Modal ships cfg.envs as a modal.Secret
        envs.update(island_hmac_secret(args))
    token_path = os.path.expanduser(HF_TOKEN_PATH)
    if "HF_TOKEN" not in envs and os.path.isfile(token_path):
        with open(token_path, encoding="utf-8") as f:
            envs["HF_TOKEN"] = f.read().strip()
    store_volume = modal_checkpoint_store(args) if rl else None
    if store_volume is not None:
        from ..cloud.modal_ckpt_store import MODAL_VOLUME_ENV, STORE_IMPL
        from ..rl.engine.resume import STORE_IMPL_ENV

        envs[MODAL_VOLUME_ENV] = store_volume[0]  # the learner commits this Volume
        envs[STORE_IMPL_ENV] = STORE_IMPL  # the island builds the Modal store from yeto.cloud
    volume_name = volume_mount = None
    from ..cloud.droppable import island_keeps_spot_volume

    if rl and island_keeps_spot_volume(args):
        volume_name = _rl_checkpoint_storage_name(args.cluster_prefix, learner_id)
        volume_mount = _rl_checkpoint_mount(args.rl_completed_groups_path).replace("~", "/root", 1)
    # Codex run bundle: the sky task mounts it at CODEX_CONTAINER_DIR; Modal
    # mounts the same directory through the image (add_local_dir).
    all_mounts = dict(getattr(task, "file_mounts", None) or {})
    codex_dir = all_mounts.get(CODEX_CONTAINER_DIR)
    # Every other sky file_mount (local prompt file, initial adapter, HF token
    # file) is mounted into the container the same way; `~` is /root in Modal.
    extra_mounts = {
        ("/root" + target[1:] if target.startswith("~") else target): str(source)
        for target, source in all_mounts.items()
        if target != CODEX_CONTAINER_DIR and not str(source).startswith(("s3://", "gs://", "r2://"))
    }
    requirements: tuple[str, ...] = ()
    if not rl:
        req_file = REPO_ROOT / "requirements.txt"
        requirements = ("torch",) + tuple(
            line.strip()
            for line in req_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        )
    return ModalIslandConfig(
        app_name=modal_app_name(args.cluster_prefix),
        learner_id=learner_id,
        training_mode="rl" if rl else "sft",
        gpu=spec.gpu,
        gpus_per_node=spec.gpus_per_node,
        num_nodes=spec.num_nodes,
        run_script=str(getattr(task, "run", "") or ""),
        envs={k: str(v) for k, v in envs.items() if v is not None},
        region=spec.region,
        gpu_exact=bool(getattr(args, "modal_gpu_exact", False)),
        registry_login=bool(registry_login),
        registry_creds=registry_login,
        image_ref=image_ref_from_rl_image(args.rl_image) if rl else None,
        setup_script=str(getattr(task, "setup", "") or "") if rl else None,
        pip_requirements=requirements,
        volume_name=volume_name,
        volume_mount=volume_mount,
        codex_dir=codex_dir,
        codex_mount=CODEX_CONTAINER_DIR if codex_dir else None,
        extra_mounts=extra_mounts,
        checkpoint_store_volume_name=None if store_volume is None else store_volume[0],
        checkpoint_store_mount=None if store_volume is None else MODAL_CHECKPOINT_STORE_MOUNT,
        tape_volume_name=getattr(args, "modal_tape_volume", None) or None,
        named_secrets=((modal_sandbox_secret_name(args, spec),)
                       if getattr(args, "modal_sandbox_secret", None) else ()),
        model_volume_name=(getattr(args, "modal_model_volume", None) or None) if rl else None,
        model_volume_mount=MODEL_STORE_MOUNT if rl and getattr(args, "modal_model_volume", None) else None,
        cpu_override=getattr(args, "modal_cpu", None),
        memory_gib_override=getattr(args, "modal_memory_gib", None),
        tape_subdir=(modal_tape_subdir(args.cluster_prefix, learner_id)
                     if getattr(args, "modal_tape_volume", None) else None),
        workdir=code_workdir(args),
        # Opt-in overrides (default: Modal runner defaults, 10 retries / 24 h).
        # Acceptance runs pass --modal-retries 0 so a learner exit is final
        # (no re-run billing) and --modal-timeout-s as the Modal-side hard stop.
        # --no-island-relaunch ("a failed island is never relaunched") forces the
        # platform retries to 0 too: Modal's own retry re-runs a failed island
        # container by itself (up to 10 times) and would silently undo the flag --
        # a second paid container, replaying from the last checkpoint, that the
        # launcher neither tracks nor tears down. Wins over an explicit --modal-retries N.
        **({"retries": 0} if getattr(args, "no_island_relaunch", False)
           else {"retries": int(args.modal_retries)}
           if getattr(args, "modal_retries", None) is not None else {}),
        **({"timeout_s": int(args.modal_timeout_s)}
           if getattr(args, "modal_timeout_s", None) is not None else {}),
    )


def modal_tape_subdir(cluster_prefix: str, learner_id: int) -> str:
    """Per-run, per-island directory of the Modal tape Volume."""
    from ..modal_runner import modal_app_name

    return f"{modal_app_name(cluster_prefix)}/l{learner_id}"


def _modal_tape_run_dir(args):
    # A module-level import: `run()` binds `runs` only part-way through, and
    # its finally block (where the tape is pulled) also runs on early failures.
    from .. import runs as _runs

    return _runs.run_dir(args.cluster_prefix)


def pull_modal_tapes(modal_ops, modal_cfgs: dict, run_dir) -> dict[str, str]:
    """Copy each Modal island's tape Volume subdir into
    ``<run_dir>/modal-tape/<island>/``; returns {island: "ok" | error}.
    Never raises: a missing tape is reported, not fatal to teardown."""
    out: dict[str, str] = {}
    for name, cfg in modal_cfgs.items():
        vol, sub = getattr(cfg, "tape_volume_name", None), getattr(cfg, "tape_subdir", None)
        if not vol or not sub:
            continue
        dest = Path(run_dir) / "modal-tape" / name
        try:
            modal_ops.pull_tape(vol, sub, str(dest))
            files = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
            out[name] = "ok" if files else "empty"
            print(f"[launcher] Modal tape of {name}: {vol}:{sub} -> {dest} ({len(files)} file(s))")
        except Exception as e:  # noqa: BLE001
            out[name] = f"error: {e}"
            print(f"[launcher] WARN: could not pull the Modal tape of {name} from "
                  f"{vol}:{sub}: {e}", file=sys.stderr)
    return out


def require_modal_for_gpu_exact(args, specs: list[ClusterSpec]) -> None:
    """--modal-gpu-exact only means something on Modal islands; refuse it
    rather than silently launching unpinned learners elsewhere."""
    if getattr(args, "modal_gpu_exact", False):
        other = sorted({s.cloud for s in specs if s.cloud != "modal"})
        if other:
            raise ValueError(
                f"--modal-gpu-exact applies only to Modal islands; --gpu also has {other}"
            )


def _tail_modal(modal_ops, call_id: str, prefix: str, collector=None, guard=None) -> int:
    """Stream a Modal island's container logs (the Modal twin of _tail).
    `guard` (modal_runner.ContainerIdGuard) sees every line to catch container changes."""
    while True:
        try:
            for line in modal_ops.stream_logs(call_id):
                # a Modal log entry may hold several lines: prefix each one
                for part in str(line).rstrip("\n").split("\n"):
                    print(f"[{prefix}] {part.rstrip()}", flush=True)
                if collector is not None:
                    collector.feed(line)
                if guard is not None:
                    guard.feed(line)
            return 0
        except Exception as e:  # transient stream drops: reconnect
            print(f"[{prefix}] log stream error: {e}; retrying", flush=True)
            time.sleep(5)
MODAL_STOP_VERIFY_ATTEMPTS = 5
MODAL_STOP_VERIFY_DELAY_S = 5.0


def _list_modal_apps() -> list[dict]:
    """Rows of ``modal app list --json`` (app_id, description, state, tasks,
    created_at, stopped_at). Raises when the listing fails."""

    import subprocess as _subprocess

    proc = _subprocess.run(
        [sys.executable, "-m", "modal", "app", "list", "--json"], capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"modal app list failed: {(proc.stdout + proc.stderr).strip() or proc.returncode}")
    return json.loads(proc.stdout or "[]")


def _created_unix(row) -> float | None:
    from datetime import datetime

    value = row.get("created_at")
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _verify_modal_app_stopped(modal_ops, args, *, run_started_unix: float | None = None) -> bool:
    """After stop_app, confirm on the provider that nothing of this app runs.

    ``modal app list`` can hold several rows with the same app name (earlier
    runs). EVERY row with this name must be stopped with 0 tasks (bounded
    retries); the rows created at/after this run's start are this run's app,
    and their app ids are recorded in <run dir>/teardown.json.
    """

    from .. import runs

    checks, confirmed, ours = [], False, []
    for attempt in range(MODAL_STOP_VERIFY_ATTEMPTS):
        try:
            rows = [r for r in _list_modal_apps() if r.get("description") == modal_ops.app_name]
        except Exception as e:  # noqa: BLE001 - an unreadable listing is "unverified"
            checks.append(f"error: {e}")
        else:
            ours = [r.get("app_id") for r in rows
                    if run_started_unix is not None
                    and (_created_unix(r) or 0) >= run_started_unix - 1]
            summary = [{"app_id": r.get("app_id"), "state": str(r.get("state", "")).lower(),
                        "tasks": r.get("tasks"), "created_at": r.get("created_at")}
                       for r in rows]
            checks.append(summary)
            if all(s["state"] == "stopped" and int(s["tasks"] or 0) == 0 for s in summary):
                confirmed = True  # also when no row is listed any more
                break
        if attempt + 1 < MODAL_STOP_VERIFY_ATTEMPTS:
            time.sleep(MODAL_STOP_VERIFY_DELAY_S)
    record = {"provider": "modal", "app": modal_ops.app_name, "this_run_app_ids": ours,
              "confirmed_stopped": confirmed, "checks": checks}
    try:
        path = runs.run_dir(args.cluster_prefix) / "teardown.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
    except OSError as e:
        print(f"[launcher] could not write the teardown record: {e}", file=sys.stderr)
    if not confirmed:
        print(f"[launcher] WARN: Modal app {modal_ops.app_name} not confirmed stopped after "
              f"{len(checks)} check(s) (every row with this name must be stopped with 0 "
              f"tasks; last: {checks[-1]}); stop it by hand: modal app stop "
              f"{modal_ops.app_name} (exit {TEARDOWN_UNVERIFIED_EXIT})", file=sys.stderr)
    return confirmed


def container_set_poller_for(args, modal_ops, modal_cfgs, on_change, *, start: bool = True):
    """The log-independent container guard (modal_runner.ContainerSetPoller), only when
    a new container is never expected: no launcher relaunch, no checkpoint-store resume
    and no droppable island. None otherwise."""
    if modal_ops is None or not modal_cfgs or not callable(getattr(modal_ops, "list_container_ids", None)):
        return None
    if effective_recover_timeout(args) != 0 or resumes_in_new_container(args):
        return None
    if any(droppable_rejoins_in_new_container(args, cfg) for cfg in modal_cfgs.values()):
        return None
    from ..modal_runner import ContainerSetPoller

    expected = sum(max(1, int(getattr(cfg, "num_nodes", 1) or 1)) for cfg in modal_cfgs.values())
    poller = ContainerSetPoller(modal_ops.list_container_ids, expected, on_change)
    return poller.start() if start else poller


def container_log_follower_for(args, modal_ops, modal_cfgs, route, *, poller, start: bool = True):
    """The follower of rescheduled containers' logs (modal_runner.ContainerLogFollower),
    only when a new container is expected (no failing container guard: ``poller``
    is None) and the ops can stream one container. None otherwise."""
    if poller is not None or modal_ops is None or not modal_cfgs:
        return None
    if not (callable(getattr(modal_ops, "list_container_ids", None))
            and callable(getattr(modal_ops, "stream_container_logs", None))):
        return None
    from ..modal_runner import ContainerLogFollower

    expected = sum(max(1, int(getattr(cfg, "num_nodes", 1) or 1)) for cfg in modal_cfgs.values())
    follower = ContainerLogFollower(modal_ops.list_container_ids, modal_ops.stream_container_logs,
                                    expected, route=route)
    return follower.start() if start else follower


def effective_recover_timeout(args) -> float:
    """The fleet controller's learner relaunch budget.

    ``--no-island-relaunch`` or ``--modal-retries 0`` (a learner exit is final,
    e.g. acceptance runs) -> 0: a failed island is torn down, never relaunched
    by the launcher, so no second paid container can start before the app
    stops. Otherwise ``--recover-timeout`` unchanged (all clouds share this
    loop: sky islands relaunch through the same FleetController).
    """
    if getattr(args, "no_island_relaunch", False):
        return 0
    if getattr(args, "modal_retries", None) == 0 and not getattr(args, "modal_launcher_relaunch", False):
        return 0
    return args.recover_timeout
