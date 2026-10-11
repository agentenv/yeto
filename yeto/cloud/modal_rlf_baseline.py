"""Modal deployment shape for the RLF Qwen3-30B-A3B baseline (rlf-codex-rl 5.6).

1xA100-80GB container running ``yeto.rl.eval.rlf_baseline.run_from_config``:
SGLang bf16 from the ``yeto-fn-models`` Volume
(``hf/Qwen3-30B-A3B/ad44e777/``) + Miles session server + the Codex harness
against per-episode RLF CTF VMs (created by the in-container Modal client).

Mirrors ``modal_eval_island.build_eval_app``: same pinned Miles-ports image,
same workdir/codex mounts.  Launch (like the S17 N11 smoke):

    python tools/probes/... style driver, or:

    modal run yeto.cloud.modal_rlf_baseline --config-json run.json

Local entrypoint inputs (never baked into the image):

* ``YETO_CODEX_BUNDLE_DIR`` — the attested Codex bundle dir
  (``scripts/fetch_codex_bundle.py``), mounted at ``/opt/yeto/codex``;
* ``YETO_RLF_CTF_TASKS_DIR`` — local dir of admitted task packages, mounted
  at ``/root/rlf-tasks`` (the container-side value overrides it);
* ``YETO_RLF_BASELINE_ENVS_JSON`` — JSON file with the container env:
  the Codex contract env (``yeto.launcher.codex_container_env(
  codex_bundle_contract(...))``, same as the eval island's codex-env.json),
  ``TBENCH_REWARD_HMAC_KEY``, ``YETO_RLF_CTF_FLAGS_JSON`` (trusted flags;
  live only in the Modal Secret, never in the image or the traces) and the
  sandbox-only ``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET`` the RLF provider
  needs to open episode VMs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

APP_NAME = "yeto-rlf-baseline"
BASELINE_GPU = "A100-80GB"
MODELS_VOLUME = "yeto-fn-models"                 # hf/<name>/<rev[:8]>/ + MANIFEST.json
RESULTS_VOLUME = "yeto-rlf-baseline-runs"
MODELS_MOUNT = "/mnt/yeto-models"                # yeto.launcher.MODEL_STORE_MOUNT
RESULTS_MOUNT = "/mnt/yeto-rlf-runs"
WORKDIR_MOUNT = "/root/sky_workdir"              # same as the training/eval islands
CODEX_MOUNT = "/opt/yeto/codex"                  # yeto.launcher.CODEX_CONTAINER_DIR
TASKS_MOUNT = "/root/rlf-tasks"
ENVS_JSON_ENV = "YETO_RLF_BASELINE_ENVS_JSON"

_SECRET_ENV_KEYS = ("TBENCH_REWARD_HMAC_KEY", "YETO_RLF_CTF_FLAGS_JSON",
                    "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")


def default_image_ref() -> str:
    from yeto.rl import MILES_NEXT_IMAGE  # re-exported pin (same route as the eval island)

    return MILES_NEXT_IMAGE.removeprefix("docker:")


@dataclass
class BaselineFunctionSpec:
    """Shape of the baseline Modal function; ``gpu`` is a Modal GPU string."""

    app_name: str
    workdir: str
    codex_dir: str
    tasks_dir: str
    models_volume: str = MODELS_VOLUME
    results_volume: str = RESULTS_VOLUME
    gpu: str = BASELINE_GPU
    cpu: float = 8.0
    memory_mib: int = 65536
    timeout_s: int = 6 * 3600
    image_ref: str = field(default_factory=default_image_ref)
    envs: Mapping[str, str] = field(default_factory=dict)

    def container_env(self) -> dict[str, str]:
        # No PYTHONPATH override (S17 N11: it breaks Modal's container entrypoint).
        return {"HOME": "/root", "PYTHONUNBUFFERED": "1", "YETO_RLF_CTF_TASKS_DIR": TASKS_MOUNT,
                **{str(k): str(v) for k, v in self.envs.items()}}


def build_baseline_app(spec: BaselineFunctionSpec) -> tuple[Any, Any]:
    """(app, baseline GPU function).  Models Volume read-only; results Volume committed."""
    import modal

    image = (modal.Image.from_registry(spec.image_ref)
             .env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})
             .add_local_dir(spec.workdir, WORKDIR_MOUNT, copy=False,
                            ignore=[".git", ".venv", ".claude", "**/__pycache__", "**/*.pyc", "s1-runs"])
             .add_local_dir(spec.codex_dir, CODEX_MOUNT, copy=False)
             .add_local_dir(spec.tasks_dir, TASKS_MOUNT, copy=False))
    app = modal.App(spec.app_name)
    models = modal.Volume.from_name(spec.models_volume, create_if_missing=True)
    results = modal.Volume.from_name(spec.results_volume, create_if_missing=True)
    secret = modal.Secret.from_dict(spec.container_env())
    run = app.function(image=image, gpu=spec.gpu, cpu=spec.cpu, memory=spec.memory_mib,
                       timeout=spec.timeout_s, secrets=[secret],
                       volumes={MODELS_MOUNT: models, RESULTS_MOUNT: results},
                       name="rlf_baseline")(baseline_body)
    return app, run


def baseline_body(config: Mapping[str, Any]) -> dict[str, Any]:
    """In-container entry: run the baseline, then copy the evidence onto the results Volume."""
    import shutil
    import sys
    import time

    t0 = time.time()
    sys.path.insert(0, WORKDIR_MOUNT)
    from yeto.rl.eval import rlf_baseline

    config = dict(config)
    local_out = "/tmp/rlf-baseline-out"
    config["out_dir"] = local_out
    config.setdefault("tasks_dir", TASKS_MOUNT)
    # capture sglang/session-server logs into the results volume copy
    config.setdefault("baseline", {})
    config["baseline"].setdefault("log_dir", local_out + "/logs")
    # Trusted flags arrive via the Modal Secret env, never via run.json on disk.
    if not config.get("flags"):
        config["flags"] = json.loads(os.environ.get("YETO_RLF_CTF_FLAGS_JSON") or "{}")
    run_name = str(config.get("run_name") or time.strftime("rlf-%Y%m%d-%H%M%S", time.gmtime()))
    result = rlf_baseline.run_from_config(config)
    final_dir = Path(RESULTS_MOUNT) / run_name
    shutil.copytree(local_out, final_dir, dirs_exist_ok=True)
    import modal

    modal.Volume.from_name(str(config.get("results_volume") or RESULTS_VOLUME),
                           create_if_missing=True).commit()
    return {"run_name": run_name, "results": str(final_dir), "wall_s": round(time.time() - t0, 3),
            **result}


def spec_from_env(environ: Mapping[str, str] | None = None) -> BaselineFunctionSpec:
    """Local-side spec from the environment (see the module docstring)."""
    environ = os.environ if environ is None else environ
    repo = Path(__file__).resolve().parents[2]
    codex_dir = environ.get("YETO_CODEX_BUNDLE_DIR")
    tasks_dir = environ.get("YETO_RLF_CTF_TASKS_DIR")
    if not codex_dir or not Path(codex_dir).is_dir():
        raise RuntimeError("YETO_CODEX_BUNDLE_DIR must point at the attested Codex bundle dir")
    if not tasks_dir or not Path(tasks_dir).is_dir():
        raise RuntimeError("YETO_RLF_CTF_TASKS_DIR must point at the local admitted task packages")
    envs: dict[str, str] = {}
    envs_json = environ.get(ENVS_JSON_ENV)
    if envs_json:
        envs.update({str(k): str(v) for k, v in json.loads(Path(envs_json).read_text()).items()})
    for key in _SECRET_ENV_KEYS:
        if environ.get(key) and key not in envs:
            envs[key] = str(environ[key])
    return BaselineFunctionSpec(app_name=environ.get("YETO_RLF_BASELINE_APP", APP_NAME),
                                workdir=str(repo), codex_dir=codex_dir, tasks_dir=tasks_dir,
                                envs=envs)


try:  # the Modal entry point (only usable with the modal client installed)
    import modal as _modal

    try:
        app, baseline_run = build_baseline_app(spec_from_env())
    except RuntimeError:
        # In-container hydration re-imports this module; the local-only spec
        # env (bundle dir, tasks dir) is absent there.  The container finds
        # the function body (``baseline_body``) by qualname, not via ``app``.
        app = None
        baseline_run = None

    def main(argv: "list[str] | None" = None) -> int:
        """Local driver: ``python -m yeto.cloud.modal_rlf_baseline run.json``.

        Driver pattern (same as the S17 eval-island probes): the app is built
        locally and the run is spawned inside ``app.run()``; ``modal run -m``
        is NOT usable here because the spec needs local-only paths.
        """
        import sys

        args = sys.argv[1:] if argv is None else argv
        if app is None or baseline_run is None:
            raise RuntimeError("baseline app is only built locally (spec env missing)")
        payload = json.loads(Path(args[0]).read_text())
        with app.run():
            result = baseline_run.remote(payload)
        print(json.dumps(result, indent=1, default=str))
        return 0

except ImportError:  # pragma: no cover - CPU tests have no modal client
    _modal = None
    app = None
    baseline_run = None


if __name__ == "__main__":
    raise SystemExit(main())
