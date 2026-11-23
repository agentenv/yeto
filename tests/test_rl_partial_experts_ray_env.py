"""Generic partial experts reach the Ray workers, not just the island driver.

The island configures partial experts after ``ray start``, so its
``os.environ`` writes never reach the raylet, and Ray workers inherit the
raylet's environment plus the job's ``runtime_env`` ``env_vars`` -- nothing
else from the driver. Without forwarding, the trainer actors never froze the
non-selected experts (no ``partial-experts (in place)`` hook, a full-model
DDP grad buffer on hardware).

The cross-process test builds a worker environment the way Ray does (a
raylet environment without the switches, plus ``runtime_env``), starts a
fresh interpreter in it, and checks that the repo's ``sitecustomize`` armed
the model-provider hook on a stand-in Miles module.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from yeto.rl.adapters.miles.entry import connect_island_ray
from yeto.rl.adapters.miles.models.partial_experts import in_place
from yeto.rl.adapters.miles.models.partial_experts import selection as sel

ROOT = Path(__file__).resolve().parent.parent
KEYS = (in_place.ENV_SELECTION, in_place.GROUPED_SINGLE_PARAM_ENV)
ADDRESS = "10.0.0.1:6379"

# A worker body: report what the process sees and whether sitecustomize wrapped the provider.
WORKER = textwrap.dedent("""
    import json, os, sys
    import sitecustomize
    from miles.backends.megatron_utils import model_provider
    print(json.dumps({
        "sitecustomize": os.path.realpath(sitecustomize.__file__),
        "env": {k: os.environ.get(k) for k in sys.argv[1:]},
        "wrapped": model_provider.get_model_provider_func is not model_provider.ORIGINAL,
    }))
""")


class _FakeRay:
    def __init__(self):
        self.runtime_env = None

    def is_initialized(self):
        return False

    def init(self, *, address, runtime_env):
        assert address == ADDRESS
        self.runtime_env = runtime_env


def _selection_file(tmp_path: Path) -> Path:
    path = tmp_path / "selection.json"
    sel.ExpertSelection.in_place("uniform", model="qwen3_moe", num_experts=8, num_trainable=4,
                                 ep_size=2, layers=(1, 2)).write(path)
    return path


def _fake_miles(tmp_path: Path) -> Path:
    root = tmp_path / "fake_miles"
    pkg = root / "miles" / "backends" / "megatron_utils"
    pkg.mkdir(parents=True)
    for part in (root / "miles", root / "miles" / "backends", pkg):
        (part / "__init__.py").write_text("")
    (pkg / "model_provider.py").write_text(textwrap.dedent("""
        def get_model_provider_func(args, role="actor"):
            raise AssertionError("not called here")

        ORIGINAL = get_model_provider_func
    """))
    return root


def _runtime_env(environ: dict[str, str]) -> dict[str, str]:
    ray = _FakeRay()
    assert connect_island_ray(environ=dict(environ), ray_module=ray) == ADDRESS
    return ray.runtime_env["env_vars"]


def _raylet_env() -> dict[str, str]:
    """The environment ``ray start`` gave the raylet: no partial-experts switch, no yeto switch."""

    return {k: v for k, v in os.environ.items()
            if k not in KEYS and not k.startswith("YETO_") and k not in ("PYTHONPATH", "PYTHONSTARTUP")}


def _run_worker(env_vars: dict[str, str]) -> dict:
    env = {**_raylet_env(), **env_vars}  # Ray merges the job's runtime_env env_vars over the raylet's
    proc = subprocess.run([sys.executable, "-c", WORKER, *KEYS], env=env, cwd=str(ROOT.parent),
                          capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _driver_environ(tmp_path: Path) -> dict[str, str]:
    """The island driver after ``_configure_generic_partial_experts`` (it writes island_env)."""

    environ = {"RAY_ADDRESS": ADDRESS, "PYTHONPATH": os.pathsep.join([str(ROOT), str(_fake_miles(tmp_path))])}
    environ.update(in_place.island_env(str(_selection_file(tmp_path))))
    return environ


def test_runtime_env_carries_both_partial_experts_switches(tmp_path):
    environ = _driver_environ(tmp_path)
    env_vars = _runtime_env(environ)
    assert {k: env_vars.get(k) for k in KEYS} == {
        in_place.ENV_SELECTION: environ[in_place.ENV_SELECTION],
        in_place.GROUPED_SINGLE_PARAM_ENV: "0",
    }
    assert in_place.worker_env(environ) == dict(in_place.island_env(environ[in_place.ENV_SELECTION]))


def test_without_a_selection_nothing_new_is_forwarded():
    # Every other island (the customer K3.1 path included: its freeze is a Miles CLI flag)
    # keeps the same runtime_env, even when the driver happens to set the NVTE switch.
    plain = _runtime_env({"RAY_ADDRESS": ADDRESS})
    with_nvte = _runtime_env({"RAY_ADDRESS": ADDRESS, in_place.GROUPED_SINGLE_PARAM_ENV: "0",
                              in_place.ENV_SELECTION: ""})
    assert plain == with_nvte
    assert not set(KEYS) & set(plain)
    assert in_place.worker_env({}) == {}


def test_a_ray_worker_process_arms_the_freeze_from_the_runtime_env(tmp_path):
    env_vars = _runtime_env(_driver_environ(tmp_path))
    report = _run_worker(env_vars)
    assert report["sitecustomize"] == os.path.realpath(ROOT / "sitecustomize.py")
    assert report["env"] == {k: env_vars[k] for k in KEYS}
    assert report["wrapped"] is True

    # The pre-fix runtime_env (driver os.environ only): the worker sees neither switch and
    # Miles' provider stays unwrapped -- every expert trains.
    stale = {k: v for k, v in env_vars.items() if k not in KEYS}
    report = _run_worker(stale)
    assert report["env"] == dict.fromkeys(KEYS)
    assert report["wrapped"] is False
