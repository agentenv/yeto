"""The trainer's weight-update NCCL group uses the SGLang engines' channel count.

SGLang's deterministic inference with TP > 1 pins ``NCCL_MIN_NCHANNELS =
NCCL_MAX_NCHANNELS = 8`` inside every engine process; the trainer rank of
Miles' weight-update group kept its topology count (24 on an H100 node), and
the first broadcast's connection setup failed in every engine with "Message
truncated : received 6144 bytes instead of 2048" (24 versus 8 connection
records of 256 bytes). The trainer now gives that one communicator the
engines' count through ``ProcessGroupNCCL.Options().config.min_ctas/max_ctas``.

The channel model below is NCCL's per-rank rule (``src/graph/connect.cc``,
``ncclTopoPostset``): clamp the agreed count to ``min(NCCL_MAX_NCHANNELS,
config.maxCTAs)`` and duplicate up to ``max(NCCL_MIN_NCHANNELS,
config.minCTAs)``. Connection setup exchanges one record per channel.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from yeto.rl.adapters.miles import weight_update_channels as wuc
from yeto.rl.adapters.miles.entry import connect_island_ray

ROOT = Path(__file__).resolve().parent.parent
ADDRESS = "10.0.0.1:6379"
H100_TOPOLOGY_CHANNELS = 24
NCCL_CONNECT_BYTES = 256
MAXCHANNELS = 64


def _v2_args(**changes):
    """The failing hardware island: fixed partition, SGLang TP 2, deterministic inference (yeto default)."""

    values = dict(rollout_num_gpus=2, rollout_num_gpus_per_engine=2, sglang_enable_deterministic_inference=True,
                  debug_train_only=False, colocate=False, update_weight_transfer_mode="broadcast")
    values.update(changes)
    return types.SimpleNamespace(**values)


class _FakeRay:
    runtime_env = None

    def is_initialized(self):
        return False

    def init(self, *, address, runtime_env):
        self.runtime_env = runtime_env


def _runtime_env(miles_args, **environ):
    ray = _FakeRay()
    connect_island_ray(environ={"RAY_ADDRESS": ADDRESS, **environ}, ray_module=ray, miles_args=miles_args)
    return ray.runtime_env["env_vars"]


def _engine_env(tp_size: int, deterministic: bool) -> dict[str, str]:
    """SGLang's ``_handle_deterministic_inference`` on CUDA (arg_groups/attention_hook.py)."""

    if not (deterministic and tp_size > 1):
        return {}
    n = str(wuc.SGLANG_DEFAULT_CHANNELS)
    return {"NCCL_MIN_NCHANNELS": n, "NCCL_MAX_NCHANNELS": n}


def _channels(topology: int, env: dict[str, str], *, min_ctas: int = 1, max_ctas: int = MAXCHANNELS) -> int:
    env_min = int(env.get("NCCL_MIN_NCHANNELS", 0))
    env_max = int(env.get("NCCL_MAX_NCHANNELS", MAXCHANNELS))
    return max(min(topology, env_max, max_ctas), max(env_min, min_ctas))


def _trainer_channels(env_vars: dict[str, str]) -> int:
    """The trainer rank's channels in the update group, from what the island forwards."""

    pinned = env_vars.get(wuc.ENV)
    if not pinned:
        return _channels(H100_TOPOLOGY_CHANNELS, {})
    return _channels(H100_TOPOLOGY_CHANNELS, {}, min_ctas=int(pinned), max_ctas=int(pinned))


def test_update_group_channel_counts_match_the_engines():
    env_vars = _runtime_env(_v2_args())
    engine = _channels(H100_TOPOLOGY_CHANNELS, _engine_env(tp_size=2, deterministic=True))
    trainer = _trainer_channels(env_vars)
    # before: the engines expect 2048 bytes of connection records, the trainer sends 6144
    assert (trainer * NCCL_CONNECT_BYTES, engine * NCCL_CONNECT_BYTES) == (2048, 2048)
    assert env_vars[wuc.ENV] == "8"


def test_the_engines_override_travels_too():
    assert _runtime_env(_v2_args(), SGLANG_DETERMINISTIC_NCCL_NCHANNELS="4")[wuc.ENV] == "4"
    with pytest.raises(ValueError, match="positive"):
        _runtime_env(_v2_args(), SGLANG_DETERMINISTIC_NCCL_NCHANNELS="0")


@pytest.mark.parametrize("changes", [
    dict(debug_train_only=True),  # trainer-only island (the customer K3.1 trainer): no engine
    dict(rollout_num_gpus=0),
    dict(rollout_num_gpus_per_engine=1),  # SGLang pins nothing at TP 1 (the passing dense runs)
    dict(sglang_enable_deterministic_inference=False),
])
def test_nothing_is_forwarded_when_the_engines_do_not_pin(changes):
    args = _v2_args(**changes)
    env_vars = _runtime_env(args)
    assert wuc.ENV not in env_vars
    assert env_vars == _runtime_env(None)
    if args.rollout_num_gpus and not args.debug_train_only:  # engines exist: both sides unpinned
        engine_env = _engine_env(args.rollout_num_gpus_per_engine, args.sglang_enable_deterministic_inference)
        assert _trainer_channels(env_vars) == _channels(H100_TOPOLOGY_CHANNELS, engine_env)


def test_nccl_options_set_both_bounds():
    class Options:
        def __init__(self):
            self.config = types.SimpleNamespace(min_ctas=None, max_ctas=None)

    fake = types.SimpleNamespace(ProcessGroupNCCL=types.SimpleNamespace(Options=Options))
    options = wuc.nccl_options(8, fake)
    assert (options.config.min_ctas, options.config.max_ctas) == (8, 8)
    with pytest.raises(RuntimeError, match="per-communicator"):
        wuc.nccl_options(8, types.SimpleNamespace())


def test_patch_only_touches_nccl_groups_and_skips_rocm(monkeypatch):
    calls = []
    module = types.ModuleType(wuc._BROADCAST)
    module.init_process_group = lambda **kwargs: calls.append(kwargs) or "pg"
    monkeypatch.setenv(wuc.ENV, "8")
    monkeypatch.setattr(wuc, "nccl_options", lambda n: ("opts", n))
    monkeypatch.setattr(wuc, "_is_rocm", lambda: False)
    wuc.install_on_broadcast_module(module)
    wuc.install_on_broadcast_module(module)  # once per module
    assert module.init_process_group(backend="nccl", group_name="miles-pp_0") == "pg"
    module.init_process_group(backend="gloo", group_name="g")
    module.init_process_group(backend="nccl", group_name="given", pg_options="mine")
    monkeypatch.setattr(wuc, "_is_rocm", lambda: True)
    module.init_process_group(backend="nccl", group_name="rocm")
    assert [c.get("pg_options") for c in calls] == [("opts", 8), None, "mine", None]


WORKER = textwrap.dedent("""
    import json, os, sys
    import sitecustomize
    from yeto.rl.adapters.miles import weight_update_channels as wuc
    wuc.nccl_options = lambda n: ["opts", n]  # this host's torch may lack NCCL
    wuc._is_rocm = lambda: False
    from miles.backends.training_utils.weight_update.protocols import broadcast
    kwargs = broadcast.init_process_group(backend="nccl", world_size=3, rank=0, group_name="miles-pp_0")
    print(json.dumps({"env": os.environ.get(wuc.ENV), "pg_options": kwargs.get("pg_options")}))
""")


def _fake_miles(tmp_path: Path) -> Path:
    root = tmp_path / "fake_miles"
    pkg = root / "miles/backends/training_utils/weight_update/protocols"
    pkg.mkdir(parents=True)
    part = root / "miles"
    for sub in ("", "backends", "backends/training_utils", "backends/training_utils/weight_update",
                "backends/training_utils/weight_update/protocols"):
        (part / sub / "__init__.py").write_text("")
    (pkg / "broadcast.py").write_text("def init_process_group(**kwargs):\n    return kwargs\n")
    return root


def _run_worker(env_vars: dict[str, str]) -> dict:
    raylet = {k: v for k, v in os.environ.items() if not k.startswith("YETO_") and k != "PYTHONPATH"}
    proc = subprocess.run([sys.executable, "-c", WORKER], env={**raylet, **env_vars}, cwd=str(ROOT.parent),
                          capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_trainer_worker_builds_the_update_group_with_the_engines_count(tmp_path):
    pythonpath = os.pathsep.join([str(ROOT), str(_fake_miles(tmp_path))])
    env_vars = _runtime_env(_v2_args(), PYTHONPATH=pythonpath)
    assert _run_worker(env_vars) == {"env": "8", "pg_options": ["opts", 8]}
    # a worker without the forwarded count keeps Miles' call unchanged
    assert _run_worker({k: v for k, v in env_vars.items() if k != wuc.ENV}) == {"env": None, "pg_options": None}


def _torch_has_nccl_options() -> bool:
    try:
        import torch.distributed as dist
    except Exception:  # noqa: BLE001 - no torch: skip
        return False
    return hasattr(getattr(dist, "ProcessGroupNCCL", None), "Options")


@pytest.mark.skipif(not _torch_has_nccl_options(), reason="needs a PyTorch build with NCCL")
def test_real_torch_options_accept_the_bounds():
    options = wuc.nccl_options(8)
    assert (options.config.min_ctas, options.config.max_ctas) == (8, 8)
