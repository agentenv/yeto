"""Heterogeneous islands joining one session on the real Rust syncer. CPU, no Ray.

Two decoupled bridges (``yeto.rl.decoupled``) whose HELLO differs in exactly
one backend-identity field, everything else (layout, revision, LoRA config,
compat group, LR schedule) identical:

* ``device_family`` ``nvidia`` vs ``ascend`` (gap analysis T1);
* the verl train backend ``fsdp2`` vs ``megatron``, i.e. a different
  ``param_map_sha256`` for the same canonical layout (gap analysis T2).

Today the device family and the param map are hashed into the session
contract, so the second island is refused at the handshake.  The identity
split (contract = semantic layout only; hardware / engine are recorded and
admitted by an explicit policy) must make both islands merge.  The
heterogeneous cases are therefore strict xfail: they are the acceptance test
of that split and flip to XPASS (a failure under ``strict``) once it lands,
at which point the marker is removed.  The homogeneous control must pass
before and after.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest
import torch

from free_ports import free_port as _port
from yeto.rl.adapters.verl.identity import backend_identity
from yeto.rl.core import canonical_state
from yeto.rl.decoupled import DecoupledBridgeConfig, DecoupledRlBridge
from yeto.rl.engine.backend_identity import lr_schedule_sha256

ROOT = Path(__file__).resolve().parent.parent
MODEL_REVISION = "a" * 40
LORA_CONFIG_HASH = "b" * 64
COMPAT_GROUP = "nvidia-h100"  # identical on both islands: only the identity field under test differs
LR_SCHEDULE = lr_schedule_sha256("constant", None, 1e-5)
TOTAL_STEPS = 4  # two sweeps of two fragments
HANDSHAKE_TIMEOUT_S = 20.0



class HelloRefusedOnContract(AssertionError):
    """The syncer refused the second HELLO on the session contract alone
    (same layout fingerprint): the T1/T2 symptom, and nothing else."""


# T1 / T2 refusal not yet lifted (identity split pending).  ``raises`` keeps
# the xfail honest: any other failure (timeout, layout mismatch) is a real failure.
IDENTITY_SPLIT_PENDING = pytest.mark.xfail(
    strict=True, raises=HelloRefusedOnContract, reason="T1/T2: device_family and param map are hashed into the session "
                        "contract; the syncer refuses the second HELLO until the identity split lands")

CASES = {
    "same_identity": ({"device_family": "nvidia"}, {"device_family": "nvidia"}),
    "device_family": ({"device_family": "nvidia"}, {"device_family": "ascend"}),
    "train_backend": ({"device_family": "nvidia", "train_backend": "fsdp2"},
                      {"device_family": "nvidia", "train_backend": "megatron"}),
}


@pytest.fixture(scope="module")
def syncer_binary():
    subprocess.run(["cargo", "build", "-q"], cwd=ROOT / "syncer", check=True)
    return ROOT / "syncer/target/debug/yeto-syncer"


def _state(version, tensors):
    return canonical_state(version, tensors, base_model_revision=MODEL_REVISION,
                           lora_config_hash=LORA_CONFIG_HASH)


def _start_syncer(binary, port, tmp_path, log):
    argv = [str(binary), "--port", str(port), "--learners", "2", "--quorum", "2",
            "--grace-ms", "0", "--pipeline", "2", "--sync-interval-steps", "2",
            "--delta-correction", "none", "--total-steps", str(TOTAL_STEPS),
            "--outer-lr", "0.7", "--outer-momentum", "0.9", "--quorum-timeout-s", "30",
            "--final-grace-s", "10", "--learner-weight", "equal",
            "--event-tape", str(tmp_path / "tape.jsonl")]
    return subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)


def _bridge(initial, port, learner_id, identity_kwargs):
    identity = backend_identity(compat_group=COMPAT_GROUP, **identity_kwargs)
    return DecoupledRlBridge(initial, DecoupledBridgeConfig(
        syncer_addr=("127.0.0.1", port), learner_id=learner_id, total_fragment_steps=TOTAL_STEPS,
        num_fragments=2, pipeline=2, local_horizon=2, expected_specs=initial.specs,
        base_model_revision=MODEL_REVISION, lora_config_hash=LORA_CONFIG_HASH,
        canonical_layout_hash=initial.layout_hash, wan_streams=0,
        backend_identity_sha256=identity.sha256(), lr_schedule_sha256=LR_SCHEDULE,
        compat_group=COMPAT_GROUP))


def _handshake_both(bridges):
    """Start both bridges concurrently (a fixed roster waits for both HELLOs)
    and take the initial cut on each; return the per-island error, if any."""
    errors: list[BaseException | None] = [None] * len(bridges)
    cuts = [None] * len(bridges)

    def run(index):
        try:
            bridges[index].start()
            cut = bridges[index].wait_for_initial_cut(optimizer_steps=0, action_tokens=0)
            bridges[index].commit_initial_cut(cut, optimizer_steps=0, action_tokens=0)
            cuts[index] = cut
        except BaseException as exc:  # noqa: BLE001 - reported to the main thread
            errors[index] = exc

    threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(len(bridges))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(HANDSHAKE_TIMEOUT_S)
    hung = [i for i, thread in enumerate(threads) if thread.is_alive()]
    return cuts, errors, hung


@pytest.mark.parametrize("case", [
    "same_identity",
    pytest.param("device_family", marks=IDENTITY_SPLIT_PENDING),
    pytest.param("train_backend", marks=IDENTITY_SPLIT_PENDING),
])
def test_islands_with_different_identity_share_one_session(syncer_binary, tmp_path, case):
    left, right = CASES[case]
    assert backend_identity(compat_group=COMPAT_GROUP, **left).sha256() != \
        backend_identity(compat_group=COMPAT_GROUP, **right).sha256() or case == "same_identity"
    initial = _state(0, {f"base_model.model.layer{i}.lora_A.weight": torch.zeros(1, i + 1)
                         for i in range(4)})
    port = _port()
    log = open(tmp_path / "syncer.log", "w")
    process = _start_syncer(syncer_binary, port, tmp_path, log)
    bridges = [_bridge(initial, port, 0, left), _bridge(initial, port, 1, right)]
    try:
        cuts, errors, hung = _handshake_both(bridges)
        syncer_log = (tmp_path / "syncer.log").read_text()
        refused = [str(e) for e in errors if e is not None and "HELLO refused" in str(e)]
        if refused and case != "same_identity":
            fingerprints = {part.split("=", 1)[1] for part in refused[0].split()
                            if part.startswith("layout_fingerprint=")}
            if len(fingerprints) == 1 and "got session_contract_hash=" in refused[0]:
                raise HelloRefusedOnContract(refused[0])
        assert not hung and errors == [None, None], (
            f"handshake failed: errors={errors!r} hung={hung}\nsyncer log tail:\n{syncer_log[-2000:]}")
        assert all(cut is not None for cut in cuts)
        # Both islands start from the same initial cut.
        for name in initial.tensors:
            assert torch.equal(cuts[0].state.tensors[name], cuts[1].state.tensors[name])
    finally:
        for bridge in bridges:
            try:
                bridge.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


def test_fragment_binpack_matches_the_linear_scan():
    """The heap binpack picks exactly what the old linear ``min`` scan picked
    (least-filled fragment, lowest index on ties), so the layout fingerprint
    and every persisted layout hash are unchanged."""
    import random

    from yeto.protocol import layout_fingerprint
    from yeto.rl.core import CanonicalTensorSpec, build_rl_fragment_layout

    def linear(specs, count):
        ordered = sorted(specs, key=lambda spec: (-spec.numel, spec.name))
        bins, sizes = [[] for _ in range(count)], [0] * count
        for spec in ordered:
            index = min(range(count), key=lambda i: sizes[i])
            bins[index].append(spec.name)
            sizes[index] += spec.numel
        return bins

    rng = random.Random(1114)
    for count in (2, 3, 7, 64):
        shapes = [(rng.choice((1, 2, 4)), rng.choice((1, 3, 8, 8, 16))) for _ in range(300)]
        specs = [CanonicalTensorSpec(f"base_model.model.layers.{i}.mlp.proj.lora_A.weight",
                                     shape, "float32", shape[0] * shape[1])
                 for i, shape in enumerate(shapes)]
        layout = build_rl_fragment_layout(specs, count)
        assert [[name for name, _ in frag.tensors] for frag in layout.fragments] == linear(specs, count)
        assert len(layout_fingerprint(layout)) == 32
