"""S27 G-EN: the verl LoRA fully_async path and the zmq slow-joiner fix.

verl's NCCL checkpoint engine sends each bucket's metadata over zmq PUB/SUB.
Every weight update rebuilds the process group, and the unpatched consumer
opens a NEW SUB socket and subscribes just before the first bucket; a SUB that
is not subscribed yet when the publisher sends drops the message and its
consumer waits in ``recv_string`` forever (S27 G-DC, s27-dc-smoke3 on the
Megatron island; VERL-NEMOTRON §6).  The LoRA fully_async island publishes
through the same engine (``fully_async_round`` sets
``checkpoint_engine.backend=nccl``; ``yeto_push_weights`` calls
``CheckpointEngineManager.update_weights``, which runs ``init_process_group`` ->
``_connect_zmq_client`` on every update), so the default verl image (the
``fsdp`` patch profile) carries G-DC's fix: one SUB socket per publisher,
and one wait after the first connect (``patch_verl.ZMQ_SUB``, a4e40df4;
the fsdp profile takes it since S27 G-EK).

The first test patches copies of the pinned verl fork's files.  The second
runs the consumer's ``_connect_zmq_client`` (unpatched and patched source of
the fork, compiled alone) against a real zmq PUB socket over three updates;
it needs pyzmq (in the verl image, not in every test venv) and skips without.
"""
from __future__ import annotations

import ast
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.verl import fully_async_round, patch_verl

VERL_FORK = Path("/home/michael/work/verl-fork-acad9875")
needs_fork = pytest.mark.skipif(not (VERL_FORK / patch_verl.ZMQ_TARGET).is_file(), reason="no verl fork checkout")
TOPIC = "bucket_metadata"


def _patched_fork(root: Path, profile: str = "fsdp") -> Path:
    for rel in {target for target, *_ in patch_verl.PROFILES[profile]}:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(VERL_FORK / rel, root / rel)
    patch_verl.apply(root, profile)
    patch_verl.apply(root, profile)  # idempotent
    return root


@needs_fork
def test_the_default_verl_image_profile_carries_the_zmq_sub_fix_next_to_the_lora_wire(tmp_path):
    assert set(patch_verl.ZMQ_SUB) <= set(patch_verl.PROFILES["fsdp"])
    root = _patched_fork(tmp_path)
    engine = (root / patch_verl.ZMQ_TARGET).read_text()
    assert engine.count("_yeto_sub_key") == 2 and engine.count("time.sleep(1.0)  # yeto") == 1
    compile(engine, patch_verl.ZMQ_TARGET, "exec")
    # the fully_async LoRA wire (patches 3 and 4) stays in the same image
    assert patch_verl.SEND_HOOK in (root / patch_verl.SEND_TARGET).read_text()
    assert patch_verl.RECV_HOOK in (root / patch_verl.RECV_TARGET).read_text()
    # and the fully_async island publishes through the NCCL engine
    overrides = fully_async_round.fully_async_run_overrides([], 1, groups_per_round=4, rounds=2)
    assert "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl" in overrides


def _consumer_connect(source: str):
    """``NCCLCheckpointEngine._connect_zmq_client`` of ``source``, compiled alone."""

    zmq = pytest.importorskip("zmq")
    tree = ast.parse(source)
    engine = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NCCLCheckpointEngine")
    function = next(node for node in engine.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_connect_zmq_client")
    namespace = {"zmq": zmq, "time": time, "MasterMetadata": object,
                 "is_valid_ipv6_address": lambda address: ":" in address}
    exec(compile(ast.Module(body=[function], type_ignores=[]), patch_verl.ZMQ_TARGET, "exec"), namespace)
    return namespace["_connect_zmq_client"]


def _updates(connect, rounds: int = 3) -> list[dict]:
    """``rounds`` weight updates as the NCCL engine runs them: the consumer
    connects (``init_process_group``), then the publisher sends the first
    bucket's metadata at once (``BroadcastOperation._run``, rank 0)."""

    zmq = pytest.importorskip("zmq")
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://127.0.0.1:*")
    port = int(publisher.getsockopt_string(zmq.LAST_ENDPOINT).rsplit(":", 1)[1])
    consumer = SimpleNamespace(is_master=False, topic=TOPIC)
    metadata = SimpleNamespace(zmq_ip="127.0.0.1", zmq_port=port)
    out = []
    try:
        for update in range(rounds):
            started = time.monotonic()
            connect(consumer, metadata)
            connect_s = time.monotonic() - started
            publisher.send_string(TOPIC, flags=zmq.SNDMORE)
            publisher.send_pyobj({"update": update})
            received = None
            if consumer.socket.poll(3000):
                consumer.socket.recv_string()
                received = consumer.socket.recv_pyobj()["update"]
            out.append({"socket": consumer.socket, "connect_s": connect_s, "received": received})
    finally:
        publisher.close(0)
        for record in out:
            record["socket"].close(0)
    return out


@needs_fork
def test_the_patched_consumer_keeps_its_subscription_across_updates(tmp_path):
    unpatched = _consumer_connect((VERL_FORK / patch_verl.ZMQ_TARGET).read_text())
    patched = _consumer_connect((_patched_fork(tmp_path) / patch_verl.ZMQ_TARGET).read_text())
    # verl as pinned: a new SUB socket on every update (the race G-DC caught)
    fresh = _updates(unpatched)
    assert len({id(record["socket"]) for record in fresh}) == 3
    # patched: one socket, subscribed before the first send; every update's metadata arrives
    kept = _updates(patched)
    assert len({id(record["socket"]) for record in kept}) == 1
    assert [record["received"] for record in kept] == [0, 1, 2]
    assert kept[0]["connect_s"] >= 1.0 and max(record["connect_s"] for record in kept[1:]) < 0.5
