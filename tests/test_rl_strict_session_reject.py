"""S17 N14: in strict mode (--max-base-lag 0) the real Rust syncer refuses a HELLO
whose session contract differs from the running session on that connection only.

Two verl-identity islands sync; mid-run a Miles-identity island (same layout,
other backend identity, so another session contract hash) dials in.  It gets
MSG_ERROR naming both hashes and exits as a strict RL failure
(``StrictRlInvariantError``, metric ``layout_hash_mismatch``); the syncer and the
two verl islands finish every round and the syncer exits 0.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from yeto.protocol import DTYPE_F32, SessionRejectedError, SyncerClient, layout_fingerprint
from yeto.rl.core import StrictRlInvariantError
from yeto.rl.engine.backend_identity import (
    island_contract_sha256,
    lr_schedule_sha256,
    session_contract_hash,
)
from yeto.tensor_io import pack_tensor

from test_rl_integration import _layout, _port, _push, _wait_item, syncer_binary  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
VERL_ID = "3b9e60e5a2c554a2826f6cef83fdeecabfc1723d6663c4d5fd23b01e08a555e2"
MILES_ID = "9d5696a3d3b6e6d802115ef3deb970b5e4d9206d1751d71f9075a849d2909f8d"
# S17 N17: same backend, different LR schedule (e.g. --rl-lr-schedule constant vs linear)
LR_LINEAR = lr_schedule_sha256("linear", 3, 1e-5)
LR_CONSTANT = lr_schedule_sha256("constant", 3, 1e-5)
VERL_LINEAR = island_contract_sha256(VERL_ID, LR_LINEAR)
VERL_CONSTANT = island_contract_sha256(VERL_ID, LR_CONSTANT)


def _start(binary, port, checkpoint, rounds, *, learners, event_tape):
    """Strict syncer (``--max-base-lag 0``); fresh run, so no ``--resume``."""
    return subprocess.Popen(
        [str(binary), "--port", str(port), "--learners", str(learners), "--quorum", str(learners),
         "--grace-ms", "0", "--pipeline", "1", "--sync-interval-steps", "0",
         "--delta-correction", "none", "--total-steps", str(rounds), "--outer-lr", "1",
         "--outer-momentum", "0", "--quorum-timeout-s", "10", "--checkpoint-path", str(checkpoint),
         "--checkpoint-every", "1", "--max-base-lag", "0", "--learner-weight", "equal",
         "--event-tape", str(event_tape)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _contract(identity: str) -> bytes:
    return session_contract_hash(layout_fingerprint(_layout()), identity)


def _verl(port, learner_id, identity=VERL_ID, compat_group="nvidia-h100"):
    client = SyncerClient(("127.0.0.1", port), learner_id, _layout(), dtype=DTYPE_F32,
                          num_streams=0, connect_timeout=10,
                          session_contract_hash=_contract(identity), compat_group=compat_group)
    client.start()
    return client


MILES_ISLAND = textwrap.dedent("""
    import sys, time
    from yeto.protocol import DTYPE_F32, SyncerClient, layout_fingerprint
    from yeto.rl.engine.backend_identity import session_contract_hash
    sys.path.insert(0, {tests!r})
    from test_rl_integration import _layout
    contract = session_contract_hash(layout_fingerprint(_layout()), {miles!r})
    client = SyncerClient(("127.0.0.1", {port}), 1, _layout(), dtype=DTYPE_F32, num_streams=0,
                          connect_timeout=10, session_contract_hash=contract, max_reconnects=0,
                          compat_group={group!r})
    try:
        client.start()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            client.check_health()
            time.sleep(0.05)
        print("NOT REFUSED", flush=True)
        sys.exit(0)
    finally:
        client.close()
""")


# decoupling 7.7c: same backend, other card type (H200 or Ascend island in an
# H100 session); the identity hash differs too because compat_group is in it.
H200_ID = "c" * 64


@pytest.mark.parametrize("good,intruder,group", [(VERL_ID, MILES_ID, "nvidia-h100"),
                                                 (VERL_LINEAR, VERL_CONSTANT, "nvidia-h100"),
                                                 (VERL_ID, H200_ID, "nvidia-h200"),
                                                 (VERL_ID, H200_ID, "ascend-910b")],
                         ids=["backend-identity", "lr-schedule", "compat-h200", "compat-ascend"])
def test_mismatched_hello_is_refused_without_stopping_the_session(syncer_binary, tmp_path,
                                                                    good, intruder, group):
    port = _port()
    tape = tmp_path / "syncer.jsonl"
    process = _start(syncer_binary, port, tmp_path / "state.ckpt", rounds=3, learners=2,
                     event_tape=tape)
    c0 = c1 = None
    try:
        c0, c1 = _verl(port, 0, good), _verl(port, 1, good)
        c0.send_init(0, pack_tensor(torch.zeros(2), DTYPE_F32))
        for c in (c0, c1):
            assert _wait_item(c.drain_updates).version == 0
            assert _wait_item(c.drain_pulls).global_step == 1
        _push(c0, 1, 0, [1, 1])
        _push(c1, 1, 0, [3, 3])
        for c in (c0, c1):
            assert _wait_item(c.drain_updates).version == 1
            assert _wait_item(c.drain_pulls).global_step == 2

        # A Miles island dials in mid-run with learner id 1 (as a mis-wired relaunch would).
        miles = subprocess.run(
            [sys.executable, "-c", MILES_ISLAND.format(tests=str(ROOT / "tests"), miles=intruder,
                                                         port=port, group=group)],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PYTHONPATH": f"{ROOT}:{os.environ.get('PYTHONPATH', '')}"})
        (tmp_path / "miles.stderr").write_text(miles.stderr)
        assert miles.returncode == 1, miles.stdout + miles.stderr
        assert "StrictRlInvariantError: syncer refused this island" in miles.stderr
        if group == "nvidia-h100":
            assert "expected session_contract_hash=" + _contract(good).hex() in miles.stderr
            assert "got session_contract_hash=" + _contract(intruder).hex() in miles.stderr
        else:
            assert f"兼容组不同：nvidia-h100 对 {group}，容差未标定" in miles.stderr
        assert process.poll() is None, "syncer must keep running"

        # The two verl islands finish the remaining rounds unaffected.
        for step, (a, b) in ((2, ([1, 1], [1, 1])), (3, ([2, 0], [0, 2]))):
            _push(c0, step, step - 1, a)
            _push(c1, step, step - 1, b)
            if step < 3:
                for c in (c0, c1):
                    assert _wait_item(c.drain_updates).version == step
                    assert _wait_item(c.drain_pulls).global_step == step + 1
        finals = [c.wait_for_final_fragments(timeout=15) for c in (c0, c1)]
        for (manifest, _), c in zip(finals, (c0, c1)):
            assert manifest.global_step == 3
            c.acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0
        events = [json.loads(line) for line in tape.read_text().splitlines() if line.strip()]
        assert not [e for e in events if e.get("event") == "rl_strict_failure"]
    finally:
        for c in (c0, c1):
            if c is not None:
                c.close()
        if process.poll() is None:
            process.kill()
            process.wait()
        out = process.stdout.read() if process.stdout else ""
        (tmp_path / "syncer.log").write_text(out)
        print(out[-3000:])


def test_client_maps_session_refusal_to_strict_failure():
    client = SyncerClient(("127.0.0.1", 1), 0, _layout(), dtype=DTYPE_F32, num_streams=0)
    client._dispatch_live(client._gen, 10, b"session mismatch (HELLO refused, session keeps "
                                           b"running): expected session_contract_hash=aa")
    assert isinstance(client._err, SessionRejectedError)
    with pytest.raises(StrictRlInvariantError) as info:
        client.check_health()
    assert info.value.metric == "layout_hash_mismatch"
    # Other syncer errors keep the old generic classification.
    other = SyncerClient(("127.0.0.1", 1), 0, _layout(), dtype=DTYPE_F32, num_streams=0)
    other._dispatch_live(other._gen, 10, b"duplicate connection generation 5 for learner 0")
    with pytest.raises(RuntimeError, match="syncer connection failed") as info:
        other.check_health()
    assert not isinstance(info.value, StrictRlInvariantError)
