import json
import subprocess
import sys

from yeto.rl.contracts import InferencePublicationManifest, LocalStepReceipt
from yeto.rl.engine import ports


def test_ports_import_is_light():
    code = (
        "import sys, json; import yeto.rl.engine.ports; "
        "print(json.dumps([m for m in ('torch','ray','miles') "
        "if any(k == m or k.startswith(m + '.') for k in sys.modules)]))"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert json.loads(out.stdout) == []


def test_ports_reuse_contract_types():
    assert ports.LocalStepReceipt is LocalStepReceipt
    assert ports.InferencePublicationManifest is InferencePublicationManifest
    for name in ports.__all__:
        assert hasattr(ports, name)


def test_protocol_shapes():
    class Pool:
        def generate(self, rollout_id): ...
        def abort(self): ...
        def members(self): ...

    class Trainer:
        def train_step(self, batch): ...
        def onload(self): ...
        def offload(self): ...

    assert isinstance(Pool(), ports.RolloutPool)
    assert isinstance(Trainer(), ports.TrainerGroup)
    assert not isinstance(Pool(), ports.TrainerGroup)
    # E1-E3 verbs are reserved, not required in R0.
    assert not hasattr(ports.RolloutPool, "add_engines")
    assert not hasattr(ports.TrainerGroup, "save_cut")
    assert not hasattr(ports.Placement, "reconfigure")


def test_rollout_handle_metadata_mismatch():
    good = ports.GroupMetadata("g0", ("s0",), "yeto:1:" + "a" * 64, 0.5, 0.1, 10)
    bad = ports.GroupMetadata("g1", ("s1",), "yeto:0:" + "b" * 64, 0.5, 0.1, 10)
    handle = ports.RolloutBatchHandle(1, 1, "a" * 64, (good, bad), 2, 0, payload=object())
    assert handle.mismatched_groups("yeto:1:" + "a" * 64) == (bad,)
