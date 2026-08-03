from types import SimpleNamespace

import pytest
import torch

from yeto.budget_finalization import (
    finalize_learner_budget,
    validate_consolidation_tape,
    validate_learner_budget_args,
)
from yeto.fragments import build_layout
from yeto.protocol import (
    DTYPE_F32,
    BcastFragment,
    FinalFragment,
    FinalManifest,
    PullRequest,
)
from yeto.tensor_io import pack_tensor


class _TerminalClient:
    dtype = DTYPE_F32
    finalization_timeout = 0.1

    def __init__(self, params):
        self.params = params
        self.report = None
        self.push = None
        self.ack = None
        self._updates = [
            BcastFragment(0, 7, pack_tensor(torch.tensor([2.0, 3.0]), DTYPE_F32))
        ]
        self._pulls = [PullRequest(0, 11, 1)]

    def send_budget_done(self, steps):
        self.report = steps
        return 3

    def wait_for_budget_restart(self, generation):
        self.restart_generation = generation

    def check_health(self):
        return None

    def drain_updates(self):
        updates, self._updates = self._updates, []
        return updates

    def drain_pulls(self):
        pulls, self._pulls = self._pulls, []
        return pulls

    def push_fragment(self, *args):
        self.push = args
        self.params_at_push = self.params["model.weight"].detach().clone()

    def wait_for_final_fragments(self):
        manifest = FinalManifest(11, (11,))
        fragment = FinalFragment(
            0,
            11,
            pack_tensor(torch.tensor([8.0, 9.0]), DTYPE_F32),
        )
        return manifest, [fragment]

    def acknowledge_finalization(self, manifest):
        self.ack = manifest


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"syncer": "none"}, "requires a syncer"),
        ({"learner_budget_steps": 0}, "must be positive"),
        ({"learner_budget_steps": 5}, "must equal --max-local-steps"),
        (
            {
                "learner_budget_steps": 0x1_0000_0000,
                "max_local_steps": 0x1_0000_0000,
            },
            "must fit the protocol c_steps u32",
        ),
        ({"tuning": "full"}, "requires --tuning lora"),
    ],
)
def test_budget_mode_is_limited_to_replicated_adapter_benchmarks(
    overrides, message
):
    values = {
        "learner_budget_steps": 4,
        "max_local_steps": 4,
        "syncer": "host:29400",
        "tuning": "lora",
        "shard": "ddp",
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        validate_learner_budget_args(SimpleNamespace(**values))


def test_budget_mode_accepts_only_the_explicit_supported_shape():
    for shard in ("ddp", "fsdp"):
        validate_learner_budget_args(
            SimpleNamespace(
                learner_budget_steps=4,
                max_local_steps=4,
                syncer="host:29400",
                tuning="lora",
                shard=shard,
            )
        )


def test_budget_helper_keeps_frozen_params_and_uses_reordered_cutoff_base():
    layout = build_layout([("model.weight", 2)], 1)
    params = {"model.weight": torch.tensor([5.0, 7.0])}
    client = _TerminalClient(params)

    manifest = finalize_learner_budget(
        client,
        layout,
        params,
        rank=0,
        world=1,
        device=torch.device("cpu"),
        target_steps=4,
        units=256,
    )

    assert client.report == 4
    assert client.restart_generation == 3
    assert torch.equal(client.params_at_push, torch.tensor([5.0, 7.0]))
    fid, step, attempt, base, local_step, c_steps, units, payload = client.push
    assert (fid, step, attempt, base, local_step, c_steps, units) == (
        0,
        11,
        1,
        7,
        4,
        4,
        256,
    )
    assert torch.equal(
        torch.frombuffer(bytearray(payload), dtype=torch.float32),
        torch.tensor([3.0, 4.0]),
    )
    assert manifest == FinalManifest(11, (11,))
    assert client.ack == manifest
    assert torch.equal(params["model.weight"], torch.tensor([8.0, 9.0]))


def test_terminal_tape_requires_every_budgeted_learner_and_fragment(tmp_path):
    tape = tmp_path / "tape.jsonl"
    tape.write_text(
        """{"step":8,"fragment":1,"responders":[]}
{"step":9,"fragment":0,"responders":[{"id":0,"c_steps":4},{"id":1,"c_steps":4}]}
{"step":10,"fragment":1,"responders":[{"id":0,"c_steps":4},{"id":1,"c_steps":4}]}
""",
        encoding="utf-8",
    )
    validate_consolidation_tape(
        tape, cutoff_step=8, fragments=2, learners=2, budget_steps=4
    )

    tape.write_text(
        '{"step":9,"fragment":0,"responders":[{"id":0,"c_steps":4}]}\n'
        '{"step":10,"fragment":1,"responders":[{"id":0,"c_steps":4}]}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="every learner"):
        validate_consolidation_tape(
            tape, cutoff_step=8, fragments=2, learners=2, budget_steps=4
        )
