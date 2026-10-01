"""rl-infra-spec 3.5: a wrong LoRA adapter must not be admitted. The read-back body
carries ``lora:<adapter>:<module>:<layer>:A|B`` keys; LoRA mode without them fails closed."""
import asyncio
from types import SimpleNamespace

import pytest

from yeto.rl.engine.miles_adapter.publish import MilesPublisher, PublicationError
from yeto.rl.engine.ports import PublicationCause

BASE = {"rank0/model.w": "b0"}
LORA = {"rank0/lora:policy:qkv_proj:0:A": "a0", "rank0/lora:policy:qkv_proj:0:B": "b0x"}


def _pub(bodies, reference, *, lora=True):
    order = []

    class Engine:
        async def update_weight_version(self, token):
            self.v = token

        async def get_weight_version(self):
            return self.v

    class Controller:
        async def wait_cells_tracked(self, cells, timeout_seconds):
            pass

        async def start_update_weights(self, members, expected_epoch):
            return SimpleNamespace(rollout_engines=[Engine() for _ in members],
                                   snapshot_cell_id_to_hashes={c: "h" for c in members})

        async def end_update_weights(self, **_):
            pass

        async def abort_update_weights(self):
            pass

        async def check_weights(self, action):
            return [dict(b) for b in bodies]

        async def admit_cells(self, cells, **_):
            order.append("admit")

        async def start_commit_weight_version(self, **_):
            pass

        async def end_commit_weight_version(self):
            pass

    async def update_weights(*a, **k):
        pass

    args = SimpleNamespace(lora_rank=8) if lora else SimpleNamespace()
    pub = MilesPublisher(args=args, actor_model=None, rollout_executor=None,
                         inference_controller=Controller(), update_weights=update_weights,
                         flatten_checksums=lambda raw: list(raw))
    pub._reference = ("tok", dict(reference))
    pub.last_engine_checksums = {}
    return pub, order


def _run(pub):
    asyncio.run(pub._publish_members("tok", ["c2"], 1))


def test_perturbed_adapter_is_payload_mismatch_and_never_admitted():
    ref = BASE | LORA
    pub, order = _pub([BASE | LORA | {"rank0/lora:policy:qkv_proj:0:A": "PERTURBED"}], ref)
    with pytest.raises(PublicationError, match="adapter A/B differs") as ei:
        _run(pub)
    assert ei.value.cause == PublicationCause.PAYLOAD_MISMATCH
    assert list(ei.value.engine_ids) == ["engine0"]
    assert order == []


def test_matching_adapter_is_admitted():
    ref = BASE | LORA
    pub, order = _pub([dict(ref)], ref)
    _run(pub)
    assert order == ["admit"]


@pytest.mark.parametrize("bodies,ref", [
    ([BASE], BASE | LORA),            # engines blind (stock WeightChecker)
    ([BASE | LORA], BASE),            # reference blind
    ([BASE], BASE),                   # both
])
def test_lora_mode_without_adapter_keys_fails_closed(bodies, ref):
    pub, order = _pub(bodies, ref)
    with pytest.raises(PublicationError, match="cannot be verified") as ei:
        _run(pub)
    assert ei.value.cause == PublicationCause.LORA_UNVERIFIABLE
    assert ei.value.cause.value == "lora_unverifiable"
    assert order == []


def test_non_lora_mode_is_unchanged():
    pub, order = _pub([dict(BASE)], BASE, lora=False)
    _run(pub)
    assert order == ["admit"]
