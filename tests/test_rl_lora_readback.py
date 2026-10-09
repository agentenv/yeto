"""rl-infra-spec 3.5: a wrong LoRA adapter must not be admitted. The read-back body
carries ``lora:<adapter>:<module>:<layer>:A|B`` keys; LoRA mode without them fails closed."""
import asyncio
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles.publish import MilesPublisher, PublicationError
from yeto.rl.engine.ports import PublicationCause

BASE = {"rank0/model.w": "b0"}
LORA = {"rank0/lora:policy:qkv_proj:0:A": "a0", "rank0/lora:policy:qkv_proj:0:B": "b0x"}


def _pub(bodies, reference, *, lora=True, lazy=False, warmup_fails=()):
    """``lazy``: every body starts WITHOUT its lora:* keys (sglang assigns the GPU
    pool slot on the first forward that selects the adapter); the warm-up of
    engine i restores body i's keys. ``warmup_fails``: engine indices whose
    warm-up raises."""
    order = []
    served = [dict(b) for b in bodies]
    if lazy:
        served = [{k: v for k, v in b.items() if "lora:" not in k} for b in bodies]

    class Engine:
        def __init__(self, i):
            self.server_url = f"http://engine{i}:30000"
            self.i = i

        async def update_weight_version(self, token):
            self.v = token

        async def get_weight_version(self):
            return self.v

    class Controller:
        async def wait_cells_tracked(self, cells, timeout_seconds):
            pass

        async def start_update_weights(self, members, expected_epoch):
            return SimpleNamespace(rollout_engines=[Engine(i) for i, _ in enumerate(members)],
                                   snapshot_cell_id_to_hashes={c: "h" for c in members})

        async def end_update_weights(self, **_):
            pass

        async def abort_update_weights(self):
            pass

        async def check_weights(self, action):
            order.append("check")
            return [dict(b) for b in served]

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
    records = []
    pub.event_sink = lambda event, **f: records.append((event, f))

    async def warmup(engine, adapter, timeout_s):
        order.append(f"warmup:{engine.i}")
        assert adapter == "miles_lora" and timeout_s > 0
        if engine.i in warmup_fails:
            raise RuntimeError("connection refused")
        if engine.i < len(served):
            served[engine.i] = dict(bodies[engine.i])  # the pool slot now holds the adapter
        return {"status": 200, "completion_tokens": 1}

    pub.lora_warmup = warmup
    pub.records = records
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
    assert "admit" not in order
    # target-state evidence: the engine's pool holds the adapter and one A key differs
    (_, rb), = [r for r in pub.records if r[0] == "lora_readback"]
    assert rb["engines"][0]["lora_keys"] == 2 and rb["engines"][0]["lora_keys_differ"] == 1
    assert rb["blind"] == []


def test_matching_adapter_is_admitted():
    ref = BASE | LORA
    pub, order = _pub([dict(ref)], ref)
    _run(pub)
    assert order == ["warmup:0", "check", "admit"]


def test_lazily_loaded_adapter_is_warmed_up_before_the_read_back_then_admitted():
    """Defect 6 (GPU chain #2): a fresh member reports no lora:* keys until it serves a
    request. The warm-up runs BEFORE check_weights and the member is then admitted."""
    ref = BASE | LORA
    pub, order = _pub([dict(ref)], ref, lazy=True)
    _run(pub)
    assert order == ["warmup:0", "check", "admit"]
    kinds = [e for e, _ in pub.records]
    assert kinds.index("member_engines") < kinds.index("lora_warmup") < kinds.index("lora_readback")
    (_, me), = [r for r in pub.records if r[0] == "member_engines"]
    assert me["engine_urls"] == ["http://engine0:30000"] and me["target_members"] == ["engine:c2"]
    (_, w), = [r for r in pub.records if r[0] == "lora_warmup"]
    assert w["applied"] is True and w["adapter"] == "miles_lora" and w["engines"][0]["ok"]


def test_lazily_loaded_perturbed_adapter_is_still_refused_after_warm_up():
    ref = BASE | LORA
    pub, order = _pub([BASE | LORA | {"rank0/lora:policy:qkv_proj:0:A": "PERTURBED"}], ref,
                      lazy=True)
    with pytest.raises(PublicationError, match="adapter A/B differs") as ei:
        _run(pub)
    assert ei.value.cause == PublicationCause.PAYLOAD_MISMATCH and "admit" not in order


def test_failed_warm_up_is_unverifiable_not_a_mismatch_and_never_admitted():
    ref = BASE | LORA
    pub, order = _pub([dict(ref), dict(ref)], ref, lazy=True, warmup_fails=(1,))
    pub._reference = ("tok", dict(ref))
    with pytest.raises(PublicationError, match="warm-up failed") as ei:
        asyncio.run(pub._publish_members("tok", ["c2", "c3"], 1))
    assert ei.value.cause == PublicationCause.LORA_UNVERIFIABLE
    assert list(ei.value.engine_ids) == ["http://engine1:30000"]
    assert "check" not in order and "admit" not in order
    (_, w), = [r for r in pub.records if r[0] == "lora_warmup"]
    assert w["applied"] is False and w["engines"][1]["ok"] is False
    assert "connection refused" in w["engines"][1]["error"]


def test_lazy_adapter_without_warm_up_is_what_the_gpu_saw(monkeypatch):
    """Regression of the GPU observation: without the warm-up the read-back is blind
    (226 base keys, no lora:*) and admission fails closed."""
    ref = BASE | LORA
    pub, order = _pub([dict(ref)], ref, lazy=True)

    async def no_op(engine, adapter, timeout_s):
        return {"status": 200}

    pub.lora_warmup = no_op
    with pytest.raises(PublicationError, match="cannot be verified") as ei:
        _run(pub)
    assert ei.value.cause == PublicationCause.LORA_UNVERIFIABLE and "admit" not in order
    (_, rb), = [r for r in pub.records if r[0] == "lora_readback"]
    assert rb["blind"] == ["engine0"] and rb["engines"][0]["lora_keys"] == 0


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
    assert "admit" not in order


def test_non_lora_mode_is_unchanged():
    pub, order = _pub([dict(BASE)], BASE, lora=False)
    _run(pub)
    assert order == ["check", "admit"]  # no warm-up outside LoRA mode
