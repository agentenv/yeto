"""S28 G-120D: ``copy_megatron_model_to_cpu`` makes ONE host copy per buffer.

verl fully_async saves the model to the host every round (``save_model_to_cpu``) to
recompute old_log_prob with the version-1 parameters. The stock body copied each
buffer with ``.cpu().clone().pin_memory()``, so three host copies existed at once:
about 230 GB per node at the start of every round in run g120d-lh1, which the
host-memory guard stopped. The patch (``patch_verl.PINNED_COPY``) allocates one pinned
tensor and copies into it. These tests pin that the copy is bitwise identical and
that the patched body allocates exactly the source size once.
"""

import pytest

from yeto.rl.adapters.verl import patch_verl

torch = pytest.importorskip("torch")

# The stock verl body (verl/utils/megatron_utils.py, verl fork on npu1), anchor lines verbatim.
STOCK = '''
class DDP:
    def __init__(self, buffers, expert_parallel_buffers):
        self.buffers = buffers
        self.expert_parallel_buffers = expert_parallel_buffers


@torch.no_grad()
def copy_megatron_model_to_cpu(models):
    cpu_state = {}

    for model_idx, model_chunk in enumerate(models):
        if isinstance(model_chunk, DDP):
            # Handle DDP-wrapped models
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            buffer_states = []

            for buffers in model_chunk_all_buffers:
                buffer_list = []
                for buffer in buffers:
                    buffer_state = {}

                    # Copy parameter data to CPU
                    if buffer.param_data.storage().size() > 0:
                        buffer_state["param_data"] = buffer.param_data.data.cpu().clone().pin_memory()
                    else:
                        buffer_state["param_data"] = buffer.param_data.cpu_data.clone().pin_memory()

                    buffer_list.append(buffer_state)
                buffer_states.append(buffer_list)

            cpu_state[f"model_chunk_{model_idx}"] = {"buffer_states": buffer_states, "is_ddp": True}
        else:
            # Handle non-DDP models (ref module)
            model_state = {}
            for name, param in model_chunk.named_parameters():
                param_state = {"data": param.data.cpu().clone().pin_memory()}
                model_state[name] = param_state

            cpu_state[f"model_chunk_{model_idx}"] = {"model_state": model_state, "is_ddp": False}

    return cpu_state
'''


def _patched_source() -> str:
    text = STOCK
    for target, anchor, hook, mode in patch_verl.PINNED_COPY:
        if target != patch_verl.PINNED_COPY_TARGET:
            continue  # engine_workers.py: tested by the save hook test below
        assert text.count(anchor) == 1, anchor
        text = patch_verl._patched(text, anchor, hook, mode)
    return text


def _load(source: str) -> dict:
    scope: dict = {"torch": torch}
    exec(compile(source, "megatron_utils_fixture.py", "exec"), scope)
    return scope


class _Buffer:
    def __init__(self, data):
        self.param_data = data


def _models(scope):
    gen = torch.Generator().manual_seed(20261027)
    dense = [_Buffer(torch.randn(4096, generator=gen).to(torch.bfloat16)),
             _Buffer(torch.randn(1000, generator=gen))]
    expert = [_Buffer(torch.randn(777, generator=gen).to(torch.bfloat16))]
    ref = torch.nn.Linear(16, 8)
    return [scope["DDP"](dense, expert), ref]


def _flat(state):
    out = []
    for key in sorted(state):
        chunk = state[key]
        if chunk["is_ddp"]:
            out += [b["param_data"] for group in chunk["buffer_states"] for b in group]
        else:
            out += [chunk["model_state"][n]["data"] for n in sorted(chunk["model_state"])]
    return out


def test_patch_is_in_the_megatron_profile_and_idempotent():
    assert all(p in patch_verl.PROFILES["megatron"] for p in patch_verl.PINNED_COPY)
    patched = _patched_source()
    code = [line for line in patched.splitlines() if not line.lstrip().startswith("#")]
    assert not any(".clone().pin_memory()" in line for line in code)
    assert patched.count("_yeto_pinned_host_copy(") == 4  # def + three call sites
    for target, _anchor, hook, _mode in patch_verl.PINNED_COPY:
        if target == patch_verl.PINNED_COPY_TARGET:
            assert patch_verl._applied(patched, hook)


def test_pinned_copy_is_bitwise_identical_to_the_stock_copy(monkeypatch):
    # CPU-only torch cannot pin; make the stock pin_memory() a plain copy so it runs here.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self.clone())
    stock, patched = _load(STOCK), _load(_patched_source())
    models = _models(stock)
    want = _flat(stock["copy_megatron_model_to_cpu"](models))
    models[0].__class__ = patched["DDP"]
    got = _flat(patched["copy_megatron_model_to_cpu"](models))
    assert len(got) == len(want) == 5
    for a, b in zip(got, want):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        assert a.device.type == "cpu"
    # A copy, not a view of the live parameter: changing the source leaves it alone.
    source = models[0].buffers[0].param_data
    before = got[0].clone()
    source.add_(1)
    assert torch.equal(got[0], before)


def test_patched_copy_allocates_the_source_size_exactly_once(monkeypatch):
    allocated: list[int] = []
    real_empty = torch.empty

    def counting_empty(*args, **kwargs):
        out = real_empty(*args, **kwargs)
        allocated.append(out.numel() * out.element_size())
        return out

    def forbidden(name):
        def call(self, *a, **k):
            raise AssertionError(f"patched body must not call Tensor.{name}")
        return call

    scope = _load(_patched_source())
    models = _models(scope)
    sources = [b.param_data for b in models[0].buffers + models[0].expert_parallel_buffers]
    sources += [p.data for _, p in sorted(models[1].named_parameters())]
    want = sum(t.numel() * t.element_size() for t in sources)

    monkeypatch.setattr(torch, "empty", counting_empty)
    monkeypatch.setattr(torch.Tensor, "clone", forbidden("clone"))
    monkeypatch.setattr(torch.Tensor, "pin_memory", forbidden("pin_memory"))
    scope["copy_megatron_model_to_cpu"](models)
    # One allocation per tensor, each exactly the tensor's size: peak = 1x, not 3x.
    assert len(allocated) == len(sources)
    assert sum(allocated) == want


def test_stock_copy_allocated_twice_per_tensor_on_the_host():
    # Pins why the patch exists: clone() and pin_memory() each allocate a full copy
    # (and .cpu() a third one when the source is on the device).
    calls = {"clone": 0, "pin_memory": 0}
    real_clone = torch.Tensor.clone

    def clone(self, *a, **k):
        calls["clone"] += 1
        return real_clone(self, *a, **k)

    def pin(self):
        calls["pin_memory"] += 1
        return real_clone(self)

    scope = _load(STOCK)
    models = _models(scope)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.Tensor, "clone", clone)
        mp.setattr(torch.Tensor, "pin_memory", pin)
        scope["copy_megatron_model_to_cpu"](models)
    assert calls["clone"] >= 5 and calls["pin_memory"] == 5


def test_a_second_save_refills_the_first_saves_host_tensors_without_allocating(monkeypatch):
    # trigger_parameter_sync_step=1: every round saves under key 1 again. The second
    # copy must land in the first copy's tensors (no second 80 GB/node beside it).
    scope = _load(_patched_source())
    models = _models(scope)
    first = scope["copy_megatron_model_to_cpu"](models)
    first_ptrs = [t.data_ptr() for t in _flat(first)]
    for buffer in models[0].buffers + models[0].expert_parallel_buffers:
        buffer.param_data.mul_(-3)  # the next round's parameters
    with torch.no_grad():
        for p in models[1].parameters():
            p.mul_(-3)
    allocated: list[int] = []
    real_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **k: allocated.append(1) or real_empty(*a, **k))
    second = scope["copy_megatron_model_to_cpu"](models, reuse=first)
    monkeypatch.undo()
    assert allocated == []
    assert [t.data_ptr() for t in _flat(second)] == first_ptrs
    want = [b.param_data for b in models[0].buffers + models[0].expert_parallel_buffers]
    want += [p.data for _, p in sorted(models[1].named_parameters())]
    for a, b in zip(_flat(second), want):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_reuse_falls_back_to_a_fresh_tensor_when_a_shape_changed():
    scope = _load(_patched_source())
    models = _models(scope)
    first = scope["copy_megatron_model_to_cpu"](models)
    models[0].buffers[0].param_data = torch.ones(10, dtype=torch.bfloat16)  # rebuilt buffer
    second = scope["copy_megatron_model_to_cpu"](models, reuse=first)
    got = _flat(second)
    assert got[0].shape == (10,) and torch.equal(got[0], torch.ones(10, dtype=torch.bfloat16))
    assert got[1].data_ptr() == _flat(first)[1].data_ptr()  # the others are still reused


def test_save_model_to_cpu_hook_passes_the_previous_save_for_megatron_only():
    hook = patch_verl.PINNED_SAVE_HOOK
    assert patch_verl.PINNED_SAVE_ANCHOR.strip() in hook  # the stock call stays for FSDP
    assert "reuse=previous" in hook and "copy_megatron_model_to_cpu" in hook

    calls = []

    def copy_megatron_model_to_cpu(module, reuse=None):
        calls.append(reuse)
        return {"saved": len(calls)}

    class Worker:
        copy_handler = staticmethod(copy_megatron_model_to_cpu)
        actor = type("A", (), {"engine": type("E", (), {"module": object()})()})()

        def save_model_to_cpu(self, n):
            if not hasattr(self, "cpu_saved_models"):
                self.cpu_saved_models = {}
            exec(compile("if True:\n" + hook, "hook.py", "exec"), {}, {"self": self, "n": n})

    w = Worker()
    w.save_model_to_cpu(1)
    w.save_model_to_cpu(1)
    assert calls == [None, {"saved": 1}]
    assert w.cpu_saved_models[1] == {"saved": 2}
