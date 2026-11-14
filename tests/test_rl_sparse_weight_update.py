"""Sparse weight updates of the ``sglang-server`` adapter (only the changed
elements are written into the running engine). CPU only: the engine is a fake
SGLang whose ranks are small torch models with SGLang-style weight loaders
(tensor-parallel slices, fused q/k/v and gate/up, stacked and transposed
experts, a padded vocabulary), and the custom weight loader runs on them as it
would inside SGLang."""

from __future__ import annotations

import base64
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from yeto.rl.adapters.sampler_engines import sglang_sparse_loader, sparse_update
from yeto.rl.adapters.sampler_engines.server import ServerConfig
from yeto.rl.adapters.sampler_engines.sglang_server import SGLangOptions, SGLangServerEngine
from yeto.rl.engine import release_delta as rd
from yeto.rl.engine.sampler_engine import SamplerEngineError

H, I, E, VOCAB, PAD_VOCAB = 8, 6, 3, 10, 12
L = "model.layers.0."
L1 = "model.layers.1."  # a layer whose only tensor the loader transforms


def _shapes() -> dict[str, tuple[int, ...]]:
    shapes = {L + "self_attn.q_proj.weight": (H, H), L + "self_attn.k_proj.weight": (4, H),
              L + "self_attn.v_proj.weight": (4, H), L + "self_attn.o_proj.weight": (H, H),
              L1 + "mlp.gate.scale": (4,), "model.norm.weight": (H,), "model.embed_tokens.weight": (VOCAB, H)}
    for e in range(E):
        shapes[L + f"mlp.experts.{e}.gate_proj.weight"] = (I, H)
        shapes[L + f"mlp.experts.{e}.up_proj.weight"] = (I, H)
        shapes[L + f"mlp.experts.{e}.down_proj.weight"] = (H, I)
    return shapes


class RankModel(torch.nn.Module):
    """One tensor-parallel rank of a toy decoder layer with SGLang-style loaders."""

    def __init__(self, rank: int, tp: int, *, shuffle_after_load: bool = False) -> None:
        super().__init__()
        self.rank, self.tp, self.shuffle = rank, tp, shuffle_after_load
        P = torch.nn.Parameter
        bf = dict(dtype=torch.bfloat16)
        self.qkv = P(torch.zeros((H + 8) // tp, H, **bf), requires_grad=False)
        self.o = P(torch.zeros(H, H // tp, **bf), requires_grad=False)
        self.w13 = P(torch.zeros(E, 2 * I // tp, H, **bf), requires_grad=False)
        self.w2t = P(torch.zeros(E, I // tp, H, **bf), requires_grad=False)  # stored transposed
        self.scale = P(torch.zeros(4, **bf), requires_grad=False)
        self.norm = P(torch.zeros(H, **bf), requires_grad=False)
        self.embed = P(torch.zeros(PAD_VOCAB // tp, H, **bf), requires_grad=False)

    def named_parameters(self, *args: Any, **kw: Any):  # SGLang-like names
        names = {"qkv": L + "self_attn.qkv_proj.weight", "o": L + "self_attn.o_proj.weight",
                 "w13": L + "mlp.experts.w13_weight", "w2t": L + "mlp.experts.w2_weight",
                 "scale": L1 + "mlp.gate.scale", "norm": "model.norm.weight",
                 "embed": "model.embed_tokens.weight"}
        for attr, name in names.items():
            yield name, getattr(self, attr)

    def load_weights(self, weights) -> None:
        r, tp = self.rank, self.tp
        for name, w in weights:
            short = name[len(L):] if name.startswith(L) else name
            if short.startswith("self_attn.") and short.split(".")[1] in ("q_proj", "k_proj", "v_proj"):
                which = short.split(".")[1]
                rows = {"q_proj": H, "k_proj": 4, "v_proj": 4}[which] // tp
                offset = {"q_proj": 0, "k_proj": H // tp, "v_proj": H // tp + 4 // tp}[which]
                self.qkv.data.narrow(0, offset, rows).copy_(w.narrow(0, r * rows, rows))
            elif short == "self_attn.o_proj.weight":
                self.o.data.copy_(w.narrow(1, r * (H // tp), H // tp))
            elif short.startswith("mlp.experts."):
                e, kind = int(short.split(".")[2]), short.split(".")[3]
                rows = I // tp
                if kind in ("gate_proj", "up_proj"):
                    at = 0 if kind == "gate_proj" else rows
                    self.w13.data[e].narrow(0, at, rows).copy_(w.narrow(0, r * rows, rows))
                else:
                    self.w2t.data[e].copy_(w.narrow(1, r * rows, rows).t())
            elif name == L1 + "mlp.gate.scale":
                self.scale.data.copy_(w * 2)  # a value transform: cannot be patched by element
            elif name == "model.norm.weight":
                self.norm.data.copy_(w)
            elif name == "model.embed_tokens.weight":
                per = VOCAB // tp
                self.embed.data.narrow(0, 0, per).copy_(w.narrow(0, r * per, per))

    def load_from_disk(self, path: Path) -> None:
        tensors = {}
        for f in sorted(Path(path).glob("*.safetensors")):
            tensors.update(load_file(f))
        self.load_weights(tensors.items())
        if self.shuffle:  # a post-load layout change the probe does not see
            self.norm.data.copy_(self.norm.data.flip(0))


class FakeSGLangRanks:
    """The HTTP calls of the adapter, served by ``tp`` rank models."""

    def __init__(self, tp: int, **kw: Any) -> None:
        self.ranks = [RankModel(r, tp, **kw) for r in range(tp)]
        self.calls: list[tuple[str, dict]] = []
        self.version = "default"
        self.paused = False
        self.on_tensor_update = None  # test hook after the loader ran
        # like the pinned fork: /update_weights_from_tensor only inside a
        # begin/end_weight_update session (False: upstream, no session endpoints)
        self.sessions = True
        self.session_open = False

    def __call__(self, method: str, url: str, body: Any, timeout_s: float):
        path = url.split(":", 2)[-1].split("/", 1)[-1]
        path = "/" + path
        self.calls.append((path, body))
        if path == "/health":
            return 200, {}
        if path == "/pause_generation":
            self.paused = True
            return 200, {"status": "ok"}
        if path == "/continue_generation":
            self.paused = False
            return 200, {"status": "ok"}
        if path == "/update_weights_from_disk":
            for model in self.ranks:
                model.load_from_disk(Path(body["model_path"]))
            self.version = body.get("weight_version") or self.version
            return 200, {"success": True, "message": "ok", "num_paused_requests": 0}
        if self.sessions and path == "/begin_weight_update":
            assert not self.session_open, "begin_weight_update while a session is open"
            self.session_open = True
            return 200, {"success": True, "message": "Success"}
        if self.sessions and path == "/end_weight_update":
            assert self.session_open, "end_weight_update without begin_weight_update"
            self.session_open = False
            return 200, {"success": True, "message": "Success"}
        if path == "/update_weights_from_tensor":
            assert not self.sessions or self.session_open, "update_weights_from_tensor outside a session"
            assert body["load_format"] == sglang_sparse_loader.LOADER
            assert body["flush_cache"] is False
            for rank, payload in enumerate(body["serialized_named_tensors"]):
                named = pickle.loads(base64.b64decode(payload))
                # the fork sends a name with ":" to the LoRA stash, never to the loader
                assert ":" not in named[0][0], named[0][0]
                assert sglang_sparse_loader.parse_command(named[0][0])["rank"] == rank
                sglang_sparse_loader.load(self.ranks[rank], named)
            if self.on_tensor_update:
                self.on_tensor_update(self)
            if body.get("weight_version") is not None:
                self.version = body["weight_version"]
            return 200, {"success": True, "message": "Success"}
        return 404, {}

    def tensor_calls(self, op: str) -> list[list[dict]]:
        out = []
        for path, body in self.calls:
            if path == "/update_weights_from_tensor":
                cmds = [sglang_sparse_loader.parse_command(pickle.loads(base64.b64decode(p))[0][0])
                        for p in body["serialized_named_tensors"]]
                if cmds[0]["op"] == op:
                    out.append(cmds)
        return out

    def paths(self) -> list[str]:
        """The calls, without health checks and the session brackets (tested on their own)."""
        return [p for p, _ in self.calls if p not in ("/health", "/begin_weight_update", "/end_weight_update")]


class _Process:
    def poll(self):
        return None

    def send_signal(self, sig):
        pass

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


def _write_release(directory: Path, tensors: dict[str, torch.Tensor]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text("{}", encoding="utf-8")
    return directory


def _base_tensors(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {n: torch.randn(s, generator=g).to(torch.bfloat16) for n, s in _shapes().items()}


def _bump(t: torch.Tensor, index: list[int]) -> torch.Tensor:
    """``t`` with the elements ``index`` (flat) moved to the next bf16 value."""
    out = t.clone()
    bits = out.view(-1).view(torch.int16)
    for i in index:
        bits[i] += 1
    return out


def _rebuild(tmp_path: Path, base: Path, new_tensors: dict[str, torch.Tensor], name: str,
             monkeypatch) -> Path:
    """``name``: the new release as a delta mirror would rebuild it from ``base``,
    with its change manifest."""
    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 0)
    target = _write_release(tmp_path / f"{name}-trainer", new_tensors)
    deltas = tmp_path / f"{name}-delta"
    record = rd.encode_directory(base, target, deltas, fmt=1)
    out = tmp_path / name
    counts = rd.apply_directory(base, [(deltas, record.entries)], out, changes=out / rd.CHANGES_DIR)
    rd.write_changes(out, base_dir=base, files=counts)
    return out


def _engine(tmp_path: Path, fake: FakeSGLangRanks, *, tp: int, live: bool = True, **options: Any) -> SGLangServerEngine:
    config = ServerConfig(base_dir=tmp_path / "unused", card_model="H100", device_family="cuda",
                          engine_version="0.5.20", precision="bf16", quant_scheme="bf16",
                          topology=f"tp{tp}", max_model_len=4096, port=30123, in_place_reload=True,
                          request_timeout_s=30.0)
    opts = SGLangOptions(tp=tp, live_update=live, sparse_update="auto", sparse_dir=str(tmp_path / "sparse"),
                         **options)
    return SGLangServerEngine(config, opts, spawn=lambda a, e: _Process(), http=fake, sleep=lambda s: None)


def _reference(tp: int, release: Path) -> list[RankModel]:
    ranks = [RankModel(r, tp) for r in range(tp)]
    for model in ranks:
        model.load_from_disk(release)
    return ranks


def _same_weights(a: list[RankModel], b: list[RankModel]) -> bool:
    return all(torch.equal(dict(x.named_parameters())[n].view(torch.int16),
                           dict(y.named_parameters())[n].view(torch.int16))
               for x, y in zip(a, b) for n, _ in x.named_parameters())


@pytest.fixture
def started(tmp_path):
    tp = 2
    fake = FakeSGLangRanks(tp)
    base = _write_release(tmp_path / "v0", _base_tensors())
    engine = _engine(tmp_path, fake, tp=tp)
    report = engine.start(base, 0)
    return tp, fake, engine, base, report


# ---------------------------------------------------------------- the probe
def test_probe_maps_every_copied_tensor_and_leaves_transformed_ones_unmapped(started):
    tp, fake, engine, base, report = started
    prepare = report.detail["sparse_prepare"]
    assert prepare["enabled"], prepare
    assert prepare["checked"] > 0
    for mapping in engine._sparse_maps:
        assert L1 + "mlp.gate.scale" in mapping["unmapped"]
        assert set(mapping["tensors"]) == set(_shapes()) - {L1 + "mlp.gate.scale"}
    # the probe ran on swapped tensors and the release was reloaded: served weights are the release's
    assert _same_weights(fake.ranks, _reference(tp, base))
    assert "--custom-weight-loader" in engine.argv(base)


def test_probe_disables_sparse_updates_when_the_served_layout_differs(tmp_path):
    fake = FakeSGLangRanks(2, shuffle_after_load=True)
    base = _write_release(tmp_path / "v0", _base_tensors())
    engine = _engine(tmp_path, fake, tp=2)
    prepare = engine.start(base, 0).detail["sparse_prepare"]
    assert not prepare["enabled"]
    assert "layout check failed" in prepare["error"]
    new = _rebuild(tmp_path, base, {**_base_tensors(), "model.norm.weight": _bump(_base_tensors()["model.norm.weight"], [1])},
                   "v1", pytest.MonkeyPatch())
    detail = engine.load(new, 1).detail
    assert detail["sparse_fallback"] == "not_prepared"
    assert fake.paths()[-1] == "/continue_generation" and "/update_weights_from_disk" in fake.paths()[-2:]


# ---------------------------------------------------------------- updates
def test_sparse_update_writes_only_the_changed_elements(started, tmp_path, monkeypatch):
    tp, fake, engine, base, _ = started
    old = _base_tensors()
    changed = {L + "self_attn.k_proj.weight": [0, 9, 31], L + "mlp.experts.1.down_proj.weight": [5, 47],
               "model.embed_tokens.weight": [3, 79], L + "mlp.experts.2.up_proj.weight": [12]}
    new_tensors = {n: _bump(t, changed[n]) if n in changed else t for n, t in old.items()}
    new = _rebuild(tmp_path, base, new_tensors, "v1", monkeypatch)
    fake.calls.clear()

    report = engine.load(new, 1)

    detail = report.detail
    assert detail["sparse"] is True, detail
    assert detail["changed_tensors"] == len(changed)
    assert detail["changed_elements"] == sum(len(v) for v in changed.values())
    assert detail["patch_elements"] == sum(len(v) for v in changed.values())  # each element on one rank
    assert fake.paths() == ["/pause_generation", "/update_weights_from_tensor", "/continue_generation"]
    assert _same_weights(fake.ranks, _reference(tp, new))
    # the patches name only the parameters the changed tensors live in
    patched = set()
    for rank in range(tp):
        patch = json.loads((Path(engine._sparse_root()) / "patch" / f"rank{rank}" / "patch.json").read_text())
        patched |= {p[0] for p in patch["params"]}
    assert patched == {L + "self_attn.qkv_proj.weight", L + "mlp.experts.w2_weight",
                       "model.embed_tokens.weight", L + "mlp.experts.w13_weight"}
    assert engine.served_version() == 1 and fake.version == "1"


def test_consecutive_sparse_updates_keep_versions_in_step(started, tmp_path, monkeypatch):
    tp, fake, engine, base, _ = started
    tensors, served = _base_tensors(), base
    for version in (1, 2, 3):
        name = L + f"mlp.experts.{version % E}.gate_proj.weight"
        tensors = {**tensors, name: _bump(tensors[name], [version, 7 * version])}
        served = _rebuild(tmp_path, served, tensors, f"v{version}", monkeypatch)
        assert engine.load(served, version).detail["sparse"] is True
        assert engine.served_version() == version and fake.version == str(version)
    assert _same_weights(fake.ranks, _reference(tp, served))


@pytest.mark.parametrize("case", ["no_manifest", "base_mismatch", "too_many", "unmapped"])
def test_fallbacks_reload_in_full_before_touching_the_weights(started, tmp_path, monkeypatch, case):
    tp, fake, engine, base, _ = started
    old = _base_tensors()
    if case == "too_many":
        engine.options.sparse_max_ratio = 1e-6
    if case == "unmapped":
        new_tensors = {**old, L1 + "mlp.gate.scale": _bump(old[L1 + "mlp.gate.scale"], [2])}
    else:
        new_tensors = {**old, "model.norm.weight": _bump(old["model.norm.weight"], [0])}
    if case == "no_manifest":
        new = _write_release(tmp_path / "v1", new_tensors)
    elif case == "base_mismatch":
        other = _write_release(tmp_path / "other", _base_tensors(seed=1))
        new = _rebuild(tmp_path, other, new_tensors, "v1", monkeypatch)
    else:
        new = _rebuild(tmp_path, base, new_tensors, "v1", monkeypatch)
    fake.calls.clear()

    detail = engine.load(new, 1).detail

    expected = {"no_manifest": "no_change_manifest", "base_mismatch": "base_mismatch",
                "too_many": "too_many_changes", "unmapped": "unmapped"}[case]
    assert detail["sparse_fallback"] == expected, detail
    assert "/update_weights_from_tensor" not in fake.paths()
    assert fake.paths() == ["/pause_generation", "/update_weights_from_disk", "/continue_generation"]
    assert _same_weights(fake.ranks, _reference(tp, new))
    assert engine.served_version() == 1 and fake.version == "1"


def test_a_failed_check_after_the_write_reloads_in_full(started, tmp_path, monkeypatch):
    tp, fake, engine, base, _ = started
    old = _base_tensors()
    name = L + "self_attn.o_proj.weight"
    new = _rebuild(tmp_path, base, {**old, name: _bump(old[name], [1, 2, 3])}, "v1", monkeypatch)
    original = sglang_sparse_loader._apply

    def lossy(model, command):  # a write that does not land
        result = original(model, command)
        bits = model.o.data.view(-1).view(torch.int16)
        bits.zero_()
        return {**result, **sglang_sparse_loader._compare(Path(command["patch"]),
                                                          json.loads((Path(command["patch"]) / "patch.json").read_text()),
                                                          dict(model.named_parameters()))}

    monkeypatch.setitem(sglang_sparse_loader._OPS, "apply", lossy)
    fake.calls.clear()

    detail = engine.load(new, 1).detail

    assert detail["sparse"] is False and "mismatches" in detail["sparse_repair"]
    assert fake.paths() == ["/pause_generation", "/update_weights_from_tensor", "/continue_generation",
                            "/pause_generation", "/update_weights_from_disk", "/continue_generation"]
    assert _same_weights(fake.ranks, _reference(tp, new))
    assert engine.served_version() == 1 and fake.version == "1"


def test_a_loader_error_is_reported_not_raised(tmp_path):
    result = tmp_path / "r.json"
    command = {"op": "apply", "rank": 0, "patch": str(tmp_path / "missing"), "result": str(result)}
    sglang_sparse_loader.load(RankModel(0, 1), [(sglang_sparse_loader.command_name(command), torch.zeros(1))])
    body = json.loads(result.read_text())
    assert body["ok"] is False and "FileNotFoundError" in body["error"]


def test_sparse_update_is_off_for_quantized_weights_or_nextn(tmp_path):
    fake = FakeSGLangRanks(1)
    config = ServerConfig(base_dir=tmp_path, card_model="H100", device_family="cuda", engine_version="0.5.20",
                          precision="fp8", quant_scheme="fp8", topology="tp1", max_model_len=4096, port=1,
                          in_place_reload=True)
    engine = SGLangServerEngine(config, SGLangOptions(tp=1, sparse_update="auto"), http=fake)
    assert engine.sparse_off and "--custom-weight-loader" not in engine.argv(tmp_path)
    with pytest.raises(Exception):
        SGLangServerEngine(config, SGLangOptions(tp=1, sparse_update="yes"), http=fake)


# ---------------------------------------------------------------- change tracking
def test_mirror_rebuild_records_exactly_the_changed_words(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 0)
    rng = np.random.default_rng(0)
    base = tmp_path / "base"
    base.mkdir()
    words = rng.integers(0, 1 << 16, size=10_001, dtype=np.uint16)
    (base / "a.safetensors").write_bytes(words.tobytes() + b"x")  # odd size: a tail byte
    steps = []
    current = words.copy()
    for k in range(2):
        current = current.copy()
        current[rng.choice(current.size, 50, replace=False)] ^= 1
        target = tmp_path / f"t{k}"
        target.mkdir()
        (target / "a.safetensors").write_bytes(current.tobytes() + (b"y" if k else b"x"))
        steps.append(target)
    links = []
    previous = base
    for k, target in enumerate(steps):
        record = rd.encode_directory(previous, target, tmp_path / f"d{k}", fmt=1 + k)
        links.append((tmp_path / f"d{k}", record.entries))
        previous = target
    out = tmp_path / "out"
    counts = rd.apply_directory(base, links, out, changes=out / rd.CHANGES_DIR)
    got = rd.read_changed_words(out / rd.CHANGES_DIR / ("a.safetensors" + rd.CHANGED_SUFFIX))
    expected = np.flatnonzero(words != current).tolist() + [words.size]  # the tail byte changed
    assert got.tolist() == expected and counts == {"a.safetensors": len(expected)}


def test_changed_elements_maps_words_to_tensor_elements_and_refuses_header_changes(tmp_path):
    tensors = {"a": torch.zeros(4, 3, dtype=torch.bfloat16), "b": torch.zeros(5, dtype=torch.float32)}
    release = _write_release(tmp_path / "r", tensors)
    slots = sparse_update.release_slots(release)
    begin = min(s.start for s in slots.values())
    words = sorted([(slots["a"].start + 2 * 7) // 2, (slots["b"].start + 4 * 3 + 2) // 2,
                    (slots["b"].start + 4 * 3) // 2])
    changes = tmp_path / "w"
    np.array(words, dtype="<u8").tofile(changes)
    got = sparse_update.changed_elements(release, {"model.safetensors": changes, "config.json": _empty(tmp_path)})
    assert {k: v.tolist() for k, v in got.items()} == {"a": [7], "b": [3]}
    np.array([begin // 2 - 1], dtype="<u8").tofile(changes)
    with pytest.raises(sparse_update.SparseUnavailable) as error:
        sparse_update.changed_elements(release, {"model.safetensors": changes})
    assert error.value.reason == "header_changed"


def _empty(tmp_path: Path) -> Path:
    path = tmp_path / "empty.words"
    path.write_bytes(b"")
    return path


def test_default_mirror_tracks_changes_from_the_environment(tmp_path):
    assert rd.default_mirror({rd.MIRROR_ENV: str(tmp_path), rd.TRACK_CHANGES_ENV: "1"}).track_changes
    assert not rd.default_mirror({rd.MIRROR_ENV: str(tmp_path)}, store="s").track_changes


class _GroupModel(torch.nn.Module):
    """Layer 2: ``a`` copied into ``pa``, ``b`` scaled into ``pb``; ``c`` loaded nowhere."""

    def __init__(self) -> None:
        super().__init__()
        self.pa = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16), requires_grad=False)
        self.pb = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16), requires_grad=False)
        self.pc = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16), requires_grad=False)

    def named_parameters(self, *args: Any, **kw: Any):
        yield "model.layers.2.pa", self.pa
        yield "model.layers.2.pb", self.pb
        yield "model.layers.3.pc", self.pc

    def load_weights(self, weights) -> None:
        for name, w in weights:
            if name == "model.layers.2.a":
                self.pa.data.copy_(w)
            elif name == "model.layers.2.b":
                self.pb.data.copy_(w * 2)
            elif name == "model.layers.3.c":
                self.pc.data.copy_(w)


def test_probe_leaves_a_group_with_a_transformed_value_unmapped_and_maps_absent_tensors_to_nothing(tmp_path):
    out, result = tmp_path / "map.pkl", tmp_path / "r.json"
    command = {"op": "prepare", "rank": 0, "out": str(out), "result": str(result),
               "tensors": {"model.layers.2.a": [4], "model.layers.2.b": [4], "model.layers.3.c": [4],
                           "model.layers.3.elsewhere": [4]}}
    sglang_sparse_loader.load(_GroupModel(), [(sglang_sparse_loader.command_name(command), torch.zeros(1))])
    assert json.loads(result.read_text())["ok"]
    mapping = sparse_update.load_map(out)
    assert sorted(mapping["unmapped"]) == ["model.layers.2.a", "model.layers.2.b"]
    assert mapping["tensors"]["model.layers.3.elsewhere"] == []
    assert [t["param"] for t in mapping["tensors"]["model.layers.3.c"]] == ["model.layers.3.pc"]


# ---------------------------------------------------------------- review fixes
def test_changed_elements_of_a_one_byte_tensor_include_both_bytes_of_a_word(tmp_path):
    """A changed 16-bit word of a u8 tensor covers two elements; the odd one must not be lost."""
    tensors = {"a": torch.zeros(8, dtype=torch.uint8), "b": torch.zeros(4, dtype=torch.bfloat16)}
    release = _write_release(tmp_path / "r", tensors)
    slots = sparse_update.release_slots(release)
    assert slots["a"].start % 2 == 0
    changes = tmp_path / "w"
    np.array([(slots["a"].start + 4) // 2], dtype="<u8").tofile(changes)  # bytes 4 and 5 of "a"
    got = sparse_update.changed_elements(release, {"model.safetensors": changes})
    assert {k: v.tolist() for k, v in got.items()} == {"a": [4, 5]}


def test_a_plan_error_falls_back_to_the_full_reload(started, tmp_path, monkeypatch):
    """An OSError while preparing the patches (e.g. /dev/shm full) happens before any
    write: the update falls back to the full reload instead of failing."""
    tp, fake, engine, base, _ = started
    old = _base_tensors()
    new = _rebuild(tmp_path, base, {**old, "model.norm.weight": _bump(old["model.norm.weight"], [0])},
                   "v1", monkeypatch)

    def full(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(sparse_update, "rank_patches", full)
    fake.calls.clear()
    detail = engine.load(new, 1).detail
    assert detail["sparse_fallback"] == "plan_error", detail
    assert "/update_weights_from_tensor" not in fake.paths()
    assert _same_weights(fake.ranks, _reference(tp, new))
    assert engine.served_version() == 1 and fake.version == "1"


def test_a_failed_reload_after_the_probe_fails_the_start(tmp_path):
    """The probe may leave side effects in the weights; when the reload of the
    served release after it fails, the start must not report a healthy server."""
    fake = FakeSGLangRanks(2)
    base = _write_release(tmp_path / "v0", _base_tensors())
    engine = _engine(tmp_path, fake, tp=2)
    real = fake.__call__

    def http(method, url, body, timeout_s):
        if url.endswith("/update_weights_from_disk"):
            return 200, {"success": False, "message": "disk read failed"}
        return real(method, url, body, timeout_s)

    engine._http = http
    with pytest.raises(SamplerEngineError, match="after the sparse probe"):
        engine.start(base, 0)


def test_stop_removes_the_sparse_directory(started):
    tp, fake, engine, base, _ = started
    root = engine._sparse_root()
    assert root.is_dir()
    engine.stop()
    assert not root.exists()


def test_a_failed_continue_after_the_sparse_write_reloads_in_full(started, tmp_path, monkeypatch):
    """continue_generation fails after the patch went in: the update repairs with a
    full reload (its own pause/continue) instead of leaving a half-written state."""
    tp, fake, engine, base, _ = started
    old = _base_tensors()
    name = L + "self_attn.o_proj.weight"
    new = _rebuild(tmp_path, base, {**old, name: _bump(old[name], [1, 2])}, "v1", monkeypatch)
    real, failed = fake.__call__, []

    def http(method, url, body, timeout_s):
        if url.endswith("/continue_generation") and not failed:
            failed.append(True)
            return 500, {"error": "scheduler busy"}
        return real(method, url, body, timeout_s)

    engine._http = http
    fake.calls.clear()
    detail = engine.load(new, 1).detail
    assert detail["sparse"] is False and "continue_generation" in detail["sparse_repair"], detail
    assert "/update_weights_from_disk" in fake.paths()
    assert _same_weights(fake.ranks, _reference(tp, new))
    assert engine.served_version() == 1 and fake.version == "1"


def test_kill_removes_the_sparse_directory(started):
    tp, fake, engine, base, _ = started
    root = engine._sparse_root()
    assert root.is_dir()
    engine._kill()
    assert not root.exists()


# ---------------------------------------------------------------- weight-update sessions (pinned fork)
def test_every_loader_call_is_bracketed_by_a_weight_update_session(started, tmp_path, monkeypatch):
    tp, fake, engine, base, report = started
    assert report.detail["sparse_prepare"]["enabled"]
    old = _base_tensors()
    name = L + "self_attn.k_proj.weight"
    new = _rebuild(tmp_path, base, {n: _bump(t, [0, 9]) if n == name else t for n, t in old.items()},
                   "v1", monkeypatch)
    fake.calls.clear()

    assert engine.load(new, 1).detail["sparse"] is True

    paths = [p for p, _ in fake.calls if p != "/health"]
    assert paths == ["/pause_generation", "/begin_weight_update", "/update_weights_from_tensor",
                     "/end_weight_update", "/continue_generation"]
    assert not fake.session_open


def test_a_server_without_sessions_is_asked_once(tmp_path, monkeypatch):
    fake = FakeSGLangRanks(2)
    fake.sessions = False  # upstream SGLang: /begin_weight_update is a 404
    base = _write_release(tmp_path / "v0", _base_tensors())
    engine = _engine(tmp_path, fake, tp=2)
    assert engine.start(base, 0).detail["sparse_prepare"]["enabled"]
    old = _base_tensors()
    name = L + "self_attn.k_proj.weight"
    new = _rebuild(tmp_path, base, {n: _bump(t, [3]) if n == name else t for n, t in old.items()},
                   "v1", monkeypatch)

    assert engine.load(new, 1).detail["sparse"] is True

    assert [p for p, _ in fake.calls].count("/begin_weight_update") == 1
    assert "/end_weight_update" not in [p for p, _ in fake.calls]
    assert _same_weights(fake.ranks, _reference(2, new))


def test_command_names_carry_no_colon():
    command = {"op": "apply", "rank": 3, "patch": "/dev/shm/x:y", "result": "/tmp/r.json"}
    name = sglang_sparse_loader.command_name(command)
    assert ":" not in name and name.startswith(sglang_sparse_loader.PREFIX)
    assert sglang_sparse_loader.parse_command(name) == command
    assert sglang_sparse_loader.parse_command("model.layers.0.mlp.weight") is None


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_during_the_reload_after_the_sparse_probe_is_not_wrapped(tmp_path, interrupt):
    """``except BaseException`` around the post-probe reload turned Ctrl-C / SystemExit
    into ``_ProbeReloadFailed`` (a SamplerEngineError a caller may retry)."""
    fake = FakeSGLangRanks(2)
    base = _write_release(tmp_path / "v0", _base_tensors())
    engine = _engine(tmp_path, fake, tp=2)
    real = fake.__call__

    def http(method, url, body, timeout_s):
        if url.endswith("/update_weights_from_disk"):
            raise interrupt()
        return real(method, url, body, timeout_s)

    engine._http = http
    with pytest.raises(interrupt):
        engine.start(base, 0)


def test_a_failed_sparse_probe_frees_its_shared_memory(tmp_path):
    """Sparse updates turned off by a failed probe never use the element maps again,
    and stop() only removed the directory when ``sparse_off`` was None at start: the
    maps (one per rank, up to GBs in /dev/shm) stayed in RAM for the server's life."""
    fake = FakeSGLangRanks(2, shuffle_after_load=True)
    base = _write_release(tmp_path / "v0", _base_tensors())
    engine = _engine(tmp_path, fake, tp=2)
    assert not engine.start(base, 0).detail["sparse_prepare"]["enabled"]
    assert not engine._sparse_root().exists()


def test_a_failed_sparse_plan_frees_the_half_written_patch(started, tmp_path, monkeypatch):
    """/dev/shm full while the patch is written: the update falls back (nothing touched),
    and the half-written patch must not stay in shared memory until the next update."""
    tp, fake, engine, base, _ = started
    old = _base_tensors()
    name = L + "self_attn.o_proj.weight"
    new = _rebuild(tmp_path, base, {**old, name: _bump(old[name], [1, 2])}, "v1", monkeypatch)
    real = sparse_update.rank_patches

    def full(out, *a, **kw):
        real(out, *a, **kw)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(sparse_update, "rank_patches", full)
    detail = engine.load(new, 1).detail
    assert "sparse_fallback" in detail or detail.get("sparse") is False, detail
    assert not (engine._sparse_root() / "patch").exists()
    assert (engine._sparse_root() / "maps").is_dir()  # the maps stay: the next release can go sparse
