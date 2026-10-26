"""B15: the Megatron HF boundary across two tensor-name namespaces (S27 G-FT).

Megatron-Bridge names the tensors of the *base checkpoint*, the island's
full-parameter layout names the HF model's own trainable parameters.  For Qwen3
the two are one set and the boundary compared them name for name; for NemotronH
they are not (``backbone.`` against ``model.``, and 128 per-expert checkpoint
tensors against one layout tensor stacked along dim 0), so ``HfTaskIndex`` refused
377 of the island's 378 layout names (S27 G-FN, run fn-10, VERL-MEGATRON-PP §7.12.13).

The tests below use a *tiny real* NemotronH: ``transformers`` builds it on the meta
device from a four-layer config with four experts, so the conversion rules, the
natural sort order that decides the stacking rows and the ``state_dict`` / layout
difference are the production ones, not a hand-written imitation.

Two asymmetric checks are deliberate (the 10-26 lesson that a symmetric round trip
has blind spots): every expert row carries values only that row can have, so an
assembly in the wrong order is distinguishable, and the mutation tests prove the
check is the thing that fails -- a rotated row map and a dropped expert-parallel
rank both have to be refused.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from yeto.rl import megatron_hf_boundary as boundary  # noqa: E402
from yeto.rl.adapters.verl import full_parameter as fp  # noqa: E402
from yeto.rl.adapters.verl import megatron_full_parameter as mfp  # noqa: E402

STACKED = "model.layers.2.mixer.experts.up_proj"
BIAS = "backbone.layers.2.mixer.gate.e_score_correction_bias"


# ----------------------------------------------------------------------------- a tiny real NemotronH
@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory) -> Path:
    """A four-layer NemotronH (one MoE layer, four experts) as a model directory.

    Only ``config.json`` is written: every build in this file is on the meta
    device, so no tensor is allocated and no weight is downloaded."""

    if not hasattr(transformers, "NemotronHConfig"):
        pytest.skip("transformers has no NemotronHConfig")
    config = transformers.NemotronHConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=64, hybrid_override_pattern="M*E-",
        n_routed_experts=4, num_experts=4, moe_intermediate_size=8,
        mamba_num_heads=4, mamba_head_dim=4, mamba_d_state=8, mamba_d_conv=4, ssm_state_size=8,
        moe_router_enable_expert_bias=True,
    )
    directory = tmp_path_factory.mktemp("nemotron_h_tiny")
    config.save_pretrained(directory)
    assert json.loads((directory / "config.json").read_text())["model_type"] == "nemotron_h"
    return directory


def _shapes(model_dir: Path) -> dict[str, tuple[int, ...]]:
    from yeto.rl.adapters.verl.full_parameter_plan import model_shapes

    return model_shapes(model_dir)


def _state_names(model_dir: Path) -> tuple[str, ...]:
    from yeto.rl.adapters.verl.hf_checkpoint_keys import model_state_conversion

    _rules, _prefix, state = model_state_conversion(model_dir)
    return state


def _bridge_name(state_name: str) -> str:
    """The base checkpoint's name of a model state name (Bridge's namespace).

    The inverse of the two renamings this family needs, written out here so the
    test gives :func:`megatron_full_parameter.layout_map` checkpoint names it did
    not itself produce."""

    if state_name.startswith("model.embeddings.weight"):
        return "backbone.embedding.weight"
    if state_name.startswith("model."):
        return "backbone." + state_name[len("model."):]
    return state_name


def _bridge_names(model_dir: Path, experts: int = 4) -> list[str]:
    """Every checkpoint tensor name of the tiny model, per expert (unstacked)."""

    out = []
    for name in _state_names(model_dir):
        base = _bridge_name(name)
        if base.endswith((".experts.up_proj", ".experts.down_proj")):
            head, _dot, leaf = base.rpartition(".")
            out += [f"{head}.{expert}.{leaf}.weight" for expert in range(experts)]
        else:
            out.append(base)
    return out


# ----------------------------------------------------------------------------- the map itself
def test_the_layout_map_sends_every_checkpoint_name_to_the_layout(tiny_model):
    """``transformers``' own rules cover every layout name; the per-expert names
    become rows of one layout tensor and the router bias falls outside."""

    shapes = _shapes(tiny_model)
    names = _bridge_names(tiny_model)
    layout = mfp.layout_map(str(tiny_model), names, shapes)

    assert layout is not boundary.IDENTITY
    assert sorted(layout.slots) == sorted(names)  # the map mentions every name it was given
    covered = {slot.name for slot in layout.slots.values() if slot is not None}
    assert sorted(covered) == sorted(shapes), "the layout names must be covered exactly"

    # a plain rename, both halves of it
    assert layout.slot("backbone.norm_f.weight") == boundary.LayoutSlot("model.norm_f.weight", None)
    assert layout.slot("backbone.embedding.weight") == boundary.LayoutSlot("model.embeddings.weight", None)
    assert layout.slot("lm_head.weight") == boundary.LayoutSlot("lm_head.weight", None)

    # the stacked target: four sources, rows 0..3 in the loader's natural order
    rows = {slot.row: name for name, slot in layout.slots.items()
            if slot is not None and slot.name == STACKED}
    assert sorted(rows) == [0, 1, 2, 3]
    assert rows[0] == "backbone.layers.2.mixer.experts.0.up_proj.weight"
    assert rows[3] == "backbone.layers.2.mixer.experts.3.up_proj.weight"
    assert layout.row_numel[STACKED] == math.prod(shapes[STACKED][1:])
    assert shapes[STACKED][0] == 4

    # the persistent buffer: in the model's state_dict, not in the layout
    assert BIAS in _bridge_names(tiny_model) and BIAS in layout.slots
    assert layout.slot(BIAS) is None
    assert BIAS.replace("backbone.", "model.") in _state_names(tiny_model)
    assert BIAS.replace("backbone.", "model.") not in shapes


def test_a_model_whose_checkpoint_names_are_its_layout_names_needs_no_map():
    """The Qwen3 case: no meta build, no rules, the boundary as it was."""

    shapes = {"model.layers.0.mlp.up_proj.weight": (4, 2), "lm_head.weight": (8, 2)}
    assert mfp.layout_map("/does/not/exist", shapes, shapes) is boundary.IDENTITY
    assert mfp.layout_map(None, ["anything"], shapes) is boundary.IDENTITY


# ----------------------------------------------------------------------------- the index
def _records(per_rank_names: list[dict[int, list[str]]], dtype: str = "float32") -> list[dict]:
    """One manifest record per rank: ``{task index: [its Bridge HF names]}``."""

    return [{"tasks": [[index, names, dtype] for index, names in sorted(rank.items())], "refused": []}
            for rank in per_rank_names]


def _expert(index: int, leaf: str = "up_proj") -> str:
    return f"backbone.layers.2.mixer.experts.{index}.{leaf}.weight"


def _ep2_records() -> list[dict]:
    """Four ranks of PP=2 x EP=2 over one stacked layout tensor.

    Each expert-parallel rank holds two of the four experts and reports its own
    name at each task index (what ``megatron_hf_boundary.manifest`` records); the
    two pipeline ranks of one expert-parallel coordinate report the same names."""

    ep0 = {0: [_expert(0)], 1: [_expert(1)]}
    ep1 = {0: [_expert(2)], 1: [_expert(3)]}
    return _records([ep0, ep1, ep0, ep1])


def _four_row_layout(height: int = 4, row_numel: int = 6) -> boundary.LayoutMap:
    return boundary.LayoutMap({_expert(row): boundary.LayoutSlot(STACKED, row) for row in range(height)},
                              {STACKED: row_numel})


def test_many_tasks_may_write_one_layout_name_and_each_row_exactly_one_task():
    """The multi-to-one case the old ``doubled`` check refused outright."""

    index = boundary.HfTaskIndex.from_manifests(_ep2_records(), [STACKED], layout=_four_row_layout())

    assert index.by_name[STACKED] == (0, 1), "both task indices write part of the one layout tensor"
    assert index.names == {0: frozenset({STACKED}), 1: frozenset({STACKED})}
    assert index.stacked == {STACKED: 4}
    assert index.tasks_for([STACKED]) == [0, 1]
    assert index.apply_plan([STACKED]) == ([0, 1], []), "a whole layout tensor needs no read-back"
    index.check_stack_heights({STACKED: (4, 2, 3)})

    # the rows of one task index are the experts of the different EP ranks
    assert {slot.row for slot in index.slots[0]} == {0, 2}
    assert {slot.row for slot in index.slots[1]} == {1, 3}
    assert index.layout_for([0]).slots.keys() == {_expert(0), _expert(2)}


def test_two_tasks_writing_one_row_is_still_refused():
    """Multi-to-one on the name, one-to-one on the row."""

    records = _records([{0: [_expert(0)], 1: [_expert(0), _expert(1)]}])
    with pytest.raises(boundary.MegatronHfBoundaryError, match=r"two conversion tasks write.*up_proj\[0\]"):
        boundary.HfTaskIndex.from_manifests(records, [STACKED], layout=_four_row_layout())


def test_a_hole_in_the_rows_is_refused():
    records = _records([{0: [_expert(0)], 1: [_expert(2)]}])
    with pytest.raises(boundary.MegatronHfBoundaryError, match=r"exports 2 rows up to 2, 1 missing"):
        boundary.HfTaskIndex.from_manifests(records, [STACKED], layout=_four_row_layout())


def test_a_name_both_whole_and_in_rows_is_refused():
    layout = boundary.LayoutMap({_expert(0): boundary.LayoutSlot(STACKED, 0),
                                 "backbone.whole": boundary.LayoutSlot(STACKED, None)}, {STACKED: 6})
    records = _records([{0: [_expert(0)], 1: ["backbone.whole"]}])
    with pytest.raises(boundary.MegatronHfBoundaryError, match="both whole and in rows"):
        boundary.HfTaskIndex.from_manifests(records, [STACKED], layout=layout)


def test_the_router_buffers_stay_outside_the_boundary(tiny_model):
    """Option A for the 23 NemotronH router biases: the index drops them.

    They are persistent buffers, so neither this layout nor the FSDP2 island's
    holds them; dropping them keeps one layout hash for the two islands.  Without
    the drop they come back as ``extra``, which is what blocks the island."""

    shapes = _shapes(tiny_model)
    names = _bridge_names(tiny_model)
    layout = mfp.layout_map(str(tiny_model), names, shapes)
    records = _records([{position: [name] for position, name in enumerate(names)}])

    index = boundary.HfTaskIndex.from_manifests(records, shapes, layout=layout)
    index.check_stack_heights(shapes)
    assert sorted(index.by_name) == sorted(shapes), "no missing and no extra layout name"
    assert index.stacked == {STACKED: 4, STACKED.replace("up_proj", "down_proj"): 4}
    dtypes = index.master_dtypes(records)
    assert sorted(dtypes) == sorted(shapes)
    assert not [name for name in dtypes if "e_score_correction_bias" in name]
    mfp.check_hybrid_rules(shapes, dtypes, json.loads((tiny_model / "config.json").read_text()))

    kept = boundary.LayoutMap({**layout.slots, BIAS: boundary.LayoutSlot(BIAS.replace("backbone.", "model."))},
                              layout.row_numel)
    with pytest.raises(boundary.MegatronHfBoundaryError, match="names the layout does not have"):
        boundary.HfTaskIndex.from_manifests(records, shapes, layout=kept)


def test_a_dropped_expert_parallel_rank_is_refused(tiny_model):
    """The mutation that matters most: one rank's manifest missing.

    Its experts are then absent from the names the map is built from, so the rows
    renumber to a shorter complete run -- the hole check cannot see it and the
    stack height is the check that must."""

    shapes = _shapes(tiny_model)
    names = [name for name in _bridge_names(tiny_model)
             if not name.endswith((".2.up_proj.weight", ".3.up_proj.weight",
                                   ".2.down_proj.weight", ".3.down_proj.weight"))]
    layout = mfp.layout_map(str(tiny_model), names, shapes)
    records = _records([{position: [name] for position, name in enumerate(names)}])
    index = boundary.HfTaskIndex.from_manifests(records, shapes, layout=layout)
    assert index.stacked[STACKED] == 2, "rows renumber, so the hole check stays silent"
    with pytest.raises(boundary.MegatronHfBoundaryError, match="Bridge exports 2 rows, the layout has 4"):
        index.check_stack_heights(shapes)


def test_without_a_map_the_index_behaves_as_before():
    records = _records([{0: ["lm_head.weight"], 1: ["model.norm.weight"]}])
    index = boundary.HfTaskIndex.from_manifests(records, ["lm_head.weight", "model.norm.weight"])
    assert index.by_name == {"lm_head.weight": (0,), "model.norm.weight": (1,)}
    assert index.stacked == {}
    assert index.layout is boundary.IDENTITY
    with pytest.raises(boundary.MegatronHfBoundaryError, match="exports no tensor for layout names"):
        boundary.HfTaskIndex.from_manifests(records, ["lm_head.weight", "model.absent.weight"])


# ----------------------------------------------------------------------------- the assembly
ROWS, ROW_SHAPE = 4, (2, 3)
ROW_NUMEL = math.prod(ROW_SHAPE)


def _expert_tensor(row: int):
    """Row ``row``'s values, which no other row can have."""

    return (torch.arange(ROW_NUMEL, dtype=torch.float32) + 100.0 * (row + 1)).reshape(ROW_SHAPE)


def _whole_stack():
    return torch.stack([_expert_tensor(row) for row in range(ROWS)])


def _assemble(layout: boundary.LayoutMap, *, feed: list[int] | None = None,
              units: list | None = None, numel: int | None = None):
    """Run the sink over the per-expert tensors and return ``(fragment, sink)``."""

    units = units if units is not None else [[STACKED, 0, ROWS * ROW_NUMEL]]
    numel = numel if numel is not None else sum(int(stop) - int(start) for _n, start, stop in units)
    fragment = fp.FlatFragment(units, numel, reader=True)
    sink = boundary.LayoutSink(layout, fragment.add_part, want={STACKED})
    for row in (feed if feed is not None else list(range(ROWS))):
        sink(_expert(row), _expert_tensor(row))
    return fragment, sink


def test_the_sink_assembles_the_stacked_layout_tensor_in_the_fragment_buffer():
    """Nothing stacks a whole tensor: each expert lands at its row's offset."""

    fragment, sink = _assemble(_four_row_layout(ROWS, ROW_NUMEL))
    assert (sink.kept, sink.dropped, sink.skipped) == (ROWS, 0, 0)
    assert torch.equal(fragment.flat, _whole_stack().reshape(-1))
    result = fragment.result({}, want_payload=False, want_hash=True, started=0.0)
    assert result["finite"] is True and len(result["sha256"]) == 64


@pytest.mark.parametrize("broken", ["rotated", "swapped", "all-row-zero"])
def test_a_wrong_assembly_order_fails_the_check(broken):
    """The mutation check: break the row map, the assembled tensor must differ.

    Without this the test above would pass for a sink that ignored the row and
    overwrote one place, or that stacked in arrival order."""

    if broken == "rotated":
        slots = {_expert(row): boundary.LayoutSlot(STACKED, (row + 1) % ROWS) for row in range(ROWS)}
    elif broken == "swapped":
        order = {0: 1, 1: 0, 2: 2, 3: 3}
        slots = {_expert(row): boundary.LayoutSlot(STACKED, order[row]) for row in range(ROWS)}
    else:
        slots = {_expert(row): boundary.LayoutSlot(STACKED, 0) for row in range(ROWS)}

    layout = boundary.LayoutMap(slots, {STACKED: ROW_NUMEL})
    fragment, _sink = _assemble(layout)
    assert not torch.equal(fragment.flat, _whole_stack().reshape(-1)), f"{broken} must not look correct"

    records = _records([{row: [_expert(row)] for row in range(ROWS)}])
    if broken == "all-row-zero":
        # Every source claims row 0: three quarters of the tensor is never
        # written, and the index refuses the map outright.
        assert fragment.covered[STACKED] == ROW_NUMEL
        with pytest.raises(fp.VerlFullParameterError, match="没有完整导出"):
            fragment.result({}, want_payload=False, want_hash=False, started=0.0)
        with pytest.raises(boundary.MegatronHfBoundaryError, match="two conversion tasks write"):
            boundary.HfTaskIndex.from_manifests(records, [STACKED], layout=layout)
    else:
        # A rotation or a swap is still one source per row, so every structural
        # check passes and only the values tell them apart.  That is the point:
        # the order is checked by comparing values, not by counting them.
        index = boundary.HfTaskIndex.from_manifests(records, [STACKED], layout=layout)
        index.check_stack_heights({STACKED: (ROWS, *ROW_SHAPE)})
        fragment.result({}, want_payload=False, want_hash=False, started=0.0)


def test_a_missing_shard_leaves_the_fragment_short():
    """The other mutation: one expert-parallel rank's rows never arrive."""

    fragment, sink = _assemble(_four_row_layout(ROWS, ROW_NUMEL), feed=[0, 1, 3])
    assert sink.kept == 3
    assert fragment.covered[STACKED] == 3 * ROW_NUMEL < fragment.needed[STACKED]
    with pytest.raises(fp.VerlFullParameterError, match=rf"{ROWS - 1} \* {ROW_NUMEL}|没有完整导出"):
        fragment.result({}, want_payload=False, want_hash=False, started=0.0)


def test_a_fragment_over_part_of_a_stacked_tensor_takes_only_its_elements():
    """A layout tensor larger than one fragment: rows outside the fragment's
    element range are copied nowhere, the rows inside land whole."""

    start, stop = ROW_NUMEL, 3 * ROW_NUMEL  # rows 1 and 2
    fragment, sink = _assemble(_four_row_layout(ROWS, ROW_NUMEL), units=[[STACKED, start, stop]])
    assert sink.kept == ROWS
    assert torch.equal(fragment.flat, _whole_stack().reshape(-1)[start:stop])
    fragment.result({}, want_payload=False, want_hash=False, started=0.0)


def test_a_source_tensor_of_the_wrong_size_is_refused():
    layout = _four_row_layout(ROWS, ROW_NUMEL)
    fragment = fp.FlatFragment([[STACKED, 0, ROWS * ROW_NUMEL]], ROWS * ROW_NUMEL, reader=True)
    sink = boundary.LayoutSink(layout, fragment.add_part)
    with pytest.raises(boundary.MegatronHfBoundaryError, match="elements, row 1 of"):
        sink(_expert(1), torch.zeros(ROW_NUMEL + 1))

    no_size = boundary.LayoutMap(layout.slots, {})
    with pytest.raises(boundary.MegatronHfBoundaryError, match="whose row size is not known"):
        boundary.LayoutSink(no_size, fragment.add_part)(_expert(1), _expert_tensor(1))


def test_the_sink_drops_what_the_layout_does_not_hold_and_skips_what_is_not_asked():
    layout = boundary.LayoutMap({**_four_row_layout(ROWS, ROW_NUMEL).slots, BIAS: None},
                                {STACKED: ROW_NUMEL})
    fragment = fp.FlatFragment([[STACKED, 0, ROWS * ROW_NUMEL]], ROWS * ROW_NUMEL, reader=True)
    sink = boundary.LayoutSink(layout, fragment.add_part, want={STACKED})
    sink(BIAS, torch.zeros(4))
    sink("backbone.norm_f.weight", torch.zeros(4))
    assert (sink.kept, sink.dropped, sink.skipped) == (0, 1, 1)


# ----------------------------------------------------------------------------- the apply direction
def test_each_task_gets_its_own_row_of_the_whole_layout_tensor():
    """An apply sends whole layout tensors; every Bridge name is one row view."""

    whole = {STACKED: _whole_stack()}
    inputs = mfp.bridge_inputs(_four_row_layout(ROWS, ROW_NUMEL), whole)

    assert sorted(inputs) == sorted(_expert(row) for row in range(ROWS))
    for row in range(ROWS):
        assert torch.equal(inputs[_expert(row)], _expert_tensor(row))
        assert inputs[_expert(row)].shape == ROW_SHAPE
        assert inputs[_expert(row)].data_ptr() >= whole[STACKED].data_ptr(), "a view, not a copy"
    assert mfp.bridge_inputs(None, whole) == whole
    assert mfp.bridge_inputs(boundary.IDENTITY, whole) == whole


def test_a_layout_tensor_that_was_not_sent_is_left_to_the_read_back():
    layout = boundary.LayoutMap({**_four_row_layout(ROWS, ROW_NUMEL).slots,
                                 "backbone.norm_f.weight": boundary.LayoutSlot("model.norm_f.weight")},
                                {STACKED: ROW_NUMEL})
    inputs = mfp.bridge_inputs(layout, {"model.norm_f.weight": torch.zeros(4)})
    assert sorted(inputs) == ["backbone.norm_f.weight"]
