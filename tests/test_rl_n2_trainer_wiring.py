"""S25 N2 wiring on the trainer side (CPU): engine-sampler payloads in the
trainer readers, and the per-source ``offpolicy_source`` event in Miles and verl.

* ``complete_prompt_tokens``: a payload whose engine gave no prompt ids
  (``prompt_tokens_known`` False) gets its prompt ids from the trainer's
  tokenizer, or is refused;
* Miles reader (``buffer_rollout_miles._BufferRound``, stub Miles modules):
  rebuilt tokens, provenance kept in the metadata, per-source metrics;
* verl (``ports_impl.source_event``): padded trainer tensors + per-row
  provenance -> one entry per source, masked tokens only.
"""

from __future__ import annotations

import math
import types

import pytest
from test_rl_buffer_rollout import _import_plugin_with_stub_miles

from yeto.rl.adapters.sampler_engines.fake import FAKE_IDENTITY
from yeto.rl.engine.buffer_payload import PROMPT_TOKENIZED_BY, SampleBufferRolloutError, complete_prompt_tokens
from yeto.rl.engine.engine_sampler import PromptItem, sample_record
from yeto.rl.engine.offpolicy_source import (
    flat_metrics,
    masked_source_samples,
    offpolicy_source_event,
    payload_source_sample,
)
from yeto.rl.engine.sampler_engine import FINISH_LENGTH, FINISH_STOP, Completion, Provenance, SamplingParams

SOURCE = Provenance.of(FAKE_IDENTITY, sampler_id="s1", sampling=SamplingParams(max_tokens=8)).to_record()
KEY = Provenance.from_record(SOURCE).key()


def _record(prompt_ids=(), finish=FINISH_STOP, index=0):
    completion = Completion(token_ids=(40, 41), logprobs=(-0.1, -0.2), text="ok", finish_reason=finish,
                            prompt_token_ids=tuple(prompt_ids))
    return sample_record(PromptItem("p0", "2+2="), completion, reward=1.0, version=3, provenance=SOURCE,
                         group_index=0, index=index)


def test_prompt_ids_come_from_the_trainer_tokenizer_when_the_engine_gave_none():
    sample = _record()
    assert sample["metadata"]["prompt_tokens_known"] is False and sample["tokens"] == [40, 41]
    done = complete_prompt_tokens(sample, lambda text: [7, 8, 9] if text == "2+2=" else [])
    assert done["tokens"] == [7, 8, 9, 40, 41] and done["response_length"] == 2
    assert done["metadata"][PROMPT_TOKENIZED_BY] == "trainer"
    assert done["metadata"]["provenance"] == SOURCE  # the source record travels on
    assert sample["tokens"] == [40, 41]  # the payload sample itself is not changed

    with pytest.raises(SampleBufferRolloutError, match="no tokenizer"):
        complete_prompt_tokens(sample, None)
    with pytest.raises(SampleBufferRolloutError, match="no ids"):
        complete_prompt_tokens(sample, lambda text: [])
    with pytest.raises(SampleBufferRolloutError, match="response_length"):
        complete_prompt_tokens({**sample, "tokens": [1, 40, 41]}, lambda text: [7])

    known = _record(prompt_ids=(7, 8))
    assert complete_prompt_tokens(known, None)["tokens"] == [7, 8, 40, 41]  # engine ids are kept


def test_miles_reader_rebuilds_prompts_and_reports_per_source_metrics(monkeypatch):
    plugin = _import_plugin_with_stub_miles(monkeypatch, {})
    monkeypatch.setattr(plugin, "Sample", types.SimpleNamespace(from_dict=dict))
    monkeypatch.setattr(plugin, "RolloutFnTrainOutput", lambda samples, metrics: (samples, metrics))
    reader = plugin._BufferRound.__new__(plugin._BufferRound)
    reader.args = types.SimpleNamespace(rollout_sample_filter_path=None, rollout_all_samples_process_path=None)
    reader.samples_per_group = 2
    reader._tokenizer = lambda text, add_special_tokens: {"input_ids": [7, 8]}
    report = types.SimpleNamespace(metrics=lambda: {"sample_buffer/groups": 1.0})
    group = [_record(index=0), _record(finish=FINISH_LENGTH, index=1)]
    reader.reader = types.SimpleNamespace(take=lambda rollout_id, groups, current_version: ([(5, group)], report))
    reader.groups = 1

    samples, metrics = reader(types.SimpleNamespace(evaluation=False, rollout_id=2, weight_version=3))
    ((first, second),) = samples
    assert first["tokens"] == [7, 8, 40, 41] and first["group_index"] == 5 and second["index"] == 11
    assert first["metadata"]["provenance"] == SOURCE
    assert metrics["sample_buffer/groups"] == 1.0
    assert metrics[f"offpolicy_source/{KEY}/samples"] == 2.0
    assert metrics[f"offpolicy_source/{KEY}/truncated_share"] == 0.5
    # Miles' trainer does not hand its log-probs back: no ratio metrics on this side
    assert not any(name.endswith("/ratio_p50") for name in metrics)


def test_payload_source_sample_counts_without_train_logprobs():
    event = offpolicy_source_event([payload_source_sample(_record())], rollout_id=0, trainer_version=0,
                                   backend="miles")
    (entry,) = event["sources"]
    assert entry["key"] == KEY and entry["samples"] == 1 and entry["tokens"] == 0 and entry["ratio_p50"] is None
    sample = payload_source_sample(_record(), train_logprobs=[-0.1, -0.3])
    assert list(sample.rollout_logprobs) == [-0.1, -0.2]


def test_verl_source_event_groups_padded_rows_by_their_provenance():
    torch = pytest.importorskip("torch")
    from yeto.rl.adapters.verl.ports_impl import source_event

    train = torch.tensor([[-0.1, -0.2, 0.0], [-1.0, 0.0, 0.0], [-0.5, -0.5, -0.5]])
    rollout = torch.tensor([[-0.1, -0.2, 9.0], [-2.0, 0.0, 0.0], [-0.5, -0.5, -0.5]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 0, 0]])  # row 2: an ABORTED placeholder
    event = source_event(train, rollout, mask, [SOURCE, SOURCE, None], "ok",
                         rollout_id=4, trainer_version=4, tis_upper=2.0)
    sources = {s["key"]: s for s in event["sources"]}
    assert event["consumed"] == 3 and event["provenance_read"] == "ok" and event["backend"] == "verl"
    ours = sources[KEY]
    assert ours["samples"] == 2 and ours["tokens"] == 3  # the padded token (rollout 9.0) is not read
    assert math.isclose(ours["ratio_max"], math.e)  # row 1: exp(-1.0 - -2.0)
    assert sources["unknown/unknown/unknown@unknown/unknown/unknown"]["tokens"] == 0
    assert flat_metrics(event)[f"offpolicy_source/{KEY}/samples"] == 2.0


def test_masked_source_samples_checks_row_counts():
    with pytest.raises(ValueError, match="rows"):
        masked_source_samples([[0.0]], [[0.0]], [[1], [1]])
    with pytest.raises(ValueError, match="provenances"):
        masked_source_samples([[0.0]], [[0.0]], [[1]], provenances=[None, None])


def test_verl_tq_provenance_is_read_from_the_extra_fields_record(monkeypatch):
    """verl's TQ agent loop stores ``AgentLoopOutput.as_dict()``: provenance sits in
    ``extra_fields``, not under its own name (S25 T-G4)."""
    import sys
    import types

    from yeto.rl.adapters.verl import ports_impl

    class Wrapped:  # a TransferQueue read can wrap each record (NonTensorData)
        def __init__(self, data):
            self.data = data

    seen = {}

    def kv_batch_get(*, keys, partition_id, select_fields):
        seen["fields"] = select_fields
        return {"extra_fields": [Wrapped({"provenance": SOURCE}), {"other": 1}]}

    monkeypatch.setitem(sys.modules, "transfer_queue", types.SimpleNamespace(kv_batch_get=kv_batch_get))
    batch = types.SimpleNamespace(keys=["a", "b"], partition_id="train")
    assert ports_impl._row_field(batch, "provenance") == ([SOURCE, None], "ok")
    assert seen["fields"] == ["extra_fields"]

    monkeypatch.setitem(sys.modules, "transfer_queue", types.SimpleNamespace(
        kv_batch_get=lambda **_: {"extra_fields": [{}, {}]}))
    rows, read = ports_impl._row_field(batch, "provenance")
    assert rows is None and read.startswith("missing")
