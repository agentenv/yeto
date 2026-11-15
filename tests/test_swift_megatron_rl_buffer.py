"""yeto.swift_megatron.rl_buffer helpers without ms-swift/Megatron/NPU."""

import pytest

from yeto.swift_megatron import rl_buffer as rb


def sample(n=3, **kw):
    s = {"index": 0, "tokens": [1, 2, 10, 11, 12], "response_length": n, "rollout_log_probs": [-0.1] * n,
         "reward": 1.0, "status": "completed", "response": "x", "prompt": "q"}
    s.update(kw)
    return s


def test_response_fields():
    f = rb.response_fields(sample())
    assert f["response_token_ids"] == [[10, 11, 12]] and f["rollout_logprobs"] == [[-0.1] * 3]
    assert f["finish_reason"] == "stop" and f["reward"] == 1.0
    assert rb.response_fields(sample(status="truncated"))["finish_reason"] == "length"
    assert rb.response_fields(sample(rollout_log_probs=[]))["rollout_logprobs"] == []


@pytest.mark.parametrize("bad", [dict(response_length=0), dict(response_length=9), dict(rollout_log_probs=[-1.0])])
def test_response_fields_refuses_inconsistent_samples(bad):
    with pytest.raises(ValueError):
        rb.response_fields(sample(**bad))


def test_prompt_messages():
    assert rb.prompt_messages("hi") == [{"role": "user", "content": "hi"}]
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert rb.prompt_messages(msgs) == [{"role": "user", "content": "a"}]


def test_flatten_groups_keeps_order_and_checks_group_size():
    assert rb.flatten_groups([(0, [1, 2]), (5, [3, 4])], 2) == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        rb.flatten_groups([(0, [1, 2, 3])], 2)


def test_import_without_buffer_root_installs_nothing(monkeypatch):
    monkeypatch.delenv(rb.ROOT_ENV, raising=False)
    assert rb.install() is False
