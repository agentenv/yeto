"""token_alignment losses of the Harbor + pi path (K3.1 run, 2026-11-10).

Root cause seen in the pi sessions: pi omits an empty reasoning and
re-serializes tool-call arguments (``JSON.stringify``), so its echo of a turn
the proxy generated hashed differently -> ``retry_fork`` -> a new recorder
session that re-tokenizes the history.  When the sampled ids are not the
tokenizer's canonical encoding of their text (byte-level pieces of binary /
non-UTF-8 output), the new session's prompt does not extend the old stream
and the whole trial was dropped.  A byte-level mock tokenizer stands in for
the K3.1 tokenizer (``tiktoken`` is not installed here)."""

from __future__ import annotations

import asyncio
import json

import pytest

from yeto.rl.agentic import TokenAlignmentError, TokenTurn
from yeto.rl.agentic.finish import finish_sample
from yeto.rl.agentic.tokens import join_turns
from yeto.rl.harness.gateway.chains import ChainRegistry, LocateKind


class ByteTokenizer:
    """ids 0-255: single bytes; 256: the merged pair b"\\xc3\\xa9" ("é"). Encoding
    merges, so sampled ids [0xc3, 0xa9] re-encode as [256]."""

    def encode(self, text: str) -> list[int]:
        raw, out, i = text.encode("utf-8", errors="surrogateescape"), [], 0
        while i < len(raw):
            if raw[i:i + 2] == b"\xc3\xa9":
                out.append(256)
                i += 2
            else:
                out.append(raw[i])
                i += 1
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(b"\xc3\xa9" if i == 256 else bytes([i]) for i in ids).decode("utf-8", errors="replace")


TOK = ByteTokenizer()


def _turn(prompt: list[int], completion: list[int], *, lp: float = -0.5) -> TokenTurn:
    return TokenTurn(prompt_ids=tuple(prompt), completion_ids=tuple(completion),
                     completion_logprobs=tuple([lp] * len(completion)))


def _forked_turns() -> list[TokenTurn]:
    """Turn 0 samples "é" as two byte tokens; the fork's session re-encodes it merged."""
    p0 = TOK.encode("<sys>do it<asst>")
    c0 = [*TOK.encode("caf"), 0xC3, 0xA9, *TOK.encode("</a>")]  # non-canonical: merge would give 256
    tool = TOK.encode("<tool>ok<asst>")
    p1 = [*p0, *c0, *tool]
    c1 = TOK.encode("done</a>")
    reencoded = TOK.encode(TOK.decode(p1))  # what a new session renders + encodes
    assert reencoded != p1 and TOK.decode(reencoded) == TOK.decode(p1)
    return [_turn(p0, c0, lp=-0.1), _turn(p1, c1, lp=-0.2), _turn([*reencoded, *c1, *tool], TOK.encode("bye</a>"))]


def test_reencoded_history_does_not_extend_the_stream_and_the_error_says_where():
    turns = _forked_turns()
    with pytest.raises(TokenAlignmentError) as info:
        join_turns(turns)
    exc = info.value
    assert exc.turn == 2 and exc.position == len(turns[0].prompt_ids) + 3
    assert exc.stream[exc.position:exc.position + 2] == (0xC3, 0xA9) and exc.prompt[exc.position] == 256


def test_rebase_trains_only_tokens_sampled_on_the_engines_stream():
    turns = _forked_turns()
    seg = join_turns(turns, rebase=True)
    assert seg.rebased_turns == 1
    last = turns[-1]
    # the training stream is exactly the last call's prompt + completion: what the engine saw
    assert (*seg.prompt_ids, *seg.response_ids) == (*last.prompt_ids, *last.completion_ids)
    trained = [(t, lp) for t, m, lp in zip(seg.response_ids, seg.response_mask, seg.response_logprobs) if m]
    assert trained == [(t, -0.5) for t in last.completion_ids]  # earlier completions: context, mask 0
    assert not any(m for m in seg.response_mask[:len(seg.response_ids) - len(last.completion_ids)])


def test_rebase_still_refuses_a_stream_that_does_not_repeat_the_first_prompt():
    p0 = TOK.encode("<sys>a<asst>")
    with pytest.raises(TokenAlignmentError):
        join_turns([_turn(p0, [1, 2]), _turn(TOK.encode("<sys>b<asst>"), [3])], rebase=True)


def _js_stringify(arguments: str) -> str:
    return json.dumps(json.loads(arguments), separators=(",", ":"), ensure_ascii=False)


def test_pi_echo_without_empty_reasoning_and_restringified_arguments_continues_the_chain():
    registry = ChainRegistry()
    request = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    chain = registry.open_chain("rec0", request, registry.locate(request))
    args = '{"path":"/w/p.py","edits":[{"oldText": "a", "newText": "b"}],"timeout":30.0}'  # model's literal text
    generated = {"role": "assistant", "content": "", "reasoning_content": "",
                 "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "edit", "arguments": args}}]}
    registry.record_generation(chain, generated)
    echoed = {"role": "assistant", "content": "",
              "tool_calls": [{"id": "c0", "type": "function",
                              "function": {"name": "edit", "arguments": _js_stringify(args).replace("30.0", "30")}}]}
    located = registry.locate([*request, echoed, {"role": "tool", "tool_call_id": "c0", "content": "ok"}])
    assert located.kind is LocateKind.continue_ and located.chain is chain
    assert chain.messages[2] is generated  # the next turn renders what the engine sampled


def test_a_changed_call_still_forks():
    registry = ChainRegistry()
    request = [{"role": "user", "content": "u"}]
    chain = registry.open_chain("rec0", request, registry.locate(request))
    registry.record_generation(chain, {"role": "assistant", "content": "", "reasoning_content": "r", "tool_calls": [
        {"id": "c0", "type": "function", "function": {"name": "bash", "arguments": '{"command":"ls"}'}}]})
    other = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c0", "type": "function", "function": {"name": "bash", "arguments": '{"command":"pwd"}'}}]}
    assert registry.locate([*request, other]).kind is LocateKind.fork


class _Recorder:
    def __init__(self, turns):
        self.turns = turns

    async def close(self, session):
        return self.turns

    def decode(self, ids):
        return TOK.decode(ids)


class _Metrics:
    def reclassify(self, old, new):
        self.moved = (old, new)


def test_finish_sample_reports_a_token_alignment_loss():
    from yeto.rl.agentic import AgentTask, AgentTrajectory, Outcome
    from yeto.rl.agentic.types import TrialVerdict

    p0 = TOK.encode("<sys>a<asst>")
    turns = [_turn(p0, [0xC3, 0xA9]), _turn(TOK.encode("<sys>b<asst>"), [3])]
    task = AgentTask(task_id="t", agent="pi", provider="docker", trajectory_id="s1")
    verdict = TrialVerdict(rewards={"reward": 1.0}, trial_dir="/trials/x")
    trajectory = AgentTrajectory(task=task, outcome=Outcome.OK, reward=1.0, reason="", verdict=verdict)
    done = asyncio.run(finish_sample(trajectory, _Recorder(turns), "s1", _Metrics(), max_response_len=64))
    assert done.segments is None and done.trajectory.outcome is Outcome.ABORTED
    loss = done.loss
    assert loss["stage"] == "join_turns" and loss["turn"] == 1 and loss["trial_dir"] == "/trials/x"
    assert "does not extend" in loss["error"] and loss["position"] == 5
    assert loss["before_text"] == "<sys>" and loss["stream_text"].startswith("a")
    json.dumps(loss)
