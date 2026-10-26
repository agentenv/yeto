"""Agentic rounds of the engine sampler island (S28 G-120).

The engine sampler (:mod:`.engine_sampler`) writes one buffer group per prompt.
An agentic round writes one group per *task*: ``sampling.n`` agent trials of
the task (Harbor, e.g. Codex in a boat sandbox), each calling the island's own
engine through a token recorder (the records proxy,
:mod:`yeto.rl.harness.gateway.recorder`), so the group's samples carry the
exact ids the engine saw and sampled and their log-probs.

Nothing here is new agent logic; per trial the round runs the neutral layer:

1. ``recorder.open`` -> the OpenAI base URL the agent calls;
2. :func:`yeto.rl.agentic.run_agent_task` (admission, harness, wall-clock cap,
   verdict -> outcome; the reward is the Harbor verifier's);
3. :func:`yeto.rl.agentic.finish.finish_sample` (close, join the calls into one
   token stream, ABORTED on misalignment).

Lost trajectories: an ABORTED trial (infrastructure failure, Harbor timeout,
misaligned tokens) and a trial whose calls were served by another weight
version than the round's are left out of their group (never trained, never in
a group baseline, as Miles' ``check_no_aborted`` and verl's ``isolate_rows``).
A group with fewer than ``min_samples`` kept trials is not written. The round
event carries the counts (``agentic``: trials, kept, lost by reason, loss rate,
reward and response-length means).

Buffer sample (``engine_sampler.sample_record`` keys, so the trainer readers
are unchanged): ``tokens`` = prompt ids + response ids of the joined stream,
``response_length``, ``rollout_log_probs`` (0 at mask-0 positions),
``loss_mask`` (the response mask: 1 = sampled by the policy, 0 = tool output
and template tokens), ``reward``; ``metadata`` adds the trial outcome fields.

The round runs on one event loop thread kept for the island's life (the
recorder's HTTP server lives on that loop); the engine calls run in its
executor, sized to the in-flight limit.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import re
import statistics
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .engine_sampler import EngineRound, PromptItem
from .policy_versions import engine_weight_version
from .sampler_engine import FINISH_LENGTH, FINISH_STOP, SamplerEngineError

AGENT_TASK_KEY = "agent_task"
LOST_VERSION = "served_by_other_version"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


class LoopThread:
    """One asyncio loop on a daemon thread; :meth:`run` blocks for a coroutine's result."""

    def __init__(self, *, workers: int) -> None:
        self.loop = asyncio.new_event_loop()
        self.loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=max(4, int(workers))))
        self._thread = threading.Thread(target=self.loop.run_forever, name="agentic-round", daemon=True)
        self._thread.start()

    def run(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=10)


def session_id(*parts: Any) -> str:
    """A recorder session id (a URL path segment) from the round's names."""
    return _UNSAFE.sub("_", "-".join(str(p) for p in parts))[:180]


def agentic_sample(item: PromptItem, finished: Any, *, version: int, provenance: Mapping[str, str],
                   group_index: int, index: int) -> dict[str, Any]:
    """One kept trial as a buffer sample (module doc)."""
    trajectory, seg = finished.trajectory, finished.segments
    response = list(seg.response_ids)
    meta = trajectory.to_metadata()
    last = finished.turns[-1].finish_reason if finished.turns else None
    return {
        "group_index": group_index, "index": index, "prompt": item.prompt,
        "response": "", "label": item.label,
        "tokens": list(seg.prompt_ids) + response, "response_length": len(response),
        "rollout_log_probs": [float(x) for x in seg.response_logprobs],
        "loss_mask": [int(m) for m in seg.response_mask],
        "reward": float(trajectory.reward),
        "status": "truncated" if seg.truncated else "completed",
        "metadata": {**{k: v for k, v in item.metadata.items() if k != AGENT_TASK_KEY},
                     "prompt_id": item.prompt_id, "task_id": trajectory.task.task_id,
                     "trajectory_id": trajectory.task.trajectory_id,
                     "weight_version": engine_weight_version(int(version)),
                     "finish_reason": FINISH_LENGTH if seg.truncated else (last or FINISH_STOP),
                     "prompt_tokens_known": True, "provenance": dict(provenance),
                     "agent_turns": len(finished.turns),
                     "agent": {k: meta[k] for k in ("yeto_agent_outcome", "outcome_reason", "eval_report",
                                                    "agent_metrics", "trial_dir")}},
    }


class AgenticRound(EngineRound):
    """``generate(rollout_id)`` of the sampler loop for agentic tasks (module doc).

    ``recorder``: a :class:`~yeto.rl.agentic.TokenRecorder` in front of this
    island's engine; ``harness``: an :class:`~yeto.rl.agentic.AgentHarness`;
    ``task_defaults``: :meth:`AgentTask.from_mapping` defaults (provider, agent
    version, model name, budget); ``sampling_params``: what each recorder
    session is opened with (checked against the proxy's signed sampling)."""

    def __init__(self, engine: Any, writer: Any, *, recorder: Any, harness: Any, max_in_flight: int,
                 max_response_len: int, max_prompt_len: int | None = None, min_samples: int = 2,
                 task_defaults: Mapping[str, Any] | None = None,
                 sampling_params: Mapping[str, Any] | None = None,
                 loop: LoopThread | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("reward", lambda item, completion: 0.0)  # the verifier's reward is used
        super().__init__(engine, writer, **kwargs)
        if int(min_samples) < 1 or int(min_samples) > int(self.sampling.n):
            raise ValueError(f"min_samples must be in [1, n={self.sampling.n}], got {min_samples}")
        self.recorder = recorder
        self.harness = harness
        self.max_in_flight = int(max_in_flight)
        self.max_response_len = int(max_response_len)
        self.max_prompt_len = None if max_prompt_len is None else int(max_prompt_len)
        self.min_samples = int(min_samples)
        self.task_defaults = dict(task_defaults or {})
        self.sampling_params = dict(sampling_params or {"temperature": self.sampling.temperature,
                                                        "top_p": self.sampling.top_p})
        self.loop = loop or LoopThread(workers=self.max_in_flight + 8)
        self.lost: Counter[str] = Counter()  # over the island's life
        self.trials = 0

    def close(self) -> int:
        """End of the island: delete the pooled boat sandboxes (the account returns
        to the sandboxes other runs hold) and stop the loop; returns the VMs deleted."""
        deleted = 0
        try:
            from yeto.rl.harness.harbor.boat_pool import close_pools_now

            deleted = int(close_pools_now() or 0)
        finally:
            self.loop.close()
        self.emit("rl_agentic_close", boat_vms_deleted=deleted, lost_total=dict(sorted(self.lost.items())),
                  trials_total=self.trials)
        return deleted

    # ------------------------------------------------------------------ trials
    def _task(self, item: PromptItem, session: str) -> Any:
        from yeto.rl.agentic import AgentTask, HarnessConfigError

        row = item.metadata.get(AGENT_TASK_KEY)
        if not isinstance(row, Mapping):
            raise HarnessConfigError(f"prompt {item.prompt_id}: no {AGENT_TASK_KEY} mapping")
        try:
            task = AgentTask.from_mapping(row, defaults=self.task_defaults)
        except ValueError as exc:
            raise HarnessConfigError(f"prompt {item.prompt_id}: {exc}") from exc
        return dataclasses.replace(task, trajectory_id=session)

    async def _trial(self, item: PromptItem, session: str, gate: Any) -> Any:
        from yeto.rl.agentic import run_agent_task
        from yeto.rl.agentic.finish import finish_sample

        task = self._task(item, session)
        endpoint = await self.recorder.open(session, self.sampling_params)
        try:
            trajectory = await run_agent_task(task, self.harness, endpoint, gate)
        except BaseException:
            try:
                await self.recorder.close(session)
            except Exception:  # noqa: BLE001 - the trial's error is the one to report
                pass
            raise
        return await finish_sample(trajectory, self.recorder, session, gate.metrics,
                                   max_response_len=self.max_response_len, max_prompt_len=self.max_prompt_len)

    async def _round(self, rollout_id: int, items: Sequence[PromptItem], version: int) -> tuple[list, dict]:
        from yeto.rl.agentic import AdmissionGate

        gate = AdmissionGate(self.max_in_flight, policy_version=lambda: version)
        prefix = self.incarnation or "s"
        jobs = [[self._trial(item, session_id(prefix, f"r{rollout_id}", item.prompt_id, j), gate)
                 for j in range(self.sampling.n)] for item in items]
        flat = await asyncio.gather(*(job for group in jobs for job in group))
        n = self.sampling.n
        return [list(flat[i * n:(i + 1) * n]) for i in range(len(items))], gate.metrics.snapshot()

    # ------------------------------------------------------------------ round
    def __call__(self, rollout_id: int) -> None:
        from yeto.rl.agentic import Outcome

        rollout_id = int(rollout_id)
        if self.guard is not None:
            self.guard.check(rollout_id)
        items = list(self.prompts(rollout_id))
        version = self.engine.served_version()
        if version is None:
            raise SamplerEngineError("the engine serves no version; it cannot generate")
        results, outcomes = self.loop.run(self._round(rollout_id, items, version))
        if self.engine.served_version() != version:  # the loop loads only between rounds
            raise SamplerEngineError(f"served version changed from v{version} during round {rollout_id}")
        record = self.provenance.to_record()
        lost: Counter[str] = Counter()
        finish: Counter[str] = Counter()
        tokens, rewards, lengths, scored, small = 0, [], [], [], 0
        trials = sum(len(r) for r in results)
        for position, (item, group) in enumerate(zip(items, results)):
            kept = []
            for finished in group:
                if finished.segments is None or finished.trajectory.outcome is Outcome.ABORTED:
                    lost[str(finished.trajectory.reason).split(":", 1)[0] or "aborted"] += 1
                elif any(t.policy_version not in (None, version) for t in finished.turns):
                    lost[LOST_VERSION] += 1
                else:
                    kept.append(finished)
            if len(kept) < self.min_samples:
                small += 1
                continue
            samples = []
            for j, finished in enumerate(kept):
                sample = agentic_sample(item, finished, version=version, provenance=record,
                                        group_index=len(scored), index=len(scored) * self.sampling.n + j)
                finish[sample["metadata"]["finish_reason"]] += 1
                tokens += sample["response_length"]
                rewards.append(sample["reward"])
                lengths.append(int(sum(sample["loss_mask"])))
                samples.append(sample)
            scored.append((item, samples))
        self.lost.update(lost)
        self.trials += trials
        n_lost = sum(lost.values())
        agentic = {"trials": trials, "kept": trials - n_lost, "lost": dict(sorted(lost.items())),
                   "lost_rate": round(n_lost / trials, 4) if trials else 0.0,
                   "groups_below_min": small, "outcomes": dict(outcomes),
                   "reward_mean": round(statistics.fmean(rewards), 4) if rewards else None,
                   "policy_tokens_mean": round(statistics.fmean(lengths), 1) if lengths else None,
                   "lost_total": dict(sorted(self.lost.items())), "trials_total": self.trials}
        self._write_scored(rollout_id, version, scored, finish=finish, tokens=tokens, record=record,
                           samples=sum(len(s) for _, s in scored), extra={"agentic": agentic})


# ----------------------------------------------------------------------------- spec
def round_factory(spec: Mapping[str, Any], engine: Any, environ: Mapping[str, str]) -> Callable[..., AgenticRound]:
    """``make_round`` of :func:`~.engine_sampler.build_engine_sampler` from the
    spec's ``agentic`` section (``engine_sampler_entry``):

    * ``template`` (a :mod:`yeto.rl.harness.chat_templates` name), ``tokenizer``
      (default ``base_dir``), ``model`` (the model name the agent sends);
    * ``turn_max_tokens``: the signed ``max_tokens`` of each model call;
    * ``max_response_len`` / ``max_prompt_len``: the trainer's row widths (a
      longer stream is cut / ABORTED);
    * ``max_in_flight``: concurrent trials; ``min_samples`` (default 2);
    * ``proxy``: ``{"host", "port"}`` of the records proxy; the public base and
      the path secret come from ``YETO_RECORDS_PROXY_PUBLIC_BASE`` and
      ``YETO_RECORDS_PROXY_SECRET_FILE`` (the Modal tunnel, written in the container);
    * ``task_defaults``: :meth:`AgentTask.from_mapping` defaults; ``provider``
      falls back to ``HARBOR_ENV_TYPE``.

    Harbor's settings (task directories, trials, provider kwargs) are the
    environment variables both backends read (``HarborSettings.from_env``)."""
    from functools import partial
    from pathlib import Path

    from yeto.rl.agentic import HarnessConfigError
    from yeto.rl.harness.chat_templates import ChatTemplate
    from yeto.rl.harness.gateway import records_env
    from yeto.rl.harness.gateway.proxy import RecordsProxy
    from yeto.rl.harness.gateway.recorder import ProxyTokenRecorder
    from yeto.rl.harness.harbor import HarborHarness, HarborSettings, build_trial_config
    from yeto.rl.harness.harbor.boat_client import check_in_flight_cap

    section = dict(spec["agentic"])
    try:
        sampling = dict(spec["sampling"])
        signed = {"max_tokens": int(section["turn_max_tokens"]),
                  "temperature": float(sampling.get("temperature", 1.0)), "top_p": float(sampling.get("top_p", 1.0))}
        tokenizer = section.get("tokenizer") or spec.get("base_dir")
        template = ChatTemplate.load(str(section["template"]), tokenizer_path=tokenizer)
        model = str(section.get("model", "model"))
        max_in_flight = int(section["max_in_flight"])
        max_response_len = int(section["max_response_len"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HarnessConfigError(f"spec agentic section: bad or missing {exc}") from exc
    proxy_cfg = dict(section.get("proxy") or {})
    secret_file = str(environ.get(records_env.PROXY_SECRET_FILE_ENV, "") or "").strip()
    public_base = str(environ.get(records_env.PROXY_PUBLIC_BASE_ENV, "") or "").strip() or None
    if public_base and not secret_file:
        raise HarnessConfigError("a public records-proxy base needs a path secret (the tunnel is public)")
    proxy = RecordsProxy(engine, template, model=model, signed_sampling=signed,
                         sampler_id=str(spec.get("sampler_id", "records-proxy")))
    recorder = ProxyTokenRecorder(proxy, host=str(proxy_cfg.get("host", "127.0.0.1")),
                                  port=int(proxy_cfg.get("port", 0)), public_base=public_base,
                                  path_secret=Path(secret_file).read_text().strip() if secret_file else None)
    settings = HarborSettings.from_env()
    harness = HarborHarness(partial(build_trial_config, settings=settings), setup_retries=settings.setup_retries)
    defaults = {"model_name": model, **dict(section.get("task_defaults") or {})}
    provider = str(environ.get("HARBOR_ENV_TYPE", "") or "").strip().lower()
    if provider and "provider" not in defaults:
        defaults["provider"] = provider
    if defaults.get("provider") == "boat":
        check_in_flight_cap(max_in_flight)
    loop = LoopThread(workers=max_in_flight + 8)

    def make(engine_: Any, writer: Any, **kwargs: Any) -> AgenticRound:
        return AgenticRound(engine_, writer, recorder=recorder, harness=harness, max_in_flight=max_in_flight,
                            max_response_len=max_response_len,
                            max_prompt_len=None if section.get("max_prompt_len") is None
                            else int(section["max_prompt_len"]),
                            min_samples=int(section.get("min_samples", 2)), task_defaults=defaults,
                            sampling_params={"temperature": signed["temperature"], "top_p": signed["top_p"]},
                            loop=loop, **kwargs)

    return make
