"""The spot sampler entry: the rollout runtime on the syncer data path.

Compatibility entry (same spec, same command line); the runtime is
:mod:`yeto.rl.engine.engine_sampler_entry` with ``data_path: syncer``
(:mod:`yeto.rl.engine.rollout_data_path`, :mod:`.syncer_path`).

A spot sampler is the engine sampler island (:mod:`yeto.rl.engine.engine_sampler`,
:mod:`yeto.rl.engine.engine_sampler_entry`): ``SamplerLoop`` admits rounds
by backlog, ``PolicyFollower`` loads new policies through ``PortReleaseLoader``
with the post-load checks, ``EngineRound`` generates, scores and writes each
group with ``GroupWriter``, ``SamplerMembership`` and the reclaim marker
handle replacement. Only its two channels differ:

* weights: :class:`~.release_store.SubscriberReleaseStore` (a syncer
  subscriber, no store) instead of the volume's ``PolicyReleaseStore``;
* samples: :class:`~.relay_buffer.RelaySampleBuffer` (the head's relay)
  instead of the volume's ``FileSampleBuffer``.

It stops when the syncer says the run finished, when the syncer link stays
down for ``syncer_lost_s``, when the relay did not answer for ``relay_gone_s``,
or on the reclaim marker (PREEMPTED). Every relay item names the cut that
generated each of its versions; the base weights served before the
first cut are named by the digest of ``base_dir``.

``python -m yeto.rl.spot_rollout.sampler --spec SPEC.json``: the engine
sampler spec (``engine``, ``sampling``, ``prompts``, ``reward``, ``checks``,
``event_tape``, ``max_prompt_tokens``, ``max_backlog_groups``, ``base_dir``)
plus ``syncer`` (host:port of the actor syncer that serves the subscribers and
the relay), ``channel`` (the training island it feeds), ``base_model`` (HF
directory with config and tokenizer; default ``base_dir``), ``work_dir``;
optional ``num_streams`` (4), ``reload_every`` (1 merged version),
``converter`` (``module:callable``; default Miles' ``megatron_to_hf``),
``model_name``, ``layout_timeout_s`` (1800), ``syncer_lost_s`` (1800), ``relay_gone_s`` (900),
``num_rollouts`` (no limit), ``store`` (local membership records; default
``<work_dir>/store``), ``telemetry`` (false: no device telemetry). The subscriber id is
:func:`~.release_store.subscriber_id_for` of ``sampler_id`` (the task
dispenser's sampler holder id); the HMAC key comes from ``YETO_ISLAND_HMAC_KEY``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from yeto.rl.engine.rollout_data_path import NO_LIMIT_ROUNDS  # noqa: F401 - the earlier name

from .syncer_path import SpotChannels, SyncerDataPath  # noqa: F401 - the earlier names


def label_boat_sandboxes(spec: Mapping[str, Any], incarnation: str,
                         environ: Mapping[str, str] | None = None) -> list[str]:
    """With ``YETO_BOAT_RUN_ID`` set, label this process's boat sandboxes
    with the island (``sampler_id``) and this incarnation, and delete the
    sandboxes an earlier incarnation of this island left (a preempted VM runs
    no exit cleanup). Returns the deleted sandbox ids. The labels go to
    ``os.environ`` (the boat sandboxes of this process read them there)."""
    import os

    from yeto.rl.harness.harbor import boat_labels

    env = os.environ if environ is None else environ
    run = str(env.get(boat_labels.RUN_ID_ENV, "")).strip()
    if not run:
        return []
    labels = {boat_labels.RUN_ID_ENV: run, boat_labels.ISLAND_ENV: str(spec["sampler_id"]),
              boat_labels.INCARNATION_ENV: str(incarnation)}
    os.environ.update(labels)
    deleted = boat_labels.sweep_stale_incarnations(labels)
    print(f"[spot-sampler] boat labels: run {run} island {spec['sampler_id']} incarnation {incarnation}; "
          f"deleted {len(deleted)} sandbox(es) of earlier incarnations", flush=True)
    return deleted


def run_spec(spec: Mapping[str, Any], *, environ: Mapping[str, str] | None = None,
             incarnation: str | None = None, resource_sampler: Any = None,
             reclaim_deadline: Any = None) -> tuple[int, SpotChannels | None]:
    """Run the rollout runtime of ``spec`` on the syncer data path; returns
    (rounds, data path). A thin shell over :func:`yeto.rl.engine.engine_sampler_entry.run`."""
    from yeto.rl.engine import engine_sampler_entry as entry

    spec = dict(spec)
    spec.setdefault("data_path", "syncer")
    made: dict[str, Any] = {}
    if resource_sampler is None and spec.get("telemetry") is False:
        resource_sampler = lambda emit: None  # noqa: E731 - CPU tests: no device telemetry
    rounds = entry.run(spec, environ=environ, incarnation=incarnation, resource_sampler=resource_sampler,
                       on_data_path=lambda path: made.setdefault("path", path),
                       reclaim_deadline=reclaim_deadline)
    return rounds, made.get("path")


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m yeto.rl.spot_rollout.sampler --spec SPEC.json``: the engine
    sampler entry with ``data_path`` defaulting to ``syncer`` (the reclaim
    deadline and exit code are the entry's)."""
    from yeto.rl.engine.engine_sampler_entry import main as entry_main

    return entry_main(argv, default_data_path="syncer", prog="python -m yeto.rl.spot_rollout.sampler")


if __name__ == "__main__":
    raise SystemExit(main())
