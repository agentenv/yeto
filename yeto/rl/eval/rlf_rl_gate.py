"""Real-GPU RLF RL gate runner and evidence chain (rlf-codex-rl task 5.8).

One gate run, in this order:

1. admit the CTF task packages (``fixture-ctf-ezshell``: solvable in ~2
   turns, provides success; ``fixture-ctf-ret2win``: provides reward
   variance) into the RLF catalog and render Miles ``--prompt-data`` rows
   (``yeto.rl.rlf.catalog.write_prompt_rows`` via :func:`write_prompt_data`);
2. launch one Miles GRPO island on the small ``qwen35_08b`` profile (0.8B,
   the S17-verified harness stack; LoRA, 1xGPU) with
   ``--custom-rm-path yeto.rl.adapters.miles.rewards.rlf_success`` and
   ``--advantage-estimator grpo`` (both already fixed by
   the Miles run translation) and the W4 RLF environment provider
   (``YETO_HARNESS_ENVIRONMENT_PROVIDER=yeto.cloud.modal_reward_env:rlf_modal_provider``
   + ``YETO_RLF_CTF_TASKS_DIR`` / ``YETO_RLF_CTF_FLAGS_JSON``).  The 30B
   capability surface is covered separately by the 5.6 baseline; 30B RL
   training is a follow-up item (design decision 2026-10-11);
3. after training, collect the gate evidence from real learner/rollout
   artifacts — never hand-filled:
   - rewards/success: the island's completed-groups checkpoint
     (``--completed-groups-path``), i.e. the serialized Miles samples the
     learner actually trained on, each carrying the ``rlf_success``
     reward value and ``metadata.success``;
   - advantages: recomputed per GRPO group with
     ``yeto.rl.rlf.contracts.group_advantages`` (the computation of record
     designated by W3 for paths where the learner does not export
     per-sample advantages);
   - policy update: ``contracts.policy_update_evidence`` over a parameter
     summary of the initial adapter vs the final trained adapter
     (see "代拍板" note below);
4. judge with ``contracts.rl_gate`` (hard criteria: non-empty reward,
   reward_std>0, success_count>0, advantage_abs_sum>0, policy_update) and
   write the evidence JSON (reward detail, success, advantage_abs_sum,
   policy-update digest before/after).  All-zero or constant reward MUST
   fail the gate.

代拍板 (policy-update collection, pending main-agent sign-off): the Miles
fork exports no per-step parameter snapshot hook, so the minimal viable
collection is a parameter summary (per-tensor ``float(sum)``) of the
initial LoRA adapter (``--initial-adapter`` snapshot, or the base snapshot
at t0) versus the final saved adapter, fed to
``contracts.policy_update_evidence``.  Rationale: it reuses W3's digest
contract unchanged, proves the optimizer step changed policy parameters,
and needs no fork patch; it cannot attribute the change to a specific
optimizer step, which the gate does not require.

代拍板 v2 (2026-10-11): after gate attempt 1 lost every island-side file
on teardown, the collection of record is the island event tape's
``rl_publication`` payload hashes (initial vs final policy version) — see
:func:`collect_policy_update_from_tape`.  The adapter-file path remains as
``--adapter-before/--adapter-after`` for runs that arrange adapter
snapshots themselves.

Flag hygiene: trusted flags live only in ``YETO_RLF_CTF_FLAGS_JSON`` (the
Modal Secret).  Every persisted evidence string is scrubbed through
``yeto.rl.rlf.modal.scrub`` with the flag values; flags are never logged.

The verdict is announced only for evidence collected from the real island
artifacts (``collector="miles_island"``).  CPU tests inject fake learners
and only exercise the orchestration: an injected collection can never
produce a passing verdict, so the gate cannot be declared green off GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from yeto.rl.rlf.catalog import import_catalog, write_prompt_data
from yeto.rl.rlf.contracts import (
    AdmissionError,
    PolicyUpdateEvidence,
    group_advantages,
    policy_update_evidence,
    rl_gate,
)
from yeto.rl.rlf.modal import scrub

EVIDENCE_SCHEMA = "yeto.rlf.rl_gate.evidence/v1"
DEFAULT_PROFILE = "qwen35_08b"
DEFAULT_TASK_IDS = ("fixture-ctf-ezshell", "fixture-ctf-ret2win")
GATE_TASKS_DIR_ENV = "YETO_RLF_CTF_TASKS_DIR"
GATE_FLAGS_ENV = "YETO_RLF_CTF_FLAGS_JSON"


class GateEvidenceError(RuntimeError):
    """Collected artifacts cannot support a gate judgment (fail closed)."""


@dataclass
class GateConfig:
    """One 5.8 gate run.  Flags never leave the trusted side (never logged)."""

    tasks_dir: str
    flags: Mapping[str, str]                    # task_id -> trusted flag
    profile: str = DEFAULT_PROFILE
    task_ids: tuple[str, ...] = DEFAULT_TASK_IDS
    repeat: int = 2                             # prompt rows per task
    rounds: int = 2                             # Miles --num-rollout
    groups_per_round: int = 4                   # --rollout-batch-size
    samples_per_group: int = 4                  # --n-samples-per-prompt
    lora_rank: int = 16
    evidence: Mapping[str, Any] = field(
        default_factory=lambda: {"runtime_quality_gate": {"healthy": True}})

    def __post_init__(self) -> None:
        missing = [t for t in self.task_ids if t not in self.flags]
        if missing:
            raise GateEvidenceError(f"no trusted flag for tasks: {missing}")

    def digest(self) -> str:
        """Config digest WITHOUT the flags (safe to persist)."""
        blob = json.dumps(
            {"profile": self.profile, "task_ids": sorted(self.task_ids),
             "repeat": self.repeat, "rounds": self.rounds,
             "groups_per_round": self.groups_per_round,
             "samples_per_group": self.samples_per_group,
             "lora_rank": self.lora_rank,
             "flag_sha256": {t: hashlib.sha256(self.flags[t].encode()).hexdigest()
                             for t in sorted(self.flags)}},
            sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------- prompt data

def prepare_prompt_data(config: GateConfig, out_path: str | Path) -> Path:
    """Admit the gate task packages and write Miles ``--prompt-data`` JSONL."""
    roots = [Path(config.tasks_dir) / task_id for task_id in config.task_ids]
    catalog = import_catalog(roots, evidence=dict(config.evidence))
    return write_prompt_data(catalog, out_path, repeat=config.repeat)


# ---------------------------------------------------- rollout sample collection

def _load_island_checkpoint(path: str | Path) -> Mapping[str, Any]:
    """Real loader: the island's completed-groups checkpoint (lazy torch)."""
    import torch  # GPU/learner side only; CPU tests inject ``loader``

    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or "completed_groups" not in payload:
        raise GateEvidenceError(f"island checkpoint {path} lacks completed_groups")
    return payload


def collect_rollout_samples(
    checkpoint_path: str | Path,
    *,
    loader: Callable[[str | Path], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Real reward/success collection from the learner's completed groups.

    Returns one record per sample: ``group_index``/``index`` (position in the
    checkpoint), ``task_id``, ``reward`` and ``success``.  A sample without a
    graded ``reward`` or ``metadata.success`` aborts the gate (fail closed) —
    silent zeros are exactly what this gate exists to reject.
    """
    payload = (loader or _load_island_checkpoint)(checkpoint_path)
    groups = payload.get("completed_groups")
    if not isinstance(groups, list) or not groups:
        raise GateEvidenceError("island checkpoint has no completed groups")
    records: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        if not isinstance(group, list):
            raise GateEvidenceError(f"completed group {group_index} is not a sample list")
        for index, sample in enumerate(group):
            if isinstance(sample, list):  # compaction segments share one outcome
                if not sample:
                    raise GateEvidenceError("empty segment list in completed group")
                sample = sample[0]
            if not isinstance(sample, Mapping):
                raise GateEvidenceError(f"sample {group_index}/{index} is not a mapping")
            reward = sample.get("reward")
            metadata = sample.get("metadata") or {}
            if not isinstance(reward, (int, float)) or isinstance(reward, bool):
                raise GateEvidenceError(f"sample {group_index}/{index} lacks a numeric reward")
            if "success" not in metadata:
                raise GateEvidenceError(
                    f"sample {group_index}/{index} lacks metadata.success "
                    "(rlf_success reward must write it)")
            records.append({"group_index": group_index, "index": index,
                            "task_id": metadata.get("rlf_task_id") or metadata.get("task_id"),
                            "reward": float(reward),
                            "success": bool(metadata["success"])})
    return records


def collect_rollout_samples_from_tape(
    tape_path: str | Path,
) -> list[dict[str, Any]]:
    """Real reward/success collection from ``rl_sample_outcome`` tape events.

    rlf-codex-rl 5.8 代拍板: on the ports engine the island checkpoint only
    ever holds an empty completed-group queue, so sample-level evidence rides
    the event tape (mirrored to the Modal volume, survives the island). The
    same fail-closed rules as the checkpoint path: any gap in reward/success
    aborts the gate.

    Two event shapes are accepted (rlf-codex-rl 5.8 attempt 8 lesson: the
    ports launcher binds ``--rollout-all-samples-process-path`` to
    ``rollout_meta_hook.extract_rollout_metadata``, so the legacy
    ``queue_completed_groups`` hook never runs there; the meta hook already
    reports per-sample records which the driver tapes as
    ``rl_trajectory_reward`` behind ``--rl-observe-timeline``):

    - ``rl_sample_outcome`` (tapes of the removed legacy engine, 054c20d5);
    - ``rl_trajectory_reward`` (rl-fn-codex-rollout 1.0 observe-only pipeline;
      ``sample_index`` fills ``index``).
    """
    outcomes: list[dict[str, Any]] = []
    for line in Path(tape_path).expanduser().read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        if record.get("event") not in ("rl_sample_outcome", "rl_trajectory_reward"):
            continue
        outcomes.append(record)
    if not outcomes:
        raise GateEvidenceError(
            "event tape has no rl_sample_outcome/rl_trajectory_reward records"
        )
    records: list[dict[str, Any]] = []
    for order, outcome in enumerate(outcomes):
        reward = outcome.get("reward")
        if not isinstance(reward, (int, float)) or isinstance(reward, bool):
            raise GateEvidenceError(f"tape sample {order} lacks a numeric reward")
        if outcome.get("success") is None:
            raise GateEvidenceError(
                f"tape sample {order} lacks success "
                "(rlf_success reward must write it)"
            )
        records.append({"group_index": int(outcome.get("group_index", order)),
                        "index": int(outcome.get("index", outcome.get("sample_index", 0))),
                        "task_id": outcome.get("task_id"),
                        "reward": float(reward),
                        "success": bool(outcome["success"])})
    return records


# ------------------------------------------------------- policy update evidence

def _summarize_params_real(path: str | Path) -> dict[str, float]:
    """Real loader: per-tensor ``float(sum)`` over a safetensors/torch file or dir."""
    tensors: dict[str, Any] = {}
    root = Path(path).expanduser()
    files = sorted(root.rglob("*.safetensors")) if root.is_dir() else [root]
    if files and all(f.suffix == ".safetensors" for f in files):
        from safetensors.torch import load_file  # lazy, learner side only

        for file in files:
            tensors.update(load_file(str(file)))
    else:
        import torch  # lazy, learner side only

        payload = torch.load(root, map_location="cpu", weights_only=True)
        state = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
        if not isinstance(state, Mapping):
            raise GateEvidenceError(f"cannot read parameter tensors from {path}")
        tensors = dict(state)
    summary: dict[str, float] = {}
    for name, tensor in tensors.items():
        try:
            summary[str(name)] = float(tensor.float().sum())
        except (TypeError, AttributeError, RuntimeError) as error:
            raise GateEvidenceError(f"non-tensor parameter {name!r} in {path}") from error
    if not summary:
        raise GateEvidenceError(f"no parameter tensors found in {path}")
    return summary


def collect_policy_update(
    before_path: str | Path,
    after_path: str | Path,
    *,
    step: int,
    loader: Callable[[str | Path], Mapping[str, float]] | None = None,
) -> PolicyUpdateEvidence:
    """Digest the initial vs final adapter parameters (see module 代拍板 note)."""
    load = loader or _summarize_params_real
    return policy_update_evidence(dict(load(before_path)), dict(load(after_path)), step=step)


def collect_policy_update_from_tape(
    tape_path: str | Path,
    *,
    step: int,
) -> PolicyUpdateEvidence:
    """Policy-update proof from the island event tape's ``rl_publication`` records.

    代拍板 v2 (2026-10-11, gate attempt 2): the first gate run lost every
    island-side artifact when the island was torn down after the trainer
    crash.  The event tape is the one channel that is mirrored to a Modal
    Volume (``--modal-tape-volume``) and pulled even on failure, and the
    driver publishes one ``rl_publication`` event per policy version with a
    learner-computed payload hash.  Comparing the lowest- and
    highest-version payload hashes proves an optimizer step changed policy
    parameters without any island-side file surviving — the same
    step-attribution limitation as the initial-vs-final adapter digest
    (代拍板 v1), with strictly better survivability.

    Fail closed: fewer than two distinct published policy versions, or a
    publication without a payload hash, aborts the gate.
    """
    publications: dict[int, str] = {}
    with Path(tape_path).expanduser().open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # prefixed/garbled lines are not gate evidence
            if not isinstance(event, Mapping) or event.get("event") != "rl_publication":
                continue
            version = event.get("policy_version")
            digest = event.get("sync/publication_payload_hash")
            if not isinstance(version, int) or not isinstance(digest, str) or not digest:
                raise GateEvidenceError(
                    f"rl_publication event lacks policy_version/payload hash: {event!r}"
                )
            publications.setdefault(version, digest)
    if len(publications) < 2:
        raise GateEvidenceError(
            f"event tape has {len(publications)} published policy version(s); "
            "the gate needs the initial and at least one post-training publication"
        )
    first, last = min(publications), max(publications)
    before_digest, after_digest = publications[first], publications[last]
    updated = before_digest != after_digest
    return PolicyUpdateEvidence(
        int(step), updated, before_digest, after_digest,
        ("policy_payload",) if updated else (),
    )


# ------------------------------------------------------------------ gate rows

def build_gate_rows(
    samples: Sequence[Mapping[str, Any]],
    update: PolicyUpdateEvidence,
) -> list[dict[str, Any]]:
    """Gate rows: real rewards/success, GRPO group advantages, real update flag.

    Advantages come from ``contracts.group_advantages`` per rollout group
    (reward minus group mean) — the designated computation of record; the
    ``policy_update`` flag comes from the collected PolicyUpdateEvidence,
    never from a literal.
    """
    by_group: dict[Any, list[Mapping[str, Any]]] = {}
    for record in samples:
        by_group.setdefault(record["group_index"], []).append(record)
    rows: list[dict[str, Any]] = []
    for group_index in sorted(by_group):
        group = by_group[group_index]
        advantages = group_advantages([r["reward"] for r in group])
        for record, advantage in zip(group, advantages):
            rows.append({"reward": float(record["reward"]),
                         "advantage": float(advantage),
                         "metadata": {"success": bool(record["success"]),
                                      "task_id": record.get("task_id"),
                                      "group_index": group_index},
                         "policy_update": update.updated})
    return rows


# ---------------------------------------------------------------------- run

def run_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    collector: str,
    config: GateConfig | None = None,
    update: PolicyUpdateEvidence | None = None,
) -> dict[str, Any]:
    """Judge the rows with ``rl_gate`` and assemble the evidence record.

    ``collector`` names the evidence source; only ``"miles_island"`` (the real
    learner artifacts) may produce ``verdict=True``.  Anything else (CPU fake
    learners in tests, manual drills) exercises the orchestration but the
    verdict stays False — the gate is never announced off the real run.
    """
    try:
        result = rl_gate(list(rows))
        gate: dict[str, Any] = {"passed": True, "result": result}
    except AdmissionError as error:
        gate = {"passed": False, "error": str(error)}
    real = collector == "miles_island"
    verdict = bool(gate["passed"]) and real
    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "collected_at": time.time(),
        "collector": collector,
        "verdict": verdict,
        "verdict_note": (
            "gate criteria evaluated on real Miles island artifacts"
            if real else
            "orchestration-only collection (injected/fake learner): "
            "the 5.8 gate is never announced off the real GPU run"),
        "gate": gate,
        "rows": [dict(row) for row in rows],
    }
    if update is not None:
        evidence["policy_update_evidence"] = {
            "step": update.step, "updated": update.updated,
            "before_digest": update.before_digest, "after_digest": update.after_digest,
            "changed": list(update.changed)}
    if config is not None:
        evidence["config_digest"] = config.digest()
        evidence["profile"] = config.profile
        evidence["task_ids"] = sorted(config.task_ids)
    return evidence


def write_evidence(evidence: Mapping[str, Any], out_path: str | Path,
                   *, secrets: Sequence[str] = ()) -> Path:
    """Persist the evidence JSON with every trusted flag scrubbed."""
    text = json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scrub(text, *secrets) + "\n")
    return path


def main(argv: list[str] | None = None) -> None:
    """``python -m yeto.rl.eval.rlf_rl_gate {prompt-data|gate} ...`` (island side)."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pd = sub.add_parser("prompt-data", help="admit packages, write Miles --prompt-data")
    pd.add_argument("--tasks-dir", required=True)
    pd.add_argument("--out", required=True)
    pd.add_argument("--repeat", type=int, default=2)
    pd.add_argument("--flags-json", default=None,
                    help="path to the trusted flags JSON (never logged)")
    gate = sub.add_parser("gate", help="collect island artifacts, judge, write evidence")
    gate.add_argument("--completed-groups", default=None,
                    help="island checkpoint with completed groups; optional when "
                    "--tape-events carries rl_sample_outcome/rl_trajectory_reward "
                    "records (ports engine)")
    gate.add_argument("--adapter-before", default=None)
    gate.add_argument("--adapter-after", default=None)
    gate.add_argument("--tape-events", default=None,
                    help="island event tape (rl-island-*.jsonl): policy-update proof "
                    "from rl_publication payload hashes (代拍板 v2); mutually "
                    "exclusive with --adapter-before/--adapter-after")
    gate.add_argument("--step", type=int, required=True)
    gate.add_argument("--out", required=True)
    gate.add_argument("--flags-json", default=None)
    args = parser.parse_args(argv)

    secrets: list[str] = []
    flags_path = getattr(args, "flags_json", None)
    if flags_path:
        flags = json.loads(Path(flags_path).read_text())
        secrets = [str(v) for v in flags.values()]
    if args.command == "prompt-data":
        if not flags_path:
            raise SystemExit("--flags-json is required (fail closed)")
        config = GateConfig(tasks_dir=args.tasks_dir, flags=flags, repeat=args.repeat)
        prepare_prompt_data(config, args.out)
        return
    if args.completed_groups:
        samples = collect_rollout_samples(args.completed_groups)
    elif args.tape_events:
        samples = collect_rollout_samples_from_tape(args.tape_events)
    else:
        raise SystemExit("rollout samples need --completed-groups or --tape-events")
    adapter_pair = (args.adapter_before, args.adapter_after)
    if args.tape_events and any(adapter_pair):
        raise SystemExit("--tape-events is mutually exclusive with --adapter-before/--adapter-after")
    if args.tape_events:
        update = collect_policy_update_from_tape(args.tape_events, step=args.step)
        update_source = "event_tape"
    elif all(adapter_pair):
        update = collect_policy_update(args.adapter_before, args.adapter_after, step=args.step)
        update_source = "adapter_files"
    else:
        raise SystemExit("policy-update proof needs --tape-events or --adapter-before + --adapter-after")
    rows = build_gate_rows(samples, update)
    evidence = run_gate(rows, collector="miles_island", update=update)
    evidence["policy_update_source"] = update_source
    evidence["sample_source"] = "completed_groups" if args.completed_groups else "event_tape"
    write_evidence(evidence, args.out, secrets=secrets)
    if not evidence["verdict"]:
        raise SystemExit(f"RL gate failed: {evidence['gate']}")


if __name__ == "__main__":
    main()
