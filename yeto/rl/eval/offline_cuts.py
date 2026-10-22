"""Offline evaluation of a run's round cuts on a fixed held-out prompt set (S27 R-5).

After (or during) a run, this evaluator walks the committed round cuts of a
checkpoint store, serves each cut through an :class:`EvalEngine`, samples the
held-out prompts, scores every response with a rule-based neutral reward
(:mod:`yeto.rl.rewards`) and writes one score curve: score against train step
and against wall clock. It is the offline variant of the eval island
(:mod:`yeto.rl.eval.island`); nothing here joins training.

Inputs::

    <store>/round-cuts/<cut_id>/manifest.json   committed cut (yeto.rl.engine.cut)
    held-out JSONL                              one row per prompt: prompt + label

Outputs (``out_dir``)::

    cuts/<cut_id>.json   per-cut result: every sample with its response and reward
    curve.json           ruler + one point per cut, sorted by train step
    curve.csv            the same points as a table

The "ruler" is the hash of everything that defines the score: held-out file
sha256, prompt/label keys, reward identity (module, qualname, source sha256),
sampling settings and samples per prompt. A per-cut result is reused only when
its ruler matches, so a rerun evaluates only new cuts and a changed ruler
never mixes into the curve.

Wall clock: the manifest's ``created_unix`` (set by the cut writer) when
present; an older cut without it falls back to the manifest file mtime (the
manifest is written last, so its mtime is the commit time, but a copy of the
store loses it). ``wall_time_source`` records which one was used. ``elapsed_s`` is relative to
``run_start`` when given, else to the earliest cut.

Engines: :class:`SglangHttpEngine` drives a running SGLang server (load the
cut's weights from disk, then batch ``/generate``). Turning a cut's trainer
shards into a directory SGLang can load is backend-specific and is injected as
``materialize`` (e.g. a Megatron-to-HF conversion). CPU tests use a fake
engine.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from yeto.rl.engine.cut import MANIFEST_NAME, CutError, load_manifest
from yeto.rl.engine.resume import ROUND_CUTS
from yeto.rl.rewards.registry import RewardPlugin, load_reward
from yeto.rl.rewards.types import Trajectory

from . import stats

CURVE_SCHEMA = "yeto.offline_cut_curve/v1"
CUT_RESULT_SCHEMA = "yeto.offline_cut_result/v1"
CSV_FIELDS = ("cut_id", "train_step", "next_rollout_id", "policy_version", "outer_version",
              "wall_time", "elapsed_s", "mean_reward", "success_rate", "stderr", "n_prompts",
              "n_samples")


class OfflineEvalError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# cuts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CutPoint:
    """One committed cut and where it sits on the run's step / time axes."""

    cut_id: str
    path: str
    train_step: int  # optimizer steps applied by the island (CutProgress.local_step)
    next_rollout_id: int
    policy_version: int
    policy_hash: str
    outer_version: int | None
    wall_time: float  # epoch seconds
    wall_time_source: str = "manifest_mtime"


def _wall_time(manifest: Any, entry: Path) -> dict[str, Any]:
    if manifest.created_unix is not None:
        return {"wall_time": float(manifest.created_unix), "wall_time_source": "manifest_created_unix"}
    return {"wall_time": (entry / MANIFEST_NAME).stat().st_mtime, "wall_time_source": "manifest_mtime"}


def discover_cuts(store: str | os.PathLike[str], *, cut_root: str | os.PathLike[str] | None = None,
                  skipped: list[dict[str, str]] | None = None) -> list[CutPoint]:
    """Every committed cut under ``<store>/round-cuts`` (or ``cut_root``), sorted by train step.

    A directory without a valid manifest is not a cut (an interrupted save); it is
    left out and, when ``skipped`` is given, listed there with the reason."""
    root = Path(cut_root) if cut_root is not None else Path(store) / ROUND_CUTS
    if not root.is_dir():
        raise OfflineEvalError(f"no cut directory at {root}")
    out = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            manifest = load_manifest(root, entry.name)
        except CutError as exc:
            if skipped is not None:
                skipped.append({"cut_id": entry.name, "reason": str(exc)})
            continue
        progress = manifest.progress
        outer = manifest.outer.get("outer_version")  # multi-island syncer step; absent on one island
        out.append(CutPoint(
            cut_id=manifest.cut_id, path=str(entry), train_step=int(progress.local_step),
            next_rollout_id=int(progress.next_rollout_id), policy_version=int(progress.policy_version),
            policy_hash=str(progress.policy_hash), outer_version=None if outer is None else int(outer),
            **_wall_time(manifest, entry)))
    out.sort(key=lambda c: (c.train_step, c.wall_time, c.cut_id))
    return out


# ---------------------------------------------------------------------------
# held-out set
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HeldOutItem:
    task_id: str
    prompt: Any
    label: Any


@dataclass(frozen=True)
class HeldOutSet:
    path: str
    sha256: str
    prompt_key: str
    label_key: str
    items: tuple[HeldOutItem, ...]

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, prompt_key: str = "prompt", label_key: str = "label",
             id_key: str | None = None, limit: int | None = None,
             expect_sha256: str | None = None) -> "HeldOutSet":
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expect_sha256 and expect_sha256 != digest:
            raise OfflineEvalError(f"held-out {path}: sha256 {digest} != expected {expect_sha256}")
        items = []
        for lineno, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if prompt_key not in row or label_key not in row:
                raise OfflineEvalError(f"held-out {path}:{lineno}: missing {prompt_key!r} or {label_key!r}")
            task_id = str(row[id_key]) if id_key else f"{lineno:06d}"
            items.append(HeldOutItem(task_id, row[prompt_key], row[label_key]))
            if limit is not None and len(items) >= limit:
                break
        if not items:
            raise OfflineEvalError(f"held-out {path} has no prompts")
        if len({i.task_id for i in items}) != len(items):
            raise OfflineEvalError(f"held-out {path}: duplicate task ids under {id_key!r}")
        return cls(str(path), digest, prompt_key, label_key, tuple(items))


# ---------------------------------------------------------------------------
# engine port
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GenerateRequest:
    task_id: str
    sample_index: int
    prompt: Any
    seed: int  # fixed per (task, sample): every cut sees the same sampling noise source


class EvalEngine(Protocol):
    def load(self, cut: CutPoint) -> None:
        """Serve ``cut``'s policy; later ``generate`` calls use it."""

    def generate(self, requests: Sequence[GenerateRequest], sampling: Mapping[str, Any]) -> list[str]:
        """One response text per request, in order."""


def request_seed(task_id: str, sample_index: int, base_seed: int) -> int:
    key = f"{base_seed}/{task_id}/{sample_index}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


class SglangHttpEngine:
    """A running SGLang server: ``/update_weights_from_disk`` per cut, batched ``/generate``.

    ``materialize(cut) -> model_path`` turns the cut into a directory SGLang can
    load (backend-specific conversion). ``post`` is injectable for tests."""

    def __init__(self, url: str, materialize: Callable[[CutPoint], str], *, batch_size: int = 64,
                 timeout_s: float = 3600.0, post: Callable[[str, Mapping[str, Any], float], Any] | None = None):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.url = url.rstrip("/")
        self.materialize = materialize
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        self._post = post or _http_post_json
        self.loaded: str | None = None

    def load(self, cut: CutPoint) -> None:
        model_path = self.materialize(cut)
        reply = self._post(f"{self.url}/update_weights_from_disk", {"model_path": model_path}, self.timeout_s)
        if not (isinstance(reply, Mapping) and reply.get("success")):
            raise OfflineEvalError(f"sglang could not load {model_path} for cut {cut.cut_id}: {reply!r}")
        self.loaded = cut.cut_id

    def generate(self, requests: Sequence[GenerateRequest], sampling: Mapping[str, Any]) -> list[str]:
        out: list[str] = []
        for start in range(0, len(requests), self.batch_size):
            batch = requests[start:start + self.batch_size]
            payload = {"text": [r.prompt for r in batch],
                       "sampling_params": [{**sampling, "sampling_seed": r.seed} for r in batch]}
            reply = self._post(f"{self.url}/generate", payload, self.timeout_s)
            if not isinstance(reply, list) or len(reply) != len(batch):
                raise OfflineEvalError(f"sglang /generate returned {type(reply).__name__} "
                                       f"for a batch of {len(batch)}")
            out.extend(str(r["text"]) for r in reply)
        return out


def _http_post_json(url: str, payload: Mapping[str, Any], timeout_s: float) -> Any:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 (operator-given URL)
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalSpec:
    heldout: HeldOutSet
    reward: RewardPlugin
    sampling: Mapping[str, Any] = field(default_factory=dict)
    samples_per_prompt: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.samples_per_prompt < 1:
            raise ValueError("samples_per_prompt must be >= 1")

    def ruler(self) -> dict[str, Any]:
        return {"heldout_sha256": self.heldout.sha256, "heldout_items": len(self.heldout.items),
                "prompt_key": self.heldout.prompt_key, "label_key": self.heldout.label_key,
                "reward": {"ref": self.reward.ref, **self.reward.identity()},
                "sampling": dict(self.sampling), "samples_per_prompt": self.samples_per_prompt,
                "seed": self.seed}

    def ruler_sha256(self) -> str:
        text = json.dumps(self.ruler(), sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()

    def requests(self) -> list[GenerateRequest]:
        return [GenerateRequest(item.task_id, k, item.prompt, request_seed(item.task_id, k, self.seed))
                for item in self.heldout.items for k in range(self.samples_per_prompt)]


def score_cut(cut: CutPoint, spec: EvalSpec, engine: EvalEngine) -> dict[str, Any]:
    """Load ``cut``, sample every held-out request, score; returns the per-cut result."""
    engine.load(cut)
    requests = spec.requests()
    responses = engine.generate(requests, spec.sampling)
    if len(responses) != len(requests):
        raise OfflineEvalError(f"engine returned {len(responses)} responses for {len(requests)} requests")
    labels = {item.task_id: item.label for item in spec.heldout.items}
    samples = []
    for req, response in zip(requests, responses):
        result = spec.reward.fn(Trajectory(prompt=req.prompt, response=response, label=labels[req.task_id],
                                           status="completed"))
        meta = dict(result.metadata or {})
        samples.append({"task_id": req.task_id, "sample_index": req.sample_index, "seed": req.seed,
                        "response": response, "reward": float(result.value),
                        "success": bool(meta.get("success", result.value >= 1.0))})
    return {"schema": CUT_RESULT_SCHEMA, "ruler_sha256": spec.ruler_sha256(), "cut": asdict(cut),
            "summary": summarize(samples), "samples": samples}


def summarize(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean reward, success rate and the task-level bootstrap standard error of the mean reward."""
    per_task: dict[str, list[float]] = {}
    successes = []
    for s in samples:
        per_task.setdefault(str(s["task_id"]), []).append(float(s["reward"]))
        successes.append(1.0 if s["success"] else 0.0)
    if not per_task:
        raise OfflineEvalError("no samples to summarize")
    task_means = {t: statistics.fmean(v) for t, v in per_task.items()}
    seed = int(hashlib.sha256("offline-cuts".encode()).hexdigest()[:16], 16)
    return {"mean_reward": statistics.fmean(task_means.values()),
            "success_rate": statistics.fmean(successes),
            "stderr": stats.bootstrap_se(task_means, seed=seed),
            "n_prompts": len(task_means), "n_samples": len(successes)}


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _reusable(path: Path, ruler_sha: str, cut: CutPoint) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if saved.get("ruler_sha256") != ruler_sha or saved.get("cut", {}).get("policy_hash") != cut.policy_hash:
        return None
    return saved


def evaluate_run(cuts: Sequence[CutPoint], spec: EvalSpec, engine: EvalEngine, out_dir: str | os.PathLike[str],
                 *, run_start: float | None = None, skipped: Sequence[Mapping[str, str]] = (),
                 log: Callable[[str], None] = lambda _msg: None) -> dict[str, Any]:
    """Evaluate every cut (reusing per-cut results under the same ruler) and write the curve."""
    if not cuts:
        raise OfflineEvalError("no committed cuts to evaluate")
    out = Path(out_dir)
    ruler_sha = spec.ruler_sha256()
    t0 = run_start if run_start is not None else min(c.wall_time for c in cuts)
    points = []
    for cut in cuts:
        path = out / "cuts" / f"{cut.cut_id}.json"
        result = _reusable(path, ruler_sha, cut)
        if result is None:
            log(f"evaluating cut {cut.cut_id} (step {cut.train_step})")
            result = score_cut(cut, spec, engine)
            _atomic_write_text(path, json.dumps(result, indent=1, sort_keys=True))
        else:
            log(f"reusing cut {cut.cut_id} (same ruler)")
        points.append({"cut_id": cut.cut_id, "train_step": cut.train_step,
                       "next_rollout_id": cut.next_rollout_id, "policy_version": cut.policy_version,
                       "outer_version": cut.outer_version, "wall_time": cut.wall_time,
                       "wall_time_source": cut.wall_time_source, "elapsed_s": cut.wall_time - t0,
                       **result["summary"]})
    curve = {"schema": CURVE_SCHEMA, "ruler": spec.ruler(), "ruler_sha256": ruler_sha,
             "run_start": t0, "run_start_source": "given" if run_start is not None else "earliest_cut",
             "skipped_cuts": list(skipped), "points": points}
    _atomic_write_text(out / "curve.json", json.dumps(curve, indent=1, sort_keys=True))
    _atomic_write_text(out / "curve.csv", _csv(points))
    return curve


def _csv(points: Sequence[Mapping[str, Any]]) -> str:
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for p in points:
        writer.writerow({k: ("" if p.get(k) is None or (isinstance(p.get(k), float) and math.isnan(p[k]))
                             else p[k]) for k in CSV_FIELDS})
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_callable(ref: str) -> Callable[..., Any]:
    import importlib

    module_name, sep, attr = ref.partition(":")
    if not sep:
        raise OfflineEvalError(f"{ref!r} is not 'module:function'")
    fn: Any = importlib.import_module(module_name)
    for part in attr.split("."):
        fn = getattr(fn, part)
    return fn


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m yeto.rl.eval.offline_cuts", description=__doc__.split("\n")[0])
    ap.add_argument("--store", required=True, help="checkpoint store root of the run")
    ap.add_argument("--cut-root", help="cut directory (default <store>/round-cuts)")
    ap.add_argument("--heldout", required=True, help="held-out JSONL (prompt + label per row)")
    ap.add_argument("--heldout-sha256", help="refuse a held-out file with another sha256")
    ap.add_argument("--prompt-key", default="prompt")
    ap.add_argument("--label-key", default="label")
    ap.add_argument("--id-key", help="row field used as task id (default: line number)")
    ap.add_argument("--limit", type=int, help="evaluate only the first N prompts")
    ap.add_argument("--reward", required=True, help="registered reward name or module:function")
    ap.add_argument("--samples-per-prompt", type=int, default=1)
    ap.add_argument("--sampling", default='{"temperature": 0.0, "max_new_tokens": 1024}',
                    help="sampling params JSON passed to the engine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-start", type=float, help="run start (epoch s) for elapsed_s")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sglang-url", required=True, help="running SGLang server, e.g. http://127.0.0.1:30000")
    ap.add_argument("--materialize", required=True,
                    help="module:function mapping a CutPoint to a model dir SGLang can load")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)

    heldout = HeldOutSet.load(args.heldout, prompt_key=args.prompt_key, label_key=args.label_key,
                              id_key=args.id_key, limit=args.limit, expect_sha256=args.heldout_sha256)
    spec = EvalSpec(heldout, load_reward(args.reward), json.loads(args.sampling),
                    args.samples_per_prompt, args.seed)
    skipped: list[dict[str, str]] = []
    cuts = discover_cuts(args.store, cut_root=args.cut_root, skipped=skipped)
    engine = SglangHttpEngine(args.sglang_url, _load_callable(args.materialize), batch_size=args.batch_size)
    curve = evaluate_run(cuts, spec, engine, args.out, run_start=args.run_start, skipped=skipped, log=print)
    print(json.dumps({"points": len(curve["points"]), "skipped": len(skipped), "out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
