"""E2 GPU acceptance harness, in-learner part (rl-infra-spec 4.2/4.3, plan-v3 G-4.2 + G-4.3).

Runs INSIDE the ports learner, on the real island (real Miles actor, rollout
manager, SGLang engines), instead of ``driver.run()`` when a harness plan is
present (entry.py hook; patch ``infra-e2-entry-harness-v1.patch`` for the
entry owner). Selected by ``<repo>/yeto-rl-e2-harness.json`` (present only in
an experiment snapshot, like ``yeto-rl-fault-injection.json``) or by
``YETO_RL_E2_HARNESS=<path>``. Absent: nothing changes.

Sequence (plan-v3 §1; one learner run per configuration C1/C2):

0. every rank reports the determinism settings (env + Megatron
   ``deterministic_mode``); any rank off -> ``environment_blocked``, stop;
1. handshake, start, publish; two normal rounds (``driver.run_round``);
2. at the safe point: ``save_cut`` (cut C0);
3. G-4.2 (g): ``restore_cut`` on the live trainer -> must be refused, state unchanged;
4. G-4.3 arm A: generate the frozen batch B3, train B3 on the live trainer
   (refs retained; no re-publication in either arm), read back every rank's summary;
5. ``rebuild_same_shape`` whose ``restore``:
   G-4.2 (a)-(f) on the freshly built trainer (each refused, state unchanged),
   then the valid restore of C0 (digests checked by ``restore_cut``);
6. G-4.3 arm B: train B3 on the rebuilt trainer, read back, compare with arm A
   bitwise; release B3.

Results go to ``<out_dir>/results.json`` (+ ``steps.jsonl`` as they happen).
A failed criterion raises :class:`HarnessFailed` (non-zero learner exit).
Nothing here is a GPU result until it has run on the pinned image and GPUs.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

PLAN_FILE = "yeto-rl-e2-harness.json"  # at the repo root (experiment snapshots only)
PLAN_ENV = "YETO_RL_E2_HARNESS"
PLAN_SCHEMA = "yeto.e2_harness_plan/v1"
CONFIGS = ("C1", "C2")
_REPO_ROOT = Path(__file__).resolve().parents[4]


class HarnessFailed(RuntimeError):
    pass


class EnvironmentBlocked(HarnessFailed):
    pass


def load_plan(environ: Mapping[str, str] | None = None, *, repo_root: Path | None = None) -> dict | None:
    env = os.environ if environ is None else environ
    path = env.get(PLAN_ENV) or str((repo_root or _REPO_ROOT) / PLAN_FILE)
    if not os.path.isfile(path):
        if env.get(PLAN_ENV):
            raise HarnessFailed(f"{PLAN_ENV}={path} does not exist")
        return None
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = plan_problems(plan)
    if problems:
        raise HarnessFailed("invalid E2 harness plan: " + "; ".join(problems))
    return plan


def plan_problems(plan: Mapping[str, Any]) -> list[str]:
    out = []
    if plan.get("schema") != PLAN_SCHEMA:
        out.append(f"schema must be {PLAN_SCHEMA}")
    if plan.get("config") not in CONFIGS:
        out.append(f"config must be one of {CONFIGS}")
    if not isinstance(plan.get("warmup_rounds"), int) or plan["warmup_rounds"] < 1:
        out.append("warmup_rounds must be a positive integer")
    if not plan.get("out_dir"):
        out.append("out_dir missing")
    if plan.get("expected_dp") not in (1, 2):
        out.append("expected_dp must be 1 or 2")
    return out


@dataclass
class HarnessContext:
    driver: Any  # IslandDriver
    actor: Any  # SwappableActor
    miles_args: Any
    rollout_executor: Any
    runner: Any  # LoopRunner
    algorithm: Any  # AlgorithmSpec
    base_model_revision: str
    backend_fingerprint: str
    plan: Mapping[str, Any]
    rebuild: Callable[..., Any] | None = None  # fork rebuild_training_models (tests inject)
    worker_manager: Any = None
    ledger_factory: Callable[[Path], Any] | None = None
    results: dict[str, Any] = field(default_factory=dict)


class _Recorder:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, Any] = {"criteria": {}, "steps": []}

    def step(self, step_name: str, **fields: Any) -> None:
        record = {"t": time.time(), "step": step_name, **fields}
        self.results["steps"].append(record)
        with open(self.out_dir / "steps.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, default=repr) + "\n")

    def criterion(self, name: str, ok: bool, **detail: Any) -> bool:
        self.results["criteria"][name] = {"pass": bool(ok), **detail}
        self.step("criterion", name=name, ok=bool(ok))
        return ok

    def write(self) -> dict[str, Any]:
        crit = self.results["criteria"]
        self.results["pass"] = bool(crit) and all(c["pass"] for c in crit.values())
        (self.out_dir / "results.json").write_text(
            json.dumps(self.results, indent=1, sort_keys=True, default=repr), encoding="utf-8")
        return self.results


def _per_rank(ctx: HarnessContext, fn_path: str) -> list[dict[str, Any]]:
    return [dict(r) for r in ctx.runner.run(ctx.actor.run_plugin(fn_path, {}))]


def _summaries(ctx: HarnessContext) -> dict[str, dict[str, Any]]:
    from .cut_plugin import STATE_SUMMARY, shard_name

    return {shard_name(s["coord"]): s for s in _per_rank(ctx, STATE_SUMMARY)}


def _digests(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, tuple]:
    return {k: (v["state_digest"], v["rng_digest"], v["scheduler_samples"]) for k, v in summaries.items()}


def run_harness(ctx: HarnessContext) -> dict[str, Any]:
    plan = ctx.plan
    rec = _Recorder(Path(plan["out_dir"]).expanduser() / plan["config"])
    try:
        _run(ctx, rec)
    except HarnessFailed:
        raise
    except Exception as error:  # noqa: BLE001 - recorded, then re-raised
        rec.criterion("harness_completed", False, error=f"{type(error).__name__}: {error}",
                      # RecoveryRequired carries the rebuild attempts (fork stage + cause)
                      attempts=list(getattr(error, "attempts", None) or []),
                      cause=repr(error.__cause__) if error.__cause__ is not None else None)
        rec.write()
        raise
    results = rec.write()
    if not results["pass"]:
        failed = sorted(k for k, v in results["criteria"].items() if not v["pass"])
        raise HarnessFailed(f"E2 harness {plan['config']}: failed {failed}")
    return results


def _run(ctx: HarnessContext, rec: _Recorder) -> None:
    from .cut_plugin import RANK_DETERMINISM
    from .rebuild_wiring import CutSource
    from .trainer_rebuild import rebuild_same_shape

    driver, plan = ctx.driver, ctx.plan
    trainer = driver.trainer

    # 0. determinism (plan-v3 §0): all ranks, else environment-blocked
    det = _per_rank(ctx, RANK_DETERMINISM)
    rec.step("determinism", ranks=det)
    if not det or not all(d["env_ok"] and d["deterministic_mode"] for d in det):
        rec.criterion("determinism_settings", False, ranks=det)
        rec.write()
        raise EnvironmentBlocked("determinism settings not in effect on every rank (plan-v3 §0)")
    rec.criterion("determinism_settings", True)
    if plan.get("unsafe_state_reads"):  # diagnostic sub-run only (plan-v6): no yeto read guard
        from .cut_plugin import SET_UNSAFE_STATE_READS

        flags = [dict(r) for r in ctx.runner.run(ctx.actor.run_plugin(SET_UNSAFE_STATE_READS, {"enabled": True}))]
        rec.step("diagnostic_unsafe_state_reads", ranks=flags)
    want = plan.get("lora_dropout")
    if want is not None and any(d.get("lora_dropout") != want for d in det):
        rec.criterion("configuration", False, lora_dropout=[d.get("lora_dropout") for d in det],
                      expected=want)
        rec.write()
        raise EnvironmentBlocked(f"LoRA dropout on the ranks is not the plan's {want}")
    layout = trainer.actual_layout()
    rec.step("layout", layout=layout)
    if layout["dp"] != plan["expected_dp"]:
        rec.criterion("configuration", False, layout=layout, expected_dp=plan["expected_dp"])
        rec.write()
        raise HarnessFailed(f"trainer DP {layout['dp']} != plan {plan['expected_dp']}")

    # 1. start + warm-up rounds on the normal driver path
    out = rec.out_dir
    if getattr(driver, "ledger", None) is None:
        from yeto.rl.engine.ledger import BatchLedger

        driver.ledger = (ctx.ledger_factory or BatchLedger)(out / "state")
    driver.handshake()
    start = driver.sync.start(driver)
    driver.ledger.rebase(start.rollout_id)
    driver.publish(start.state, rollout_id=start.rollout_id)
    rid = start.rollout_id
    for _ in range(int(plan["warmup_rounds"])):
        driver.safe_point(rid)
        driver.run_round(rid)
        rid += 1
    driver.safe_point(rid)
    rec.step("warmup_done", next_rollout_id=rid, local_step=driver.local_step)

    # 2. cut C0 at the safe point
    gbs = int(ctx.miles_args.global_batch_size)
    ref_load = getattr(ctx.miles_args, "ref_load", None)
    source = CutSource(
        driver=driver, trainer=trainer, rollout=driver.rollout, ledger=driver.ledger,
        algorithm=ctx.algorithm, backend_fingerprint=ctx.backend_fingerprint,
        cut_root=str(out / "cuts"), global_batch_size=gbs,
        ref_model=None if not ref_load else {"ref_load": str(ref_load),
                                             "base_model_revision": ctx.base_model_revision},
    )
    epoch = int(getattr(driver, "config_epoch", 0) or 0)
    cut_id = f"e2-{plan['config'].lower()}-c0"
    context = source.context(cut_id)
    trainer.save_cut(epoch=epoch, context=context)
    expect = source.expectation(layout, epoch=epoch)
    from yeto.rl.engine.cut import load_manifest

    manifest = load_manifest(source.cut_root, cut_id)
    saved = {s["path"]: s for s in manifest.rank_summaries}
    rec.step("cut_saved", cut_id=cut_id, files=[f.to_dict() for f in manifest.files])

    def restore(e=expect):
        return trainer.restore_cut(cut_id, epoch=epoch, root=source.cut_root, expect=e)

    # 3. G-4.2 (g): in-place restore on the live (trained) trainer is refused
    live_before = _digests(_summaries(ctx))
    try:
        restore()
        g_ok, g_err = False, "accepted"
    except Exception as error:  # noqa: BLE001 - Ray wraps the rank error
        g_ok, g_err = "freshly built" in str(error), f"{type(error).__name__}: {error}"
    g_unchanged = _digests(_summaries(ctx)) == live_before
    rec.criterion("G-4.2(g) live restore refused, state unchanged", g_ok and g_unchanged,
                  error=g_err, unchanged=g_unchanged)

    # 4. G-4.3 arm A on the live trainer. The two arms are symmetric per layout:
    # fixed-partition (production elastic layout) -> arm A re-publishes the
    # same policy and arm B goes through driver.rebuild_trainer (which
    # re-publishes); colocated -> no re-publication in either arm (a second
    # publish of loaded weights fails in SGLang; GPU C1 attempt 2) and arm B
    # calls rebuild_same_shape directly.
    colocated = bool(getattr(driver, "colocated", False))
    if not colocated:
        driver.publisher.publish(driver.published_state)
    pre_a = _summaries(ctx)
    # Compare the TRAINING state (a re-publication advances only weight_version);
    # the counter itself is checked explicitly: cut + number of re-publications.
    republished = 0 if colocated else 1
    rec.criterion("G-4.3 arm A pre-step state == cut",
                  {k: (v["train_state_digest"], v["rng_digest"]) for k, v in pre_a.items()}
                  == {k: (v.get("train_state_digest"), v["rng_digest"]) for k, v in saved.items()})
    rec.criterion("G-4.3 arm A weight_version == cut + re-publications",
                  all(_advanced(saved[k].get("weight_version"), v.get("weight_version"), republished)
                      for k, v in pre_a.items()),
                  cut={k: v.get("weight_version") for k, v in saved.items()},
                  now={k: v.get("weight_version") for k, v in pre_a.items()})
    batch = driver._generate(rid)  # the frozen batch B3 (same policy token)
    with trainer.retained_payloads():
        try:
            if colocated:  # _generate offloaded the trainer (driver order: onload, train)
                trainer.onload()
            trainer.train_step(batch)
            grad_a = trainer.last_grad_norm
            post_a = _summaries(ctx)

            # 5. rebuild (fresh trainer) -> G-4.2 (a)-(f) -> valid restore
            def restore_with_rejections() -> Any:
                _rejection_matrix(ctx, rec, source=source, cut_id=cut_id, epoch=epoch,
                                  expect=expect, layout=layout)
                return restore()

            from .trainer_rebuild import live_cursor

            before_cursor = live_cursor(driver.rollout, "before the rebuild")
            def rebuild() -> Any:
                return rebuild_same_shape(
                    trainer, args=ctx.miles_args, rollout_executor=ctx.rollout_executor,
                    actor=ctx.actor, run=ctx.runner.run, restore=restore_with_rejections,
                    rollout=driver.rollout, worker_manager=ctx.worker_manager, rebuild=ctx.rebuild,
                )

            if colocated:
                result = rebuild()  # no re-publication (see arm A)
            else:  # production path: restore, policy-hash check, re-publication to every member
                driver.at_safe_point = True
                result = driver.rebuild_trainer(rebuild, cut_policy_hash=manifest.progress.policy_hash)
            rec.step("rebuilt", outcome=result.outcome, generation=result.generation,
                     attempts=result.attempts)
            restored_hash = driver.policy_state.export().policy_tensor_hash()
            rec.criterion("G-4.3 restored policy hash == cut", restored_hash == manifest.progress.policy_hash,
                          restored=restored_hash, cut=manifest.progress.policy_hash)
            rec.criterion("G-4.3 data cursor unchanged across rebuild",
                          live_cursor(driver.rollout, "after the rebuild") == before_cursor)
            rec.criterion("G-4.3 layout unchanged across rebuild", trainer.actual_layout() == layout)
            pre_b = _summaries(ctx)
            rec.criterion("G-4.3(1) restored state/RNG == cut before step 3",
                          {k: (v["train_state_digest"], v["rng_digest"]) for k, v in pre_b.items()}
                          == {k: (v.get("train_state_digest"), v["rng_digest"]) for k, v in saved.items()})
            rec.criterion("G-4.3 arm B weight_version == cut + re-publications",
                          all(_advanced(saved[k].get("weight_version"), v.get("weight_version"), republished)
                              for k, v in pre_b.items()),
                          cut={k: v.get("weight_version") for k, v in saved.items()},
                          now={k: v.get("weight_version") for k, v in pre_b.items()})
            rec.criterion("G-4.3(1) scheduler not double-counted",
                          all(v["scheduler_samples"] == manifest.progress.scheduler_samples
                              for v in pre_b.values()))

            # 6. arm B on the rebuilt trainer
            if colocated:
                trainer.onload()
            trainer.train_step(batch)
            grad_b = trainer.last_grad_norm
            post_b = _summaries(ctx)
        finally:
            trainer.release_payload(batch)
    _compare_arms(rec, post_a, post_b, grad_a, grad_b, manifest.progress.scheduler_samples + gbs)


def _advanced(cut: int | None, now: int | None, publications: int) -> bool:
    """weight_version after ``publications`` re-publications of the cut (None = no counter)."""
    if cut is None or now is None:
        return cut is None and now is None
    return now == cut + publications


def _compare_arms(rec: _Recorder, post_a, post_b, grad_a, grad_b, samples: int) -> None:
    same_keys = set(post_a) == set(post_b)
    diffs = {}
    for k in sorted(set(post_a) & set(post_b)):
        a, b = post_a[k], post_b[k]
        bad = sorted(t for t in set(a["tensors"]) | set(b["tensors"])
                     if a["tensors"].get(t) != b["tensors"].get(t))
        if bad or a["state_digest"] != b["state_digest"] or a["rng_digest"] != b["rng_digest"]:
            diffs[k] = {"tensors": bad, "state": a["state_digest"] == b["state_digest"],
                        "rng": a["rng_digest"] == b["rng_digest"]}
    rec.criterion("G-4.3(2) adapter/master/moments/step/RNG bitwise equal", same_keys and not diffs,
                  diffs=diffs)
    rec.criterion("G-4.3(2) grad_norm bitwise equal", grad_a is not None and grad_a == grad_b,
                  arm_a=grad_a, arm_b=grad_b)
    rec.criterion("G-4.3(3) three optimizer steps, no extra scheduler step",
                  all(v["scheduler_samples"] == samples for v in (*post_a.values(), *post_b.values())),
                  expected=samples)
    rec.criterion("G-4.3(3) moments not reset",
                  all(v["moments_nonzero"] == v["optimizer_entries"] > 0 for v in post_b.values()))
    rec.criterion("L2 bf16 model copy == bf16(master)",
                  all(not v["bf16_master_mismatch"] for v in (*post_a.values(), *post_b.values())))


def _rejection_matrix(ctx: HarnessContext, rec: _Recorder, *, source: Any, cut_id: str, epoch: int,
                      expect: Any, layout: Mapping[str, int]) -> None:
    """G-4.2 (a)-(f) on the freshly built trainer; every case refused, nothing written."""
    from yeto.rl.engine.cut import cut_dir, load_manifest

    trainer = ctx.driver.trainer
    manifest = load_manifest(source.cut_root, cut_id)
    directory = cut_dir(source.cut_root, cut_id)
    own = next(f for f in manifest.files if f.coord.get("dp") == 0 and f.coord.get("tp") == 0)
    peer = next((f for f in manifest.files if f.coord.get("dp") == 1 and f.coord.get("tp") == 0), None)

    def flip_last(path: Path) -> Callable[[], None]:
        data = path.read_bytes()
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        return lambda: path.write_bytes(data)

    def truncate(path: Path) -> Callable[[], None]:
        data = path.read_bytes()
        path.write_bytes(data[:-1024])
        return lambda: path.write_bytes(data)

    ident = expect.algorithm
    cases: list[tuple[str, Callable[[], Callable[[], None] | None], Any]] = [
        ("a", lambda: flip_last(directory / own.path), expect),
        ("b", lambda: truncate(directory / own.path), expect),
        ("c", lambda: None, replace(expect, local_step=expect.local_step + 1)),
        ("d", lambda: None, replace(expect, algorithm=replace(ident, algorithm_spec_sha256="0" * 64))),
        ("e", lambda: None, replace(expect, algorithm=replace(
            ident, plugin_sha256=tuple(sorted((*ident.plugin_sha256, "yeto.rl.algos.e2_probe@" + "0" * 64)))))),
    ]
    if layout["dp"] > 1:
        if peer is None:
            rec.criterion("G-4.2(f) peer shard present", False)
        else:
            cases.append(("f", lambda: flip_last(directory / peer.path), expect))
    fresh = _digests(_summaries(ctx))
    for name, damage, exp in cases:
        undo = damage()
        try:
            trainer.restore_cut(cut_id, epoch=epoch, root=source.cut_root, expect=exp)
            refused, error = False, "accepted"
        except Exception as err:  # noqa: BLE001 - Ray wraps the rank error
            refused, error = True, f"{type(err).__name__}: {err}"
        finally:
            if undo is not None:
                undo()
        unchanged = _digests(_summaries(ctx)) == fresh
        rec.criterion(f"G-4.2({name}) refused, state unchanged", refused and unchanged,
                      error=error, unchanged=unchanged)
