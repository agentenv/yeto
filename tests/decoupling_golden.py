"""Golden samples for the yeto-framework-decoupling change (phase 0, task 1.1).

Every sample is produced on CPU only: no Ray, no GPU, no cloud.  The path is the
real one the island takes -- ``yeto launch`` CLI -> launcher island task (sky
stubbed) -> the island shell prelude run by bash under a temporary HOME -> the
learner's own ``parse_args`` -> ``resolve_rl_run_config`` -> ``build_ports_launch``
(the Miles argv the ports learner hands to Miles).  The progress/tape samples
drive the real ``IslandDriver`` with the fake engine and the fake syncers.

Recorded per configuration:
  * ``miles_argv``: the full Miles command line (temporary paths -> ``<TMP>``);
  * ``algorithm_sha256`` (launcher-side expected hash and learner-side recompute)
    and ``algorithm_semantic_sha256``;
  * ``execution_profile`` + ``contract_hash`` (see ``_execution_profile``);
  * ``plugins``: every plugin reference in the spec and every ``yeto.*`` dotted
    path in the argv, with its module file (repo-relative);
  * ``runtime_attrs`` / ``placement`` of the translated launch;
  * ``backend_identity`` without the engine commit.

No code hash is recorded (rl-identity-tiering D5, L9.10): no plugin source
hash, no ``ports_runtime_fingerprint``, no engine commit; since L9.11 the
translated ``runtime_attrs`` carry no plugin source hash either. The run code
pack pins code; a plugin-only change needs no re-record. Specs built here carry
plugin references without a source hash, so their hashes do not depend on code.

``session_contract_hash`` is NOT recorded: it is the syncer layout fingerprint of
the live LoRA tensors (``yeto.protocol.layout_fingerprint``), only known once the
model is loaded on the island.

Regenerate (only when a change is intended and documented in hash-migration.md):
    PYTHONPATH=/tmp/s15-noray python tests/decoupling_golden.py --write
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import re
import shlex
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO / "tests" / "golden" / "decoupling"
for _p in (str(REPO), str(REPO / "tests"), str(REPO / "tests" / "multinode_gpu")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------- helpers
def _module_file(dotted: str) -> Path | None:
    module = dotted.split(":", 1)[0]
    candidates = [module] if ":" in dotted else [module, module.rsplit(".", 1)[0]]
    for name in candidates:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        if spec is not None and spec.origin and spec.origin.endswith(".py"):
            return Path(spec.origin)
    return None


def _plugin_entry(path: str, recorded_sha: str | None) -> dict[str, Any]:
    file = _module_file(path)
    entry: dict[str, Any] = {"path": path}
    if recorded_sha is not None:
        entry["spec_sha256"] = recorded_sha
    if file is None:
        entry["module_file"] = None
    else:
        try:
            entry["module_file"] = str(file.resolve().relative_to(REPO))
        except ValueError:
            entry["module_file"] = "<outside-repo>/" + file.name
    return entry


def _spec_plugins(node: Any, out: dict[str, str | None]) -> None:
    if isinstance(node, dict):
        if "path" in node and set(node) <= {"path", "sha256"} and isinstance(node["path"], str):
            out[node["path"]] = node.get("sha256")  # a plugin ref (PluginRef.to_dict)
            return
        for value in node.values():
            _spec_plugins(value, out)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _spec_plugins(value, out)


_DOTTED = re.compile(r"^yeto(\.[A-Za-z_][A-Za-z0-9_]*)+(:[A-Za-z_][A-Za-z0-9_]*)?$")


def plugins_of(spec_dict: dict, argv: list[str]) -> list[dict[str, Any]]:
    found: dict[str, str | None] = {}
    _spec_plugins(spec_dict, found)
    for token in argv:
        if _DOTTED.match(token) and token not in found:
            found[token] = None
    return [_plugin_entry(p, found[p]) for p in sorted(found)]


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (frozenset, set)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return repr(value)


def _argv_value(argv: list[str], flag: str, default: str | None = None) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else default


def _execution_profile(args, launch, algorithm, *, yeto_policy_sync: bool) -> dict[str, Any]:
    """``entry.execution_profile_for`` on a Miles-args view built from the argv.

    The island builds it from Miles' parsed namespace (``parse_miles_args`` needs
    Miles, which this host must not import).  The five fields the profile reads
    are taken from the translated argv / runtime attrs / learner args instead:
    rollout_batch_size, n_samples_per_prompt, num_steps_per_rollout,
    yeto_rl_sync_preset, yeto_rl_overlap_eval.
    """
    from yeto.rl.adapters.miles.entry import execution_profile_for

    argv = list(launch.argv)
    view = types.SimpleNamespace(
        rollout_batch_size=int(_argv_value(argv, "--rollout-batch-size")),
        n_samples_per_prompt=int(_argv_value(argv, "--n-samples-per-prompt")),
        num_steps_per_rollout=int(_argv_value(argv, "--num-steps-per-rollout", "1")),
        yeto_rl_sync_preset=getattr(args, "sync_preset", "strict-avg"),
        yeto_rl_overlap_eval=bool(getattr(args, "rl_overlap_eval", False)),
        # S17 N17: the LR schedule enters the contract hash
        lr=(None if _argv_value(argv, "--lr", None) is None else float(_argv_value(argv, "--lr"))),
        lr_decay_style=_argv_value(argv, "--lr-decay-style", None),
        lr_decay_iters=(None if _argv_value(argv, "--lr-decay-iters", None) is None
                        else int(_argv_value(argv, "--lr-decay-iters"))),
        lr_warmup_iters=int(_argv_value(argv, "--lr-warmup-iters", "0")),
        min_lr=float(_argv_value(argv, "--min-lr", "0")),
    )
    for name, value in dict(launch.runtime_attrs).items():
        if name in ("yeto_rl_sync_preset", "yeto_rl_overlap_eval"):
            setattr(view, name, value)
    profile = execution_profile_for(
        view, launch, algorithm, yeto_policy_sync=yeto_policy_sync,
        expected_sha256=getattr(args, "rl_expected_algorithm_sha256", None),
    )
    return {"profile": _jsonable(profile.to_dict()), "contract_hash": profile.contract_hash}


# ---------------------------------------------------------------- providers
def small_provider():
    return types.SimpleNamespace(
        hidden_size=16, num_attention_heads=4, num_layers=2, ffn_hidden_size=32,
        num_query_groups=2, kv_channels=4, seq_length=32768, vocab_size=64,
        layernorm_epsilon=1e-6, position_embedding_type="rope", rotary_base=1000000,
        rotary_percent=1.0,
    )


# ---------------------------------------------------------------- configurations
@dataclasses.dataclass(frozen=True)
class Config:
    name: str
    title: str
    cli: Callable[[Path], tuple[str, ...]]
    fn: bool = False  # Flash-Next recipe (fp_fn provider / snapshot)
    patch: Callable[[Any, Path], None] | None = None  # extra monkeypatches before the launch


BASE = ("--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1",
        "--gpu", "aws:2xa100@us-east-1")


def _spec_file(tmp: Path, name: str, build: Callable[[], Any]) -> str:
    path = tmp / f"{name}.algorithm_spec.json"
    path.write_text(build().canonical_json())
    return str(path)


def _tis_spec():
    from yeto.rl.engine.algorithm import AlgorithmSpec

    return AlgorithmSpec(correction={"method": "tis", "tis_clip": 2.0, "tis_clip_low": 0.0})


DRGRPO_EVIDENCE = ("openspec/changes/rl-algo-grpo-knobs/evidence/2026-09-29-algo1b-g1/out/"
                   "drgrpo/algorithm_spec.json")


def _maxrl_spec():
    from yeto.rl.algos import grpo_knobs
    from yeto.rl.algos import reward_pipeline as rp
    from yeto.rl.engine.algorithm import AlgorithmSpec

    spec = AlgorithmSpec(advantage={"estimator": "grpo", "transform": "maxrl",
                                    "reward_binary": True,
                                    "reward_postprocess": rp.dispatcher_ref().to_dict(),
                                    "reward_shapers": []})
    return _without_plugin_hashes(grpo_knobs.with_pipeline_plugins(spec))


def _without_plugin_hashes(spec):
    """The spec with plugin references by path only (rl-identity-tiering L9.2), so
    the recorded hashes do not change with plugin source code (L9.10)."""
    from yeto.rl.engine.algorithm import AlgorithmSpec, _strip_plugin_hashes

    return AlgorithmSpec.from_dict(_strip_plugin_hashes(json.loads(spec.canonical_json())))


def _elastic(tmp: Path) -> tuple[str, ...]:
    res = tmp / "res.json"
    res.write_text(json.dumps({"configs": {"c0": {"trainer": 1, "rollout": 1}}, "edges": []}))
    return ("--rl-elastic", "--rl-elastic-resources", str(res), "--rl-elastic-initial-config",
            "c0", "--rl-elastic-cells", "a")


def _fn_cli(_tmp: Path) -> tuple[str, ...]:
    import test_rl_fn_train_args as fta

    return tuple(fta._argv("2x8"))


CONFIGS: tuple[Config, ...] = (
    Config("grpo_default", "GRPO 默认（ports，固定分区，strict-avg）", lambda t: BASE),
    Config("grpo_tis", "GRPO + TIS 失配修正（tis_clip 2.0）",
           lambda t: BASE + ("--rl-algorithm-spec", _spec_file(t, "tis", _tis_spec))),
    Config("decoupled", "decoupled 外层同步（--rl-sync-preset decoupled）",
           lambda t: BASE + ("--rl-sync-preset", "decoupled", "--local-rl-rounds-per-sync", "2")),
    Config("drgrpo", "Dr.GRPO 常数分母归约（G1 drgrpo 证据里的 spec 原文件）",
           lambda t: BASE + ("--rl-algorithm-spec", str(REPO / DRGRPO_EVIDENCE))),
    Config("seq_adv_maxrl", "seq_adv：MaxRL 优势变换（含奖励流水线插件）",
           lambda t: BASE + ("--rl-algorithm-spec", _spec_file(t, "maxrl", _maxrl_spec))),
    Config("elastic", "岛内弹性模式（--rl-elastic，单配置资源表）",
           lambda t: BASE + _elastic(t)),
    Config("fn_2x8", "Flash-Next 2×8 正式训练形状（fntrain.sh print 2x8）", _fn_cli, fn=True),
)


# ---------------------------------------------------------------- recording
def record_config(config: Config) -> dict[str, Any]:
    import _pytest.monkeypatch as mpmod

    from rl_e2e_launch import island_run, learner_from_run
    from yeto.rl.adapters.miles import island_entry as learner
    from yeto.rl.engine import run_config
    from yeto.rl.engine.run_config import resolve_rl_run_config

    tmp = Path(tempfile.mkdtemp(prefix="yeto-golden-"))
    mp = mpmod.MonkeyPatch()
    try:
        cli = config.cli(tmp)
        if config.patch:
            config.patch(mp, tmp)
        run = island_run(cli, mp)
        args, _env = learner_from_run(run, tmp / "home")
        yeto_policy_sync = not getattr(args, "rl_single_island_no_sync", False)
        if config.fn:
            import fp_fn
            from yeto.rl.profiles import qwen3_8_next as q

            mp.setattr(run_config, "_resolve_ref_load",
                       lambda a, model_path: a.megatron_ref_load or str(model_path))
            rc = resolve_rl_run_config(
                args, model_path=fp_fn.FN_SNAPSHOT, rollout_model_path=None,
                prompt_path="/root/yeto-rl/prompts.jsonl", eval_prompt_path=None,
                provider=fp_fn.fn_provider(48),
                target_modules=sorted({m.rsplit(".", 1)[-1] for m in q.LORA_TARGET_MODULES}),
                yeto_policy_sync=yeto_policy_sync)
        else:
            rc = resolve_rl_run_config(
                args, model_path="/model", rollout_model_path=None,
                prompt_path="/root/yeto-rl/prompts.jsonl", eval_prompt_path=None,
                provider=small_provider(), target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                yeto_policy_sync=yeto_policy_sync)
        launch = learner.build_ports_launch(args, rc, ())
        spec = launch.algorithm
        learner_spec = learner._ports_spec(args)
    finally:
        mp.undo()
    tmp_s = str(tmp)

    def norm(text: str) -> str:
        return text.replace(tmp_s, "<TMP>")

    argv = [norm(t) for t in launch.argv]
    cli_norm = [norm(t) for t in cli]
    return {
        "name": config.name,
        "title": config.title,
        "launch_cli": cli_norm,
        "miles_argv": argv,
        "miles_argv_sha256": hashlib.sha256(shlex.join(argv).encode()).hexdigest(),
        "algorithm_spec": json.loads(spec.canonical_json()),
        "algorithm_sha256": spec.sha256(),
        "algorithm_semantic_sha256": spec.semantic_sha256(),
        "algorithm_sha256_launcher_expected": getattr(args, "rl_expected_algorithm_sha256", None),
        "algorithm_sha256_learner_ports_spec": learner_spec.sha256(),
        "launch_algorithm_sha256": launch.algorithm_sha256,
        "execution_profile": _execution_profile(args, launch, spec,
                                                yeto_policy_sync=yeto_policy_sync),
        "placement": json.loads(norm(json.dumps(_jsonable(launch.placement)))),
        "runtime_attrs": json.loads(norm(json.dumps(_jsonable(dict(launch.runtime_attrs))))),
        "plugins": plugins_of(json.loads(spec.canonical_json()), argv),
        # Phase 5 (design D7); without the engine commit (L9.10, design D5).
        "backend_identity": _backend_identity(),
        "session_contract_hash": None,
        "session_contract_hash_note": "运行时由 LoRA 张量布局算出（yeto.protocol.layout_fingerprint），"
                                      "再与岛契约 v3（后端身份、学习率计划、语义哈希、代码包摘要）绑定"
                                      "（yeto.rl.engine.backend_identity），离线不可得",
    }


def _backend_identity() -> dict:
    from yeto.rl.adapters.miles.identity import backend_identity

    identity = backend_identity(compat_group="nvidia-h100")  # 7.7b: fixed sample card type
    return {k: v for k, v in identity.to_dict().items() if k != "engine_commit"}


# ---------------------------------------------------------------- fake-engine tapes / progress
# Fields that carry wall-clock time, durations or host-specific values.
_VOLATILE_KEYS = re.compile(r"(^|[/_])(ts|time|timestamp|wall|elapsed|seconds|duration|pid|host|"
                            r"started_at|finished_at|monotonic)($|[/_])|_s$|_ms$|_seconds$")


def _scrub(node: Any, tmp: str) -> Any:
    if isinstance(node, dict):
        return {k: ("<VOLATILE>" if _VOLATILE_KEYS.search(k) else _scrub(v, tmp))
                for k, v in node.items()}
    if isinstance(node, list):
        return [_scrub(v, tmp) for v in node]
    if isinstance(node, str):
        return node.replace(tmp, "<TMP>")
    if isinstance(node, float) and node != node:
        return "NaN"
    return node


def _tape(path: Path, tmp: str) -> list[dict]:
    return [_scrub(json.loads(line), tmp) for line in path.read_text().splitlines() if line.strip()]


def _torch_payload(path: Path, tmp: str) -> Any:
    """Decoded payload (the file bytes embed wall-clock fields such as
    ``local_round_stats.*_seconds``, so only the scrubbed payload is compared)."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    return _scrub(_jsonable(_tensors_to_lists(payload)), tmp)


def _tensors_to_lists(node: Any) -> Any:
    import torch

    if isinstance(node, torch.Tensor):
        return {"tensor": node.tolist(), "dtype": str(node.dtype)}
    if isinstance(node, dict):
        return {k: _tensors_to_lists(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_tensors_to_lists(v) for v in node]
    return node


def record_tapes() -> dict[str, Any]:
    import torch

    import test_rl_engine_driver as td
    from yeto.rl.core import build_rl_fragment_layout
    from yeto.rl.engine.bridges import (DecoupledIslandProgress, DecoupledSync, LocalOnlySync,
                                        StrictIslandProgress)
    from yeto.rl.engine.fake import FakeDecoupledSyncer

    out: dict[str, Any] = {}

    # 1) colocated-serial, local-only, three rounds with eval (driver tape only)
    tmp = Path(tempfile.mkdtemp(prefix="yeto-golden-tape-"))
    engine = td._engine()
    driver = td._driver(engine, LocalOnlySync(3), tmp,
                        evaluate=lambda r: {"pass_rate": 0.5}, eval_interval=2)
    driver.run()
    out["local_only_colocated"] = {"tape": _tape(tmp / "events.jsonl", str(tmp)),
                                   "engine_calls": _jsonable(td._engine_calls(engine))}

    # 2) strict-avg single island: schema-3 progress file + driver tape + bridge tape
    tmp = Path(tempfile.mkdtemp(prefix="yeto-golden-strict-"))
    engine = td._engine(torch.tensor([1.0, 3.0]))
    args = td._strict_args(tmp, engine)
    syncer = td._strict_syncer(engine, learners=1, rounds=1)
    driver = td._strict_driver(tmp, engine, syncer, learner_id=0, rounds=1,
                               progress=StrictIslandProgress(args))
    driver.run()
    out["strict_progress"] = {"progress_schema3": _torch_payload(tmp / "island.pt", str(tmp)),
                              "tape": _tape(tmp / "island-0.jsonl", str(tmp))}

    # 3) decoupled run-until-stop: schema-4 progress file + driver tape
    tmp = Path(tempfile.mkdtemp(prefix="yeto-golden-decoupled-"))
    args = td._dargs(tmp)
    engine = td._dengine()
    initial = engine.canonical(0)
    syncer = FakeDecoupledSyncer(build_rl_fragment_layout(initial.specs, 2), initial.tensors,
                                 learners=1, total_steps=4, pipeline=2)
    sync = DecoupledSync(args, client_factory=lambda _bridge: syncer.client(0))
    driver = td._driver(engine, sync, tmp, progress=DecoupledIslandProgress(args),
                        max_rollouts=20)
    driver.run()
    out["decoupled_progress"] = {"progress_schema4": _torch_payload(tmp / "island.pt", str(tmp)),
                                 "tape": _tape(tmp / "events.jsonl", str(tmp))}
    return out


# ---------------------------------------------------------------- files
def dumps(payload: Any) -> str:
    return json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


REPO_TOKEN = "<REPO>"


def render_all() -> dict[str, str]:
    """Samples as text; the checkout path is replaced by ``<REPO>`` so any worktree compares equal."""
    files = {f"{c.name}.json": dumps(record_config(c)) for c in CONFIGS}
    files["fake_engine_tapes.json"] = dumps(record_tapes())
    return {name: text.replace(str(REPO), REPO_TOKEN) for name, text in files.items()}


def main(argv: list[str]) -> int:
    files = render_all()
    if "--write" in argv:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (GOLDEN_DIR / name).write_text(text)
    for name, text in files.items():
        print(name, hashlib.sha256(text.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
