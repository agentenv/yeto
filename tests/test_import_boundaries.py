"""Static import-boundary check (yeto-framework-decoupling task 1.2, design D4).

Pure ``ast``: the checked modules are never imported, nothing is started.
Rules follow spec ``rl-framework-neutral-core`` (requirement 静态边界检查); the
convention is written down in ``yeto/rl/engine/README.md``.

  * neutral core -- ``yeto/rl/engine/`` (adapter excluded), ``yeto/rl/rewards/``,
    ``yeto/rl/harness/``, ``yeto/rl/algos/`` and today's loose neutral modules
    (reward / filter / algorithm-extension / harness files that still live
    directly under ``yeto/rl/``) -- MUST NOT import a training or inference
    framework (miles, miles_plugins, megatron, sglang, verl, vllm, torch_npu),
    a backend adapter, the legacy Miles engine modules, or a cloud library
    (sky, modal, ``yeto.modal_runner``, ``yeto.shape.providers``);
  * backend adapters MUST NOT import each other;
  * the cloud / launch layer MUST NOT import a backend adapter.

Existing violations are listed in ``tests/import_boundary_allowlist.txt`` as
``file<TAB>forbidden module<TAB># note``.  The list only shrinks: a new
violation fails, an entry whose import is gone fails ("please delete it"), and
the entry count must equal ``ALLOWLIST_CAP`` (lower the cap when you delete).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ALLOWLIST = REPO / "tests" / "import_boundary_allowlist.txt"
README = "yeto/rl/engine/README.md"
# Upper bound of allowlist entries; may only go down (D4).
ALLOWLIST_CAP = 5  # S26 E0.5: Codex harness removed; rl-code-quality-cleanup 1.10: legacy engine and launcher->overlay gone

FRAMEWORKS = ("miles", "miles_plugins", "megatron", "sglang", "verl", "vllm", "torch_npu")
ADAPTERS = ("yeto.rl.adapters",)
LEGACY = ("yeto.rl.miles", "yeto.rl.miles_overlay", "yeto.rl.overlays", "yeto.rl.learner",
          "yeto.rl.miles_full_parameter", "yeto.rl.miles_full_parameter_dense",
          "yeto.rl.miles_chunked_full_parameter", "yeto.rl.miles_sao_streaming",
          # phase 4 forwarders (old paths of modules moved under yeto/rl/adapters/miles/)
          "yeto.rl.flash_next_provider", "yeto.rl.deepseek_v4_bridge",
          "yeto.rl.deepseek_v4_clone_lora", "yeto.rl.deepseek_v4_expert_clone",
          "yeto.rl.deepseek_v4_expert_full", "yeto.rl.deepseek_v4_expert_full_runtime",
          "yeto.rl.sglang_deepseek_v4_clone", "yeto.rl.dense_sweep_wire",
          "yeto.rl.miles_full_parameter_probe", "yeto.rl.miles_full_parameter_manifest_probe",
          "yeto.rl.miles_full_parameter_continuation_probe")
CLOUD = ("sky", "modal", "yeto.modal_runner", "yeto.shape.providers")

CORE_FORBIDDEN = FRAMEWORKS + ADAPTERS + LEGACY + CLOUD

# Neutral-core scopes: directories (recursive) and single files, repo-relative.
CORE_DIRS = ("yeto/rl/engine", "yeto/rl/rewards", "yeto/rl/harness", "yeto/rl/algos")
CORE_EXCLUDED_DIRS: tuple[str, ...] = ()
CORE_FILES = (
    # rewards / filters (audit R1-R3)
    "yeto/rl/math_reward.py", "yeto/rl/gsm8k_reward.py", "yeto/rl/length_reward.py",
    "yeto/rl/numeric_answer_reward.py",
    "yeto/rl/filters.py",
    # algorithm extensions living beside the package (audit A1, A3, A5)
    "yeto/rl/critic_warmup.py", "yeto/rl/teacher_forcing.py", "yeto/rl/grad_audit.py",
    # harness test workload (audit R14)
    "yeto/rl/tool_wait_workload.py",
    # S27 G-FG: Megatron masters and the HF-named boundary (they act on the objects
    # the backend hands them; no framework import)
    "yeto/rl/megatron_masters.py", "yeto/rl/megatron_hf_boundary.py",
    # S27 G-FL: the Ascend recipe's plugin table (imports the user's plugin files by
    # path at install time; no framework import of its own) and Megatron's gradient
    # check (reads attributes off the DDP config the backend hands it)
    "yeto/rl/ascend_plugin_stack.py", "yeto/rl/megatron_grad_checks.py",
)
# Backend adapters: name -> directory.  Each must not import the others.
ADAPTER_DIRS = {"miles": ("yeto/rl/adapters/miles",),
                "verl": ("yeto/rl/adapters/verl",),
                "sampler_engines": ("yeto/rl/adapters/sampler_engines",)}
# Cloud / launch layer (audit A8, A9, C1-C3).
CLOUD_FILES = ("yeto/launcher.py", "yeto/modal_runner.py", "yeto/cli.py",
               "yeto/rl/stage_w_entry.py")
CLOUD_DIRS = ("yeto/shape", "yeto/cloud", "yeto/launch")


@dataclass(frozen=True)
class Violation:
    file: str
    module: str  # the forbidden module family as listed in the rule
    line: int
    imported: str
    rule: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.file, self.module)


def _matches(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _module_name(rel: str) -> str:
    parts = rel[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def imported_names(path: Path, rel: str) -> list[tuple[int, str]]:
    """Every module named by an import statement (top-level or lazy) or by a
    literal ``importlib.import_module`` / ``__import__`` call."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
    package = _module_name(rel)
    if not rel.endswith("__init__.py"):
        package = package.rsplit(".", 1)[0]
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                stem = ".".join(base + ([node.module] if node.module else []))
            else:
                stem = node.module or ""
            out.append((node.lineno, stem))
            out += [(node.lineno, f"{stem}.{a.name}") for a in node.names if a.name != "*"]
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in ("import_module", "__import__"):
                out.append((node.lineno, node.args[0].value))
    return out


def _files(root: Path, dirs, files=(), excluded=()) -> list[str]:
    found: set[str] = set()
    for d in dirs:
        base = root / d
        if base.is_dir():
            for p in base.rglob("*.py"):
                rel = p.relative_to(root).as_posix()
                if not any(rel.startswith(x + "/") for x in excluded):
                    found.add(rel)
    for f in files:
        if (root / f).is_file():
            found.add(f)
    return sorted(found)


def scan(root: Path = REPO) -> list[Violation]:
    found: dict[tuple, Violation] = {}

    def check(rel: str, forbidden, rule: str) -> None:
        for line, name in imported_names(root / rel, rel):
            for prefix in forbidden:
                if _matches(name, prefix):
                    v = Violation(rel, prefix, line, name, rule)
                    found.setdefault((rel, prefix, line), v)

    for rel in _files(root, CORE_DIRS, CORE_FILES, CORE_EXCLUDED_DIRS):
        check(rel, CORE_FORBIDDEN, "中立核心")
    for name, dirs in ADAPTER_DIRS.items():
        others = tuple(p for other, ds in ADAPTER_DIRS.items() if other != name for p in ds)
        others = tuple(p.replace("/", ".") for p in others)
        for rel in _files(root, dirs):
            check(rel, others, f"适配层 {name} 互相 import")
    for rel in _files(root, CLOUD_DIRS, CLOUD_FILES):
        check(rel, ADAPTERS, "云层/启动层 import 适配层")
    return sorted(found.values(), key=lambda v: (v.file, v.module, v.line))


def load_allowlist(path: Path = ALLOWLIST) -> dict[tuple[str, str], str]:
    entries: dict[tuple[str, str], str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        cols = raw.split("\t")
        key = (cols[0], cols[1])
        assert key not in entries, f"duplicate allowlist entry {key}"
        entries[key] = cols[2] if len(cols) > 2 else ""
    return entries


def problems(violations: list[Violation], allowlist: dict, cap: int) -> list[str]:
    out = []
    keys = {v.key for v in violations}
    for v in violations:
        if v.key not in allowlist:
            out.append(f"新增违规 [{v.rule}] {v.file}:{v.line} import {v.imported} "
                       f"(被禁模块 {v.module})；规则见 {README}")
    for key in sorted(set(allowlist) - keys):
        out.append(f"白名单条目已不存在于代码：{key[0]}\t{key[1]} —— 请删白名单条目"
                   f"（{ALLOWLIST.relative_to(REPO)}），并把 ALLOWLIST_CAP 减一；见 {README}")
    if len(allowlist) > cap:
        out.append(f"白名单 {len(allowlist)} 条超过上限 {cap}：白名单只减不增（{README}）")
    elif len(allowlist) < cap:
        out.append(f"白名单已减到 {len(allowlist)} 条：请把 ALLOWLIST_CAP 从 {cap} 改为 "
                   f"{len(allowlist)}（只减不增）")
    return out


# Removed with the legacy Miles engine (rl-code-quality-cleanup 1.10); no file
# anywhere, tests included, may import them again.  No allowlist.
REMOVED = ("yeto.rl.adapters.miles.legacy", "yeto.rl.adapters.miles.models.rollout_legacy_port")
REMOVED_SCAN_DIRS = ("yeto", "scripts", "tools", "tests")


def removed_module_imports(root: Path = REPO) -> list[str]:
    hits: dict[tuple[str, int], str] = {}
    for rel in _files(root, REMOVED_SCAN_DIRS):
        for line, name in imported_names(root / rel, rel):
            if any(_matches(name, prefix) for prefix in REMOVED):
                hits.setdefault((rel, line), f"{rel}:{line} import {name}")
    return list(hits.values())


# The import scan above misses a removed module named in text, e.g. a Miles
# hook path ("module:attr") or a dotted path handed to load_function.  Any
# mention of the removed packages in code, tests or docs is reported.
REMOVED_TEXT = ("adapters.miles.legacy", "rollout_legacy_port")
REMOVED_TEXT_SCOPES = ("yeto", "scripts", "tools", "tests", "docs")
REMOVED_TEXT_SUFFIXES = (".py", ".md", ".sh", ".json", ".toml", ".yaml", ".yml", ".txt")
REMOVED_TEXT_EXEMPT = ("tests/test_import_boundaries.py",)  # names them to forbid them


def removed_module_mentions(root: Path = REPO) -> list[str]:
    hits = []
    for scope in REMOVED_TEXT_SCOPES:
        base = root / scope
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            rel = path.relative_to(root).as_posix()
            if (not path.is_file() or path.suffix not in REMOVED_TEXT_SUFFIXES
                    or rel in REMOVED_TEXT_EXEMPT or "__pycache__" in path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), start=1):
                for needle in REMOVED_TEXT:
                    if needle in line:
                        hits.append(f"{rel}:{number} mentions {needle}")
    return hits


# ------------------------------------------------------------------ tests
def test_removed_legacy_engine_is_never_imported():
    assert not removed_module_imports(), "\n".join(removed_module_imports())


def test_removed_legacy_engine_is_never_named():
    found = removed_module_mentions()
    assert not found, "\n".join(found)


def test_removed_module_name_in_text_is_caught(tmp_path):
    root = _tree(tmp_path, {
        "yeto/hooks.py": 'HOOK = "yeto.rl.adapters.miles.legacy.engine:sync"\n',
        "docs/notes.md": "see rollout_legacy_port.generate_rollout\n",
    })
    assert removed_module_mentions(root) == [
        "yeto/hooks.py:1 mentions adapters.miles.legacy",
        "docs/notes.md:1 mentions rollout_legacy_port",
    ]


def test_removed_module_import_is_caught(tmp_path):
    root = _tree(tmp_path, {"scripts/x.py": "from yeto.rl.adapters.miles.legacy import engine\n"})
    assert removed_module_imports(root) == ["scripts/x.py:1 import yeto.rl.adapters.miles.legacy"]


def test_repository_respects_the_import_boundaries():
    found = problems(scan(), load_allowlist(), ALLOWLIST_CAP)
    assert not found, "\n".join(found)


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp_path


def test_injected_framework_import_fails(tmp_path):
    root = _tree(tmp_path, {"yeto/rl/engine/driver.py": "def f():\n    import miles.ray\n"})
    found = problems(scan(root), {}, 0)
    assert len(found) == 1 and "yeto/rl/engine/driver.py:2" in found[0]
    assert "miles.ray" in found[0] and README in found[0]


def test_relative_import_of_the_adapter_and_cloud_library_fail(tmp_path):
    root = _tree(tmp_path, {
        "yeto/rl/engine/cut.py": "from ..adapters.miles.reshard import ReshardPlan\nimport sky\n",
        "yeto/rl/adapters/miles/x.py": "import miles\nimport megatron.core\n",  # adapter: ok
        "yeto/rl/algos/a.py": "import importlib\nimportlib.import_module('sglang.srt')\n",
    })
    got = {(v.file, v.module) for v in scan(root)}
    assert got == {("yeto/rl/engine/cut.py", "yeto.rl.adapters"),
                   ("yeto/rl/engine/cut.py", "sky"), ("yeto/rl/algos/a.py", "sglang")}


def test_adapters_and_cloud_layer_rules(tmp_path):
    root = _tree(tmp_path, {
        "yeto/rl/adapters/verl/x.py": "from yeto.rl.adapters.miles import pins\n",
        "yeto/rl/adapters/miles/y.py": "import yeto.rl.adapters.miles.pins\n",  # own adapter: ok
        "yeto/launcher.py": "from yeto.rl.adapters.miles import entry\nimport sky\n",
    })
    got = {(v.file, v.module) for v in scan(root)}
    assert got == {("yeto/rl/adapters/verl/x.py", "yeto.rl.adapters.miles"),
                   ("yeto/launcher.py", "yeto.rl.adapters")}


def test_fixed_violation_still_listed_asks_to_delete_the_entry(tmp_path):
    root = _tree(tmp_path, {"yeto/rl/engine/driver.py": "import json\n"})
    stale = {("yeto/rl/engine/driver.py", "yeto.rl.miles"): "# E3"}
    found = problems(scan(root), stale, 1)
    assert len(found) == 1 and "请删白名单条目" in found[0]


def test_allowlist_cap_only_goes_down(tmp_path):
    root = _tree(tmp_path, {"yeto/rl/engine/driver.py": "import miles\n"})
    entry = {("yeto/rl/engine/driver.py", "miles"): ""}
    assert problems(scan(root), entry, 1) == []
    assert any("超过上限" in p for p in problems(scan(root), entry, 0))
    assert any("ALLOWLIST_CAP" in p for p in problems(scan(root), entry, 2))


def test_allowlist_entries_carry_a_note():
    for key, note in load_allowlist().items():
        assert note.startswith("#"), f"allowlist entry {key} needs an audit note"


# ------------------------------------------------- one import path per module
# rl-code-quality-cleanup 0.5/0.6 (spec "模块只有一个正式引用路径"): the 46
# forwarding stubs are gone; no file may import an old path.
OLD_PATHS = ("yeto.rl.engine.miles_adapter",) + tuple(m for m in LEGACY if m != "yeto.rl.overlays")
OLD_PATH_SCOPES = ("yeto", "tests", "scripts", "tools")


def scan_old_paths(root: Path = REPO) -> list[str]:
    out = []
    for rel in _files(root, OLD_PATH_SCOPES):
        try:
            names = imported_names(root / rel, rel)
        except SyntaxError:
            continue
        for line, name in names:
            for old in OLD_PATHS:
                if _matches(name, old):
                    out.append(f"{rel}:{line} imports old path {name} (forwarding stub {old} was deleted)")
    return sorted(set(out))


def test_no_file_imports_an_old_forwarding_path():
    found = scan_old_paths()
    assert not found, "\n".join(found)


def test_forwarding_stubs_are_gone():
    assert not (REPO / "yeto/rl/engine/miles_adapter").exists()
    for old in OLD_PATHS:
        if old == "yeto.rl.learner":
            continue  # command-line entry only (python -m), not importable
        assert not (REPO / (old.replace(".", "/") + ".py")).exists(), old


def test_old_path_import_is_reported(tmp_path):
    root = _tree(tmp_path, {"tests/t.py": "from yeto.rl import learner\n",
                            "tools/u.py": "import yeto.rl.engine.miles_adapter.entry\n"})
    found = scan_old_paths(root)
    assert len(found) == 2 and "tests/t.py:1" in found[0] and "tools/u.py:1" in found[1]


def test_learner_entry_is_not_importable():
    import importlib

    import pytest

    with pytest.raises(ImportError, match="command-line entry only"):
        importlib.import_module(".".join(("yeto", "rl", "learner")))
