"""S26 E0.4: the Harbor layer of the Miles / verl images and the start check."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from yeto.launch import image_check
from yeto.rl.harness.harbor import image_layer
from yeto.rl.harness.harbor.pins import HARBOR_COMMIT, HARBOR_VERSION

ROOT = Path(__file__).resolve().parents[1]
LAYER = ROOT / "docker" / "harbor-layer"
ASSERT_SCRIPT = ROOT / "docker" / "image-assert" / "assert_image_versions.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


layer = _load(LAYER / "harbor_layer.py", "harbor_layer")
SPEC = json.loads((LAYER / "versions.json").read_text())
HARBOR_LINE = f"harbor @ git+https://github.com/harbor-framework/harbor@{HARBOR_COMMIT}"

# Base versions the S25 N10.9 check read from the two images (install_diff in
# SESSION25/logs/T-N109/check-miles2.log and check-verl.log).
MILES_BEFORE = """\
filelock==4.0.0
importlib-metadata==9.0.1
openai==2.6.1
PyYAML==6.0.1
websockets==17.1
aiohttp==3.14.5
torch==2.11.0+cu130
-e file:///sgl-workspace/sglang/python
"""
VERL_BEFORE = """\
aiohttp==3.14.1
yarl==1.24.2
importlib_metadata==9.0.0
openai==2.41.1
filelock==3.29.4
pyyaml==6.0.3
websockets==16.0
torch==2.13.0+cu130
vllm==0.29.0
"""


def _after(before: str, backend: str, **moves: str) -> dict[str, str]:
    """The freeze an install exactly as versions.json allows would leave (plus ``moves``)."""
    after = layer.parse_freeze(before)
    for name, change in SPEC["backends"][backend]["changes"].items():
        after[name] = change["layer"]
    after |= {"litellm": SPEC["pins"]["litellm"]["version"], "boat-sdk": SPEC["boat_sdk"],
              "harbor": "@ git+https://github.com/harbor-framework/harbor@" + HARBOR_COMMIT,
              "supabase": "2.33.0"}
    after |= {layer.canonical(k): v for k, v in moves.items()}
    return after


# --- versions.json -----------------------------------------------------------

def test_versions_json_pins_the_same_harbor_as_the_yeto_pins():
    assert SPEC["harbor"]["commit"] == HARBOR_COMMIT and SPEC["harbor"]["version"] == HARBOR_VERSION
    assert SPEC["harbor"]["requirement"] == HARBOR_LINE


def test_versions_json_records_the_moves_n109_saw_and_holds_filelock():
    miles, verl = (SPEC["backends"][b]["changes"] for b in ("miles", "verl"))
    assert {n: (c["base"], c["layer"]) for n, c in miles.items()} == {
        "openai": ("2.6.1", "2.54.0"), "pyyaml": ("6.0.1", "6.0.3"),
        "websockets": ("17.1", "15.0.1"), "importlib-metadata": ("9.0.1", "8.9.0")}
    assert {n: (c["base"], c["layer"]) for n, c in verl.items()} == {
        "websockets": ("16.0", "15.0.1"), "importlib-metadata": ("9.0.0", "8.9.0")}
    assert "filelock" not in miles and "aiohttp" not in verl  # held at the base version
    assert all(c["why"] for c in [*miles.values(), *verl.values(), *SPEC["pins"].values()])


# --- freeze parsing and constraints -----------------------------------------

def test_parse_freeze_canonicalizes_names_and_skips_editables():
    parsed = layer.parse_freeze(MILES_BEFORE + HARBOR_LINE + "\n# comment\n")
    assert parsed["pyyaml"] == "6.0.1" and parsed["importlib-metadata"] == "9.0.1"
    assert parsed["harbor"] == "@ git+https://github.com/harbor-framework/harbor@" + HARBOR_COMMIT
    assert not any("sglang" in name for name in parsed)


def test_constraints_hold_every_base_package_except_the_allowed_moves():
    spec, backend = layer.load_spec("miles")
    lines = layer.constraints(layer.parse_freeze(MILES_BEFORE), spec, backend)
    assert "filelock==4.0.0" in lines and "torch==2.11.0+cu130" in lines and "aiohttp==3.14.5" in lines
    assert "openai==2.54.0" in lines and "openai==2.6.1" not in lines
    assert "pyyaml==6.0.3" in lines and "litellm==1.95.0" in lines
    assert len([line for line in lines if line.startswith("openai==")]) == 1


def test_unknown_backend_is_refused():
    with pytest.raises(SystemExit, match="unknown backend"):
        layer.load_spec("trl")
    with pytest.raises(ValueError, match="unknown backend"):
        image_layer.layer_commands("trl", "python")


# --- build check -------------------------------------------------------------

@pytest.mark.parametrize("backend, before", [("miles", MILES_BEFORE), ("verl", VERL_BEFORE)])
def test_check_accepts_an_install_that_moved_only_what_versions_json_allows(backend, before):
    spec, backend_spec = layer.load_spec(backend)
    assert layer.check(layer.parse_freeze(before), _after(before, backend), spec, backend_spec) == []


def test_check_refuses_the_unlisted_filelock_move_n109_made_on_miles():
    spec, backend = layer.load_spec("miles")
    problems = layer.check(layer.parse_freeze(MILES_BEFORE), _after(MILES_BEFORE, "miles", filelock="3.32.7"),
                           spec, backend)
    assert problems == ["filelock: moved 4.0.0 -> 3.32.7 by the Harbor install, not allowed by versions.json"]


def test_check_refuses_a_removed_package_a_wrong_layer_version_and_a_moved_base():
    spec, backend = layer.load_spec("verl")
    before = layer.parse_freeze(VERL_BEFORE.replace("websockets==16.0", "websockets==16.1"))
    after = _after(VERL_BEFORE, "verl", **{"importlib-metadata": "8.5.0"})
    del after["vllm"]
    problems = layer.check(before, after, spec, backend)
    assert any(p.startswith("vllm: removed") for p in problems)
    assert any(p.startswith("importlib-metadata: installed 8.5.0, versions.json allows only 8.9.0") for p in problems)
    assert any(p.startswith("websockets: base image has 16.1") for p in problems)


def test_check_refuses_a_wrong_harbor_commit_or_litellm_pin():
    spec, backend = layer.load_spec("verl")
    after = _after(VERL_BEFORE, "verl", litellm="1.106.0", harbor="@ git+https://x@deadbeef")
    problems = layer.check(layer.parse_freeze(VERL_BEFORE), after, spec, backend)
    assert {p.split(":")[0] for p in problems} == {"litellm", "harbor"}


def test_check_cli_exits_1_with_one_line_per_problem(tmp_path, capsys):
    before, after = tmp_path / "before.txt", tmp_path / "after.txt"
    before.write_text(MILES_BEFORE)
    moved = _after(MILES_BEFORE, "miles", filelock="3.32.7")
    after.write_text("\n".join(f"{n}=={v}" if not v.startswith("@") else f"{n} {v}" for n, v in moved.items()))
    assert layer.main(["check", "--backend", "miles", str(before), str(after)]) == 1
    err = capsys.readouterr().err
    assert "[harbor-layer] FAIL filelock: moved 4.0.0 -> 3.32.7" in err and "refusing the image" in err


# --- start lock and start check ----------------------------------------------

def test_start_lock_asserts_engine_packages_and_the_layer_versions():
    lock = layer.start_lock(_after(VERL_BEFORE, "verl"), SPEC, "verl")
    packages = lock["packages"]
    assert packages["torch"] == {"version": "2.13.0+cu130"} and packages["vllm"] == {"version": "0.29.0"}
    assert packages["websockets"] == {"version": "15.0.1", "import": "websockets"}
    assert packages["filelock"] == {"version": "3.29.4", "import": "filelock"}
    assert packages["harbor"] == {"version": HARBOR_VERSION, "import": "harbor"}
    assert packages["litellm"] == {"version": "1.95.0"}
    assert lock["image"] == "verl-build+harbor" and lock["decisions"]


def _matching_lock(tmp_path: Path, **override: dict) -> Path:
    checker = _load(ASSERT_SCRIPT, "assert_image_versions")
    packages = {"pytest": {"version": pytest.__version__, "import": "pytest"}} | override
    lock = {"image": "test", "cpu_arch": checker.platform.machine(), "python": None, "packages": packages}
    path = tmp_path / "versions.lock.json"
    path.write_text(json.dumps(lock))
    return path


def test_check_image_passes_on_a_matching_environment(tmp_path):
    rows = image_check.check_image(_matching_lock(tmp_path), ASSERT_SCRIPT)
    assert {row["status"] for row in rows} == {"ok"}


def test_check_image_names_every_failed_check(tmp_path):
    lock = _matching_lock(tmp_path, openai={"version": "0.0.1"}, **{"no-such-dist": {"version": "1"}})
    with pytest.raises(image_check.ImageVersionMismatch) as err:
        image_check.check_image(lock, ASSERT_SCRIPT)
    text = str(err.value)
    assert "2 failed check(s)" in text and "package no-such-dist: not installed" in text
    assert "package openai" in text and "docker/harbor-layer/versions.json" in text


def test_check_image_at_start_does_nothing_outside_a_yeto_image(tmp_path, monkeypatch):
    monkeypatch.setenv(image_check.LOCK_ENV, str(tmp_path / "absent.json"))
    image_check.check_image_at_start.cache_clear()
    try:
        assert image_check.check_image_at_start() == ()
    finally:
        image_check.check_image_at_start.cache_clear()


def test_check_image_at_start_raises_when_the_image_has_a_lock(tmp_path, monkeypatch):
    lock = _matching_lock(tmp_path, **{"no-such-dist": {"version": "1"}})
    monkeypatch.setenv(image_check.LOCK_ENV, str(lock))
    monkeypatch.setattr(image_check, "IMAGE_DIR", ASSERT_SCRIPT.parent)
    image_check.check_image_at_start.cache_clear()
    try:
        with pytest.raises(image_check.ImageVersionMismatch, match="no-such-dist"):
            image_check.check_image_at_start()
    finally:
        image_check.check_image_at_start.cache_clear()


# --- image build commands ----------------------------------------------------

def test_layer_commands_constrain_check_lock_and_assert_in_that_order():
    commands = image_layer.layer_commands("miles", "/usr/bin/python")
    joined = "\n".join(commands)
    assert f"-c {image_layer.IMAGE_LAYER_DIR}/constraints.txt '{HARBOR_LINE}' 'boat-sdk==1.10.0'" in joined
    steps = ["constraints --backend miles", "uv pip install", "check --backend miles",
             f"--out {image_layer.START_LOCK}", "assert_image_versions.py"]
    positions = [next(i for i, c in enumerate(commands) if step in c) for step in steps]
    assert positions == sorted(positions)


def test_modal_harbor_layer_ships_the_layer_files_and_the_checker():
    calls: list[tuple] = []

    class FakeImage:
        def __getattr__(self, name):
            def record(*args, **kwargs):
                calls.append((name, args, kwargs))
                return self
            return record

    image_layer.modal_harbor_layer(FakeImage(), "verl", "/workspace/verl/.venv/bin/python")
    names = [c[0] for c in calls]
    assert names == ["add_local_dir", "add_local_file", "run_commands"]
    assert calls[0][1] == (str(LAYER), image_layer.IMAGE_LAYER_DIR) and calls[0][2] == {"copy": True}
    assert calls[1][1][1] == f"{image_layer.IMAGE_DIR}/assert_image_versions.py"
    assert any("check --backend verl" in c for c in calls[2][1])
