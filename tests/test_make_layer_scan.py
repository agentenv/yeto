"""docker/miles-ports/make_layer.py: the base's SGLang editable-install metadata is
found by scanning (a layer listing, a metadata directory), never by a layer number
or a version string written into the script (CPU, fake manifests and listings)."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "docker" / "miles-ports" / "make_layer.py"


@pytest.fixture(scope="module")
def ml():
    spec = importlib.util.spec_from_file_location("make_layer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SITE = "opt/sglang/lib/python3.12/site-packages"


def _listing(version: str, with_finder: bool = True) -> list[str]:
    mod = "__editable___sglang_" + version.replace(".", "_").replace("+", "_")
    names = [f"{SITE}/sglang-{version}.dist-info/", f"{SITE}/sglang-{version}.dist-info/METADATA",
             f"{SITE}/sglang-{version}.dist-info/RECORD", f"{SITE}/__editable__.sglang-{version}.pth",
             "sgl-workspace/sglang/python/sglang/__init__.py", "root/.cache/x"]
    if with_finder:
        names.append(f"{SITE}/{mod}_finder.py")
    return names


def test_metadata_is_recognised_in_a_layer_listing_for_any_version(ml):
    for version in ("0.5.21.dev64+ge5bba1f", "0.5.23", "0.5.24.dev3+g35708f7", "0.5.23.post1+miles.g35708f7"):
        found = ml.sglang_metadata_names(_listing(version))
        assert found["version"] == version
        assert found["dist_info"] == f"sglang-{version}.dist-info"
        assert found["pth"] == f"__editable__.sglang-{version}.pth"
        assert found["finder"].startswith("__editable___sglang_") and found["finder"].endswith("_finder.py")
        assert ml.layer_holds_sglang_metadata(_listing(version))
    # a directory listing (no site-packages prefix) works the same; a missing
    # finder file name is derived from the version (as setuptools spells it)
    found = ml.sglang_metadata_names([n[len(SITE) + 1:] for n in _listing("0.5.23", with_finder=False)
                                      if n.startswith(SITE)])
    assert found["finder"] == ml.editable_module_name("0.5.23") + "_finder.py"
    # other layers: torch's dist-info, the sglang source tree alone, nothing
    assert not ml.layer_holds_sglang_metadata([f"{SITE}/torch-2.14.1.dist-info/METADATA",
                                               "sgl-workspace/sglang/python/sglang/version.py"])
    assert not ml.layer_holds_sglang_metadata([])
    # the rust extension's dist-info is not the sglang one
    assert not ml.layer_holds_sglang_metadata([f"{SITE}/sglang_kernel-0.4.9.dist-info/METADATA",
                                               f"{SITE}/sglang-router-0.2.dist-info/METADATA"])


def _history(steps: list[str]) -> dict:
    hist = []
    for step in steps:
        if step == "ENV":
            hist.append({"created_by": "ENV X=1", "empty_layer": True})
        else:
            hist.append({"created_by": step})
    return {"history": hist}


def test_candidate_layers_follow_the_build_history_then_newest_first(ml):
    steps = [
        "RUN apt-get install -y libibverbs1",
        "ENV",
        "RUN cd /tmp/sglang_deps/python && uv pip install .[all]",   # deps only: weak hint
        "RUN git clone Megatron-LM && pip install -e .",             # another editable install
        "ENV",
        'RUN cd /sgl-workspace/sglang && git checkout -f $SGLANG_COMMIT && pip install -e "python[all]" --no-deps',
        "RUN git clone miles /root/miles && pip install -e . --no-deps",
        "RUN pip install pyrage",
    ]
    layers = [{"digest": f"sha256:{i:064x}", "size": 10 + i} for i in range(6)]
    order = ml.candidate_metadata_layers({"layers": layers}, _history(steps))
    # the SGLang editable install (layer 3) first, the deps layer (1) next, then 5, 4, 2, 0
    assert order == [3, 1, 5, 4, 2, 0]
    # a history that does not line up with the layers is ignored: last layer first
    bad = _history(steps[:3])
    assert ml.candidate_metadata_layers({"layers": layers}, bad) == [5, 4, 3, 2, 1, 0]
    assert ml.candidate_metadata_layers({"layers": layers}, {}) == [5, 4, 3, 2, 1, 0]


def test_helper_modes_run_from_the_command_line(tmp_path):
    layers = [{"digest": f"sha256:{i:064x}", "size": 1} for i in range(3)]
    (tmp_path / "m.json").write_text(json.dumps({"layers": layers}))
    (tmp_path / "c.json").write_text(json.dumps(_history([
        "RUN apt", 'RUN cd /sgl-workspace/sglang && pip install -e "python[all]"', "RUN pip install x"])))
    out = subprocess.run([sys.executable, "-I", str(SCRIPT), "find-layer", "--manifest", str(tmp_path / "m.json"),
                          "--config", str(tmp_path / "c.json")], check=True, capture_output=True, text=True).stdout
    assert [row["index"] for row in json.loads(out)] == [1, 2, 0]
    assert json.loads(out)[0]["digest"] == layers[1]["digest"]
    ok = subprocess.run([sys.executable, "-I", str(SCRIPT), "check-listing"],
                        input="\n".join(_listing("0.5.23")), capture_output=True, text=True)
    assert ok.returncode == 0
    no = subprocess.run([sys.executable, "-I", str(SCRIPT), "check-listing"],
                        input="root/miles/README.md\n", capture_output=True, text=True)
    assert no.returncode == 1


def _wheel(path: Path, dist: str, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
        zf.writestr(f"{dist}.dist-info/METADATA", b"Name: x\n")
        zf.writestr(f"{dist}.dist-info/RECORD", b"")
    return path


def test_wheels_pure_python_always_binary_only_for_the_image_platform(ml, tmp_path):
    pure = _wheel(tmp_path / "peft-0.20.0-py3-none-any.whl", "peft-0.20.0", {"peft/__init__.py": b""})
    assert set(ml.wheel_files(pure)) == {"peft/__init__.py", "peft-0.20.0.dist-info/METADATA",
                                         "peft-0.20.0.dist-info/RECORD", "peft-0.20.0.dist-info/INSTALLER"}
    binary = _wheel(tmp_path / "pyrage-1.4.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
                    "pyrage-1.4.0", {"pyrage/__init__.py": b"", "pyrage/pyrage.abi3.so": b"ELF"})
    with pytest.raises(SystemExit, match="only pure-Python"):
        ml.wheel_files(binary)
    with pytest.raises(SystemExit, match="only pure-Python"):
        ml.wheel_files(binary, platform="aarch64")
    assert "pyrage/pyrage.abi3.so" in ml.wheel_files(binary, platform="x86_64")
    with_data = _wheel(tmp_path / "x-1.0-py3-none-any.whl", "x-1.0", {"x-1.0.data/scripts/x": b"#!"})
    with pytest.raises(SystemExit, match=".data"):
        ml.wheel_files(with_data)


def test_version_file_and_editable_module_name(ml):
    text = ml.version_file("0.5.24.dev3+g35708f7").decode()
    assert "__version__ = version = '0.5.24.dev3+g35708f7'" in text
    assert ml.editable_module_name("0.5.24.dev3+g35708f7") == "__editable___sglang_0_5_24_dev3_g35708f7"


def test_k31_lock_names_the_verified_base_and_the_checker_reads_it():
    lock = json.loads((REPO / "docker" / "miles-k31" / "versions.lock.json").read_text())
    assert lock["cpu_arch"] == "x86_64" and lock["python"] == "3.12"
    base = lock["base_image"]
    assert base["name"] == "docker.io/radixark/miles:kimi-k31"
    assert base["manifest_digest"].startswith("sha256:4a3295266f102ad9")
    assert base["index_digest"].startswith("sha256:7b640c2b5ff9744d")
    assert lock["base_build_args"]["CUDA_VERSION"] == "13.0.3"
    assert lock["base_build_args"]["SGL_NCCL_VERSION"] == "2.30.7"
    assert lock["commits"]["torch_memory_saver"].startswith("118b09b3")
    assert lock["packages"]["nvidia-nccl-cu13"]["version"] == "2.30.7"
    # the stdlib checker accepts the lock's shape (every check fails or records here: no image)
    spec = importlib.util.spec_from_file_location(
        "assert_image_versions", REPO / "docker" / "image-assert" / "assert_image_versions.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    report = checker.run(lock, commits_file=Path("/nonexistent/commits.json"), machine="x86_64")
    checks = {row["check"] for row in report.rows}
    assert {"cpu_arch", "python", "package pyrage", "commits", "driver"} <= checks
    assert any(row["check"] == "cpu_arch" and row["status"] == "ok" for row in report.rows)


def test_build_scripts_parse():
    for script in ("docker/miles-k31/build.sh", "scripts/build_miles_ports_image.sh"):
        subprocess.run(["bash", "-n", str(REPO / script)], check=True)
    text = (REPO / "scripts" / "build_miles_ports_image.sh").read_text()
    assert '["layers"][104]' not in text and "0.5.21.dev64" not in text
    k31 = (REPO / "docker" / "miles-k31" / "build.sh").read_text()
    assert "find-layer" in k31 and "check-listing" in k31 and "--image-assert-dir" in k31
    # no TMS preload mount or patch anywhere in the K3.1 build
    assert "tms-preload" not in k31 and "hook_mode_preload" not in k31 and "TMS_PRELOAD" not in k31


def _git_repo(path: Path, files: dict[str, str]) -> str:
    path.mkdir()
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "HOME": str(path), "PATH": "/usr/bin:/bin"}
    run = lambda *a: subprocess.run(["git", "-C", str(path), *a], check=True, capture_output=True, env=env)
    run("init", "-q")
    for name, text in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text)
    run("add", "-A")
    run("commit", "-q", "-m", "c")
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True, env=env).stdout.strip()


def _commit(path: Path, files: dict[str, str | None]) -> str:
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "HOME": str(path), "PATH": "/usr/bin:/bin"}
    for name, text in files.items():
        if text is None:
            (path / name).unlink()
        else:
            (path / name).parent.mkdir(parents=True, exist_ok=True)
            (path / name).write_text(text)
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "d"], check=True, env=env)
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True, env=env).stdout.strip()


def test_fork_diff_rules_deletions_whiteout_metadata_behind_a_flag_native_refused(ml, tmp_path):
    """A fork that is not a descendant of the base (k31/integ vs the base's
    upstream-pr/4667 head): CI, recipes and examples differ, a recipe is deleted,
    setup.py gains an extra. Deleted files are reported (the layer writes whiteouts),
    install metadata needs the flag, a native file is refused outright."""
    repo = tmp_path / "fork"
    base = _git_repo(repo, {"miles/__init__.py": "", "miles/a.py": "1", "scripts/old_recipe.py": "x",
                            "setup.py": "setup()", ".github/ci.yml": "a", "miles/kernels/k.cu": "k"})
    head = _commit(repo, {"miles/a.py": "2", "scripts/old_recipe.py": None, "scripts/new.py": "y",
                          ".github/ci.yml": "b", "examples/e.py": "e"})
    changed, deleted, drift = ml.check_pure_python(repo, base, head, ("miles/",))
    assert deleted == ["scripts/old_recipe.py"] and drift == []
    assert set(changed) == {"miles/a.py", "scripts/new.py", ".github/ci.yml", "examples/e.py"}
    meta = _commit(repo, {"setup.py": "setup(extras={'novita': []})", "requirements.txt": "pyrage==1.4.0\n"})
    with pytest.raises(SystemExit, match="install metadata"):
        ml.check_pure_python(repo, base, meta, ("miles/",))
    _changed, _deleted, drift = ml.check_pure_python(repo, base, meta, ("miles/",), allow_install_metadata_drift=True)
    assert sorted(drift) == ["requirements.txt", "setup.py"]
    native = _commit(repo, {"miles/kernels/k.cu": "k2"})
    with pytest.raises(SystemExit, match="native"):
        ml.check_pure_python(repo, base, native, ("miles/",), allow_install_metadata_drift=True)
    binary_in_package = _commit(repo, {"miles/kernels/k.cu": "k", "miles/data.bin": "\x00"})
    with pytest.raises(SystemExit, match="not a pure-Python"):
        ml.check_pure_python(repo, base, binary_in_package, ("miles/",), allow_install_metadata_drift=True)


def test_build_sh_reads_the_pins_by_running_them_and_passes_the_drift_flag():
    k31 = (REPO / "docker" / "miles-k31" / "build.sh").read_text()
    assert "runpy.run_path" in k31 and "ALLOW_INSTALL_METADATA_DRIFT:+--allow-install-metadata-drift" in k31
    # the alias form in pins.py resolves to a URL, which the old regex missed
    import runpy
    ns = runpy.run_path(str(REPO / "yeto" / "rl" / "adapters" / "miles" / "pins.py"))
    assert ns["SGLANG_NEXT_REPOSITORY"].startswith("https://") and ns["MILES_NEXT_REPOSITORY"].startswith("https://")
