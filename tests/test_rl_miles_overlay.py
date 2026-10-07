"""S13 Miles overlay (critic family on image c35702e): CPU only, no Ray."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from yeto import launcher
from yeto.rl import MILES_NEXT_COMMIT
from yeto.rl import miles_overlay as mo
from yeto.rl.algos import critic_fork, sao
from yeto.rl.algos.compactionrl import compactionrl_spec
from yeto.rl.algos.vapo import vapo_spec
from yeto.rl.engine.algorithm import AlgorithmSpec


def _args(spec=None, **kw):
    base = dict(training_mode="rl", rl_engine="ports", cluster_prefix="t",
                rl_algorithm_spec_json=spec.canonical_json() if spec else None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_patch_file_matches_pinned_sha_and_fork_pin():
    data = mo.repo_patch_path().read_bytes()
    assert hashlib.sha256(data).hexdigest() == mo.CRITIC_C357_PATCH_SHA256
    assert mo.CRITIC_C357_BASE_COMMIT == MILES_NEXT_COMMIT
    assert mo.CRITIC_C357_RESULT_COMMIT == critic_fork.CRITIC_FORK_PIN


def test_patch_adds_exactly_the_fork_only_flags():
    added, in_args = set(), False
    for line in mo.repo_patch_path().read_text().splitlines():
        if line.startswith("+++ "):
            in_args = line.endswith("miles/utils/arguments.py")
        elif in_args and line.startswith("+"):
            added |= set(re.findall(r'"(--[a-z0-9-]+)"', line))
    assert added == mo.FORK_ONLY_FLAGS


@pytest.mark.parametrize("spec", [None, AlgorithmSpec(), AlgorithmSpec(advantage_estimator="ppo",
                                                                       execution={"needs_critic": True})])
def test_auto_is_off_without_fork_flags(spec):
    assert mo.resolve_overlay(_args(spec)) is None
    assert launcher._miles_overlay_manifest(_args(spec), "ports") == {}


@pytest.mark.parametrize("make", [vapo_spec, compactionrl_spec, lambda: sao.sao_algorithm_spec("coding")])
def test_auto_turns_on_for_critic_family_specs(make):
    args = _args(make())
    assert mo.spec_fork_flags(args.rl_algorithm_spec_json)
    assert mo.resolve_overlay(args) == mo.CRITIC_C357
    assert mo.resolve_overlay(_args(make(), rl_miles_overlay="off")) is None


def test_explicit_and_invalid_choices():
    assert mo.resolve_overlay(_args(None, rl_miles_overlay="critic-c357")) == mo.CRITIC_C357
    with pytest.raises(ValueError):
        mo.resolve_overlay(_args(None, rl_miles_overlay="critic-c357", rl_engine="legacy"))
    with pytest.raises(ValueError):
        mo.resolve_overlay(_args(None, rl_miles_overlay="bogus"))
    assert mo.resolve_overlay(_args(None, training_mode="sft")) is None


def test_setup_unchanged_when_off_and_appended_when_on():
    off = launcher._miles_source_setup("ports")
    assert launcher._miles_source_setup("ports", None) == off
    on = launcher._miles_source_setup("ports", mo.CRITIC_C357)
    assert on[1] == off[1] and on[0].startswith(off[0])
    tail = on[0][len(off[0]):]
    for needle in (mo.CRITIC_C357_PATCH_SHA256, "REFUSING", "MILES_REFRESHED", "git -C ~/miles apply",
                   mo.OVERLAY_RECORD_PATH, MILES_NEXT_COMMIT):
        assert needle in tail
    assert tail.index("sha256sum --check") < tail.index("apply --check") < tail.index("apply ~/")
    with pytest.raises(ValueError):
        launcher._miles_source_setup("legacy", mo.CRITIC_C357)


def test_run_manifest_records_overlay_and_mismatch():
    rec = launcher._miles_overlay_manifest(_args(vapo_spec()), "ports")["miles_overlay"]
    assert rec["patch_sha256"] == mo.CRITIC_C357_PATCH_SHA256
    assert rec["image_manifest_matches_code"] is False and rec["result_commit_pushed"] is False
    assert rec["summary"] == f"image miles {MILES_NEXT_COMMIT[:7]} + overlay {mo.CRITIC_C357_PATCH_SHA256}"


def _fake_image(tmp_path, *, head_ok=True, manifest_commit=MILES_NEXT_COMMIT):
    """A ~/miles git repo with the patch's files at base content is not
    available on CPU; instead an empty repo whose HEAD check decides refusal."""
    home = tmp_path / "home"
    miles = home / "miles"
    miles.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(miles)], check=True)
    subprocess.run(["git", "-C", str(miles), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    manifest = tmp_path / "image-manifest.json"
    manifest.write_text(json.dumps({"miles": {"commit": manifest_commit, "path": str(miles)}}))
    return home, manifest


@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("git") is None, reason="needs bash+git")
def test_setup_refuses_when_miles_is_not_the_image_c35702e(tmp_path):
    home, manifest = _fake_image(tmp_path)
    script = mo.overlay_setup(mo.CRITIC_C357).replace(mo.MILES_NEXT_IMAGE_MANIFEST, str(manifest))
    for refreshed in ("0", "1"):
        out = subprocess.run(["bash", "-c", f"set -e\nMILES_REFRESHED={refreshed}\n{script}"],
                             env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
                             capture_output=True, text=True)
        assert out.returncode != 0 and "REFUSING" in out.stderr
        assert not (home / ".yeto-miles-overlay.json").exists()


def test_runtime_manifest_flags_overlay(tmp_path, monkeypatch):
    from yeto.rl.engine import runtime_manifest as rm

    record = tmp_path / "rec.json"
    record.write_text(json.dumps({**mo.overlay_record(mo.CRITIC_C357), "applied": True}))
    monkeypatch.setattr(mo, "read_applied_record", lambda path=None: json.loads(record.read_text()))
    m = rm.collect(image=None)
    assert m["image_manifest_matches_code"] is False
    assert m["miles_overlay"]["patch_sha256"] == mo.CRITIC_C357_PATCH_SHA256
    assert m["commit_source"]["miles"].endswith("+overlay")
    monkeypatch.setattr(mo, "read_applied_record", lambda path=None: None)
    m = rm.collect(image=None)
    assert "miles_overlay" not in m and "+overlay" not in m["commit_source"]["miles"]
