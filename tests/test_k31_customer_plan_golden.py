"""The Kimi-K3.1 customer path keeps its launch plan byte for byte.

The customer runs one K3.1 buffer trainer (``--rl-single-island-no-sync``, no
syncer) through ``yeto run up``; on a first launch ``run up`` passes the run
config's ``trainer.argv`` unchanged to ``yeto launch``, so its dry-run plan is
the launch dry-run plan of that argv. This test pins that plan for:

* ``examples/k31/run.agentic-262k.json`` and ``run.baremetal-b300-262k.json``
  (placeholders filled as in ``tests/test_k31_baremetal_kit.py``);
* the customer kit's rendered ``run.json`` (``tests/fixtures/k31_customer_kit``:
  rendered by the kit's ``render.py`` from its self-test inputs, documentation
  addresses and an example bucket, paths replaced by placeholders).

The golden (``tests/fixtures/k31_customer_kit/plans.json``) was generated on
main before the sync-route wiring; the source tree hash is fixed, everything
else is compared. The test is hermetic: ``HOME`` is a temporary directory
holding a placeholder Hugging Face token file (the customer form: the token
is mounted onto every node), and the variables the launcher forwards from the
operator's environment are cleared, so the plan does not depend on the machine
that runs it. Regenerate only for an intended change:
``YETO_UPDATE_K31_PLAN_GOLDEN=1 pytest tests/test_k31_customer_plan_golden.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
K31 = ROOT / "examples" / "k31"
KIT = ROOT / "tests" / "fixtures" / "k31_customer_kit"
GOLDEN = KIT / "plans.json"
CASES = {
    "examples/run.agentic-262k.json": K31 / "run.agentic-262k.json",
    "examples/run.baremetal-b300-262k.json": K31 / "run.baremetal-b300-262k.json",
    "kit/run.json": KIT / "run.json",
}
PLACEHOLDER_TOKEN = "hf_placeholder_token_for_the_plan_golden"
# the launcher copies these from the operator's environment into the plan (island env, mounts)
FORWARDED_PREFIXES = ("HF_", "HUGGING_FACE_", "WANDB_", "CYBERGYM_", "NCCL_", "YETO_", "SKYPILOT_", "AWS_",
                      "MODAL_")


def _selection(tmp_path: Path) -> Path:
    from yeto.rl.adapters.miles.models.partial_experts.selection import ExpertSelection

    path = tmp_path / "expert_selection.ep16.json"
    ExpertSelection.synthetic("uniform", model="kimi_k31", num_experts=896, num_trainable=128, ep_size=16,
                              layers=range(1, 93)).write(path)
    return path


def _argv(run: Path, tmp_path: Path) -> list[str]:
    selection = str(_selection(tmp_path))
    fill = {
        "<run>": "bmkit", "<bucket>": "bmbucket", "<store-endpoint>": "https://store.example",
        "<inventory.b300-6x8-baremetal.yaml>": str(K31 / "inventory.b300-6x8-baremetal.yaml"),
        "<expert_selection.ep16.json>": selection, "<SELECTION>": selection,
        "<algorithm-spec.async-age12.json>": str(K31 / "algorithm-spec.async-age12.json"),
        "<model-revision>": "a" * 40, "<data-revision>": "b" * 40, "<prompt-dataset>": "org/prompts",
        "<module:function>": "yeto.rl.numeric_answer_reward:score", "<total-steps>": "100",
        "<weights-volume>": "k31-weights", "<modal-secret>": "k31-store", "<KIT>": str(KIT),
    }
    out = []
    for arg in json.loads(run.read_text())["trainer"]["argv"]:
        for key, value in fill.items():
            arg = str(arg).replace(key, value)
        assert "<" not in arg, f"unfilled placeholder in {arg!r}"
        out.append(arg)
    return out


def _normalize(text: str, tmp_path: Path) -> str:
    for real, label in ((str(tmp_path / "home"), "<HOME>"), (str(tmp_path), "<TMP>"), (str(KIT), "<KIT>"),
                        (str(ROOT), "<REPO>")):
        text = text.replace(real, label)
    return text


def _hermetic_env(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    token = home / ".cache" / "huggingface" / "token"
    token.parent.mkdir(parents=True)
    token.write_text(PLACEHOLDER_TOKEN)
    for key in list(os.environ):
        if key.startswith(FORWARDED_PREFIXES):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(home))
    import yeto.runs as runs

    monkeypatch.setattr(runs, "RUNS_DIR", home / ".yeto" / "runs")  # resolved from HOME at import


def _plan(run: Path, tmp_path: Path, monkeypatch, capsys) -> dict:
    import yeto.cli as cli
    import yeto.launcher as launcher
    import yeto.modal_runner as modal_runner
    from yeto.cloud import s3_store

    _hermetic_env(tmp_path, monkeypatch)
    for key, value in (("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test"), ("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test"),
                       ("HF_HUB_OFFLINE", "1"), ("HF_HUB_CACHE", str(tmp_path / "hf")),
                       ("HF_HOME", str(tmp_path / "hf-home"))):
        monkeypatch.setenv(key, value)
    for key in (s3_store.ENDPOINT_ENV, s3_store.REGION_ENV, s3_store.PROFILE_ENV, "AWS_ENDPOINT_URL",
                "AWS_PROFILE"):
        monkeypatch.setenv(key, "")  # records the variable, so the launcher's own setting is undone too
        monkeypatch.delenv(key)
    monkeypatch.setattr(modal_runner, "modal_available", lambda: True)
    # the source tree hash changes with every commit; the plan is pinned for everything else
    import yeto.provenance as provenance

    monkeypatch.setattr(provenance, "verify_source_tree_sha256", lambda expected=None: "5" * 64)
    calls = []
    monkeypatch.setattr(launcher, "run", lambda *a, **k: calls.append("run"))
    monkeypatch.setattr(cli, "_spawn_worker", lambda *a, **k: calls.append("spawn"))
    monkeypatch.setattr(cli.runs, "create_run", lambda *a, **k: calls.append("create"))
    argv = _argv(run, tmp_path)
    args = cli.parse_args([*argv, "--dry-run", "--preflight-model", "off"])
    assert cli.cmd_launch(args) == 0 and calls == []
    out = capsys.readouterr().out
    plan = json.loads(out[:out.rindex("}") + 1])
    text = json.dumps({"argv": argv, "plan": plan}, sort_keys=True)
    assert PLACEHOLDER_TOKEN not in text  # the token file is mounted, never copied into the plan
    return json.loads(_normalize(text, tmp_path))


@pytest.mark.parametrize("case", sorted(CASES))
def test_k31_customer_plan_is_unchanged(case, tmp_path, monkeypatch, capsys):
    update = os.environ.get("YETO_UPDATE_K31_PLAN_GOLDEN") == "1"  # read before the environment is cleared
    got = _plan(CASES[case], tmp_path, monkeypatch, capsys)
    assert "--rl-single-island-no-sync" in got["argv"]  # the customer form: no syncer, never routed
    if update:
        golden = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        golden[case] = got
        GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
        return
    assert got == json.loads(GOLDEN.read_text())[case]
