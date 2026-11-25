"""The verl layout preflight (rl-fleet-verl-islands 4.1).

A verl island plans its full-parameter layout from the HF model's parameter shapes
(no card or shard in it), so the launcher computes the same layout on the control host
(``adapters.verl.launch_flags.offline_full_parameter_layout``: the model on the meta
device from its ``config.json``) and refuses, before anything is ordered, a
``--fragments`` or a ``--rl-parameter-layout-sha256`` the islands would refuse at their
start. A fleet island added at run time runs the same check in its launch's dry run.
CPU only; a tiny Qwen3 config, no weights, no Hub.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from test_fleet_verl_island_launches import _run, verl_doc
from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.launch import fragment_preflight
from yeto.rl import island_launches as il
from yeto.rl import up

TINY = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 64,
        "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "vocab_size": 1024, "tie_word_embeddings": True, "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "torch_dtype": "bfloat16"}
RAW = json.dumps(TINY).encode()
REVISION = "a" * 40
GPU = "modal:1x2xh100,modal:1x2xh100"


@pytest.fixture(autouse=True)
def _tiny_model(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "verl-layout-preflight-test-key")
    monkeypatch.setattr(fragment_preflight, "local_hf_config", lambda model, revision: RAW)


def _planned(fragments: int, merge: str = "rda", revision: str = REVISION, max_bytes: int = 2 << 30):
    """The layout a verl island of the tiny model plans (its own planner, from the model's
    shapes on the meta device), as the island binds it."""
    import tempfile
    from pathlib import Path

    from yeto.rl.adapters.verl import full_parameter as fp
    from yeto.rl.adapters.verl.full_parameter_plan import model_shapes
    from yeto.rl.local_learner import ComponentIdentity

    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "config.json").write_bytes(RAW)
        shapes = model_shapes(directory)
    component = ComponentIdentity(role="actor", model_revision=revision, config_hash=hashlib.sha256(RAW).hexdigest())
    return fp.plan_layout(fp.island_specs(shapes), component=component, minimum_fragments=fragments,
                          max_fragment_bytes=max_bytes, matrix_merge=merge)


def _prepared(*, fragments: int, layout_hash: str, revision: str | None = REVISION, gpu: str = GPU, extra=()):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", (
        "--rl-backend", "verl", "--rl-sync-preset", "full-parameter", "--tuning", "full",
        "--rl-parameter-layout-sha256", layout_hash,
        "--fragments", str(fragments), "--reward-function", "gsm8k_reward:score", "--inner-lr", "1e-6",
        *extra), gpu=gpu)
    args.model_revision = revision
    args.controller = "local"
    _prepare_rl_args(args)
    return args


def test_the_offline_layout_is_the_islands_own_on_any_island_shape():
    from yeto.rl.adapters.verl import launch_flags

    island = _planned(4)
    sha = hashlib.sha256(RAW).hexdigest()
    for tp, pp in ((1, 1), (2, 1), (1, 2)):
        layout = launch_flags.offline_full_parameter_layout(
            TINY, tp=tp, pp=pp, minimum_fragments=4, max_fragment_bytes=2 << 30, matrix_merge="rda",
            model_revision=REVISION, config_sha256=sha)
        assert (layout.fragments, layout.layout_hash, layout.exact) == (
            island.fragments.num_fragments, island.layout_hash, True)
    # without a commit revision the hash is not known (it binds the revision); the count is
    anonymous = launch_flags.offline_full_parameter_layout(
        TINY, minimum_fragments=4, max_fragment_bytes=2 << 30, matrix_merge="rda", model_revision="main",
        config_sha256=sha)
    assert anonymous.layout_hash is None and anonymous.fragments == island.fragments.num_fragments
    # a model the meta-device build does not take: out of scope, nothing refused
    assert launch_flags.offline_full_parameter_layout(
        {"model_type": "no-such-model"}, minimum_fragments=4, max_fragment_bytes=2 << 30) is None


def test_a_matching_count_and_hash_pass_and_dp_above_one_stays_in_scope():
    layout = _planned(4)
    # two 2-card islands (FSDP2 DP 2): a Miles layout check would be out of scope; verl's is not
    args = _prepared(fragments=layout.fragments.num_fragments, layout_hash=layout.layout_hash)
    assert args.rl_parameter_layout_sha256 == layout.layout_hash


def test_a_wrong_fragment_count_is_refused_before_anything_is_ordered():
    # 64 KiB fragments: the tiny model needs more fragments than --fragments 2
    count = _planned(2, max_bytes=1 << 16).fragments.num_fragments
    assert count > 2
    with pytest.raises(ValueError, match=rf"--fragments 2 does not match the layout the trainer islands build: the "
                                         rf"verl full-parameter layout of org/model on any island shape .* has "
                                         rf"{count} fragments .* Set --fragments {count}$"):
        _prepared(fragments=2, layout_hash="e" * 64, extra=("--rl-sync-fragment-max-bytes", str(1 << 16)))


def test_a_wrong_layout_hash_is_refused_with_the_planned_one():
    layout = _planned(4)
    with pytest.raises(ValueError, match=rf"Set --rl-parameter-layout-sha256 {layout.layout_hash}$"):
        _prepared(fragments=layout.fragments.num_fragments, layout_hash="e" * 64)
    # the merge mode is bound into the hash: a strict sweep (avg) plans another one
    strict = _planned(4, merge="avg")
    assert strict.layout_hash != layout.layout_hash
    _prepared(fragments=strict.fragments.num_fragments, layout_hash=strict.layout_hash,
              extra=("--rl-sync-full-parameter-rounds", "sweep"))


def test_without_a_commit_revision_only_the_count_is_checked():
    count = _planned(4).fragments.num_fragments
    _prepared(fragments=count, layout_hash="e" * 64, revision=None)  # the hash is not known here: not refused


def test_a_fleet_island_added_with_the_fleets_layout_passes_and_a_wrong_pin_is_refused(tmp_path, monkeypatch):
    """``yeto fleet island add/move --dry-run`` runs the island launch's dry run: the same
    offline layout check refuses a fleet whose pinned hash is not the model's."""
    from yeto.rl import fleet_islands

    monkeypatch.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test-secret")
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    layout = _planned(4)

    def member(pin: str):
        run, _ = _run(tmp_path, monkeypatch)
        argv = run.cfg["fleet"]["trainer_argv"]
        argv[argv.index("--rl-parameter-layout-sha256") + 1] = pin
        argv[argv.index("--fragments") + 1] = str(layout.fragments.num_fragments)
        argv += ["--model-revision", REVISION]
        return il.trainer_members(run.cfg, run.fleet_plan())[1]

    def prepare(args):  # the launch's own preparation, its revisions given (no Hub here)
        from yeto.launcher import _prepare_rl_args

        args.data_revision, args.source_sha256, args.reward_sha256 = "b" * 40, "c" * 64, "d" * 64
        _prepare_rl_args(args)

    fleet_islands.launch_dry_run(member(layout.layout_hash), prepare=prepare)
    with pytest.raises(ValueError, match=r"--rl-parameter-layout-sha256 eeeeeeeeeeeeeeee\.\.\. is not the layout"):
        fleet_islands.launch_dry_run(member("e" * 64), prepare=prepare)


def test_run_up_dry_run_checks_every_verl_island_launchs_layout(tmp_path, monkeypatch):
    """``yeto run up`` (dry run too) prepares every launch of a per-island fleet at its
    preflight (``Run.fleet_session_problem``): the same offline layout check passes the
    model's layout and refuses a fleet whose pinned hash is not the layout of its model and
    ``--fragments``."""
    monkeypatch.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test-secret")
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    layout = _planned(4)

    def problem(pin: str, fragments: int):
        doc = verl_doc(tmp_path)
        argv = doc["fleet"]["trainer_argv"]
        argv[argv.index("--rl-parameter-layout-sha256") + 1] = pin
        argv[argv.index("--fragments") + 1] = str(fragments)
        argv += ["--model-revision", REVISION]
        path = tmp_path / "run.json"
        path.write_text(json.dumps(doc))
        return up.Run(up.load_config(str(path)), state_root=tmp_path / "state").fleet_session_problem()

    assert problem(layout.layout_hash, layout.fragments.num_fragments) is None
    wrong_pin = problem("e" * 64, layout.fragments.num_fragments)
    assert wrong_pin and "--rl-parameter-layout-sha256 eeeeeeeeeeeeeeee... is not the layout" in wrong_pin
    assert layout.layout_hash in wrong_pin
    # another --fragments is another layout: the pin of 4 fragments is refused, naming the hash of 5
    other = _planned(layout.fragments.num_fragments + 1)
    wrong_count = problem(layout.layout_hash, layout.fragments.num_fragments + 1)
    assert wrong_count and f"Set --rl-parameter-layout-sha256 {other.layout_hash}" in wrong_count
