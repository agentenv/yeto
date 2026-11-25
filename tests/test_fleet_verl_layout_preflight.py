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
from yeto.launch.fragment_preflight import remote_hf_config as REAL_REMOTE_HF_CONFIG  # conftest patches the name
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
    monkeypatch.delenv(fragment_preflight.LAYOUT_MODE_ENV, raising=False)
    monkeypatch.setattr(fragment_preflight, "local_hf_config", lambda model, revision: RAW)
    monkeypatch.setattr(fragment_preflight, "_WARNED", set())  # each test sees its own WARN lines


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


def _run_up_problem(tmp_path, monkeypatch, pin: str, fragments: int, *, dry_run: bool = False):
    """``Run.fleet_session_problem`` of a two-island verl fleet pinning ``pin`` at
    ``--fragments fragments`` (``dry_run``: ``yeto run up --dry-run``'s preflight)."""
    monkeypatch.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test-secret")
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    doc = verl_doc(tmp_path)
    argv = doc["fleet"]["trainer_argv"]
    argv[argv.index("--rl-parameter-layout-sha256") + 1] = pin
    argv[argv.index("--fragments") + 1] = str(fragments)
    argv += ["--model-revision", REVISION]
    path = tmp_path / "run.json"
    path.write_text(json.dumps(doc))
    return up.Run(up.load_config(str(path)), state_root=tmp_path / "state").fleet_session_problem(dry_run=dry_run)


def test_run_up_dry_run_checks_every_verl_island_launchs_layout(tmp_path, monkeypatch):
    """``yeto run up`` (dry run too) prepares every launch of a per-island fleet at its
    preflight (``Run.fleet_session_problem``): the same offline layout check passes the
    model's layout and refuses a fleet whose pinned hash is not the layout of its model and
    ``--fragments``."""
    layout = _planned(4)

    def problem(pin: str, fragments: int):
        return _run_up_problem(tmp_path, monkeypatch, pin, fragments)

    assert problem(layout.layout_hash, layout.fragments.num_fragments) is None
    wrong_pin = problem("e" * 64, layout.fragments.num_fragments)
    assert wrong_pin and "--rl-parameter-layout-sha256 eeeeeeeeeeeeeeee... is not the layout" in wrong_pin
    assert layout.layout_hash in wrong_pin
    # another --fragments is another layout: the pin of 4 fragments is refused, naming the hash of 5
    other = _planned(layout.fragments.num_fragments + 1)
    wrong_count = problem(layout.layout_hash, layout.fragments.num_fragments + 1)
    assert wrong_count and f"Set --rl-parameter-layout-sha256 {other.layout_hash}" in wrong_count


# -- the model's config.json not on this host -------------------------------------------------

UNVERIFIED = "verl layout hash not verified offline"


def test_without_the_models_config_run_up_refuses_a_pinned_verl_fleet_and_its_dry_run_warns(
        tmp_path, monkeypatch, capsys):
    """No config.json here and none fetched (unit tests fetch nothing): ``yeto run up``
    refuses the fleet (each island would check its pin only after its paid boot), naming
    why, the matrix merge the check plans with and how to verify or go on; its dry run
    passes with one WARN line; ``YETO_PREFLIGHT_LAYOUT=warn`` lets ``run up`` go on."""
    monkeypatch.setattr(fragment_preflight, "local_hf_config", lambda model, revision: None)
    layout = _planned(4)
    count = layout.fragments.num_fragments

    refused = _run_up_problem(tmp_path, monkeypatch, layout.layout_hash, count)
    assert refused and f"ValueError: {UNVERIFIED}: config.json of Qwen/Qwen3-4B@{REVISION} is not in this host's " \
                      "model files or Hugging Face cache, and fetching it failed: unit tests fetch nothing" in refused
    assert "The check plans with matrix merge rda (--rl-sync-matrix-merge)" in refused
    assert f"the launch pins --rl-parameter-layout-sha256 {layout.layout_hash[:16]}..." in refused
    assert f"hf download Qwen/Qwen3-4B config.json --revision {REVISION}" in refused
    assert "to launch unverified, set YETO_PREFLIGHT_LAYOUT=warn" in refused

    capsys.readouterr()
    assert _run_up_problem(tmp_path, monkeypatch, layout.layout_hash, count, dry_run=True) is None
    warned = [line for line in capsys.readouterr().err.splitlines() if UNVERIFIED in line]
    assert len(warned) == 1  # once, though every launch of the fleet is prepared
    assert warned[0].startswith(f"[launcher] WARN: {UNVERIFIED}: ") and "matrix merge rda" in warned[0]

    monkeypatch.setenv(fragment_preflight.LAYOUT_MODE_ENV, "warn")
    assert _run_up_problem(tmp_path, monkeypatch, layout.layout_hash, count) is None
    monkeypatch.setenv(fragment_preflight.LAYOUT_MODE_ENV, "maybe")
    assert "YETO_PREFLIGHT_LAYOUT='maybe': expected one of check, warn" in _run_up_problem(
        tmp_path, monkeypatch, layout.layout_hash, count)


def test_a_fetched_config_verifies_the_layout(tmp_path, monkeypatch):
    """Without a local config.json the launcher fetches that one file: the layout of the
    fetched config is checked like a local one, and the refusal names where it came from."""
    monkeypatch.setattr(fragment_preflight, "local_hf_config", lambda model, revision: None)
    fetched = []

    def remote(args):
        fetched.append((args.model, args.model_revision))
        return RAW, f"hf://{args.model}@{args.model_revision}/config.json"

    monkeypatch.setattr(fragment_preflight, "remote_hf_config", remote)
    layout = _planned(4)
    count = layout.fragments.num_fragments
    assert _run_up_problem(tmp_path, monkeypatch, layout.layout_hash, count) is None
    assert fetched and set(fetched) == {("Qwen/Qwen3-4B", REVISION)}
    wrong = _run_up_problem(tmp_path, monkeypatch, "e" * 64, count)
    assert wrong and f"(config.json from hf://Qwen/Qwen3-4B@{REVISION}/config.json)" in wrong
    assert f"Set --rl-parameter-layout-sha256 {layout.layout_hash}" in wrong


def test_a_hash_planned_with_the_wrong_matrix_merge_is_named(tmp_path, monkeypatch):
    """The fleet's islands merge with rda (the async full-parameter default): a pin planned
    with avg is refused, and the refusal says it is the avg layout."""
    rda, avg = _planned(4), _planned(4, merge="avg")
    assert rda.layout_hash != avg.layout_hash
    wrong = _run_up_problem(tmp_path, monkeypatch, avg.layout_hash, rda.fragments.num_fragments)
    assert wrong and "matrix merge rda (--rl-sync-matrix-merge) has hash " + rda.layout_hash in wrong
    assert ("the pinned hash is that layout under matrix merge avg, so it was planned with the wrong matrix merge "
            "(the islands merge with rda)") in wrong
    other = _run_up_problem(tmp_path, monkeypatch, "e" * 64, rda.fragments.num_fragments)
    assert other and "planned with the wrong matrix merge" not in other


def test_fleet_island_add_refuses_an_unverified_pin_and_its_dry_run_warns(tmp_path, monkeypatch, capsys):
    """``yeto fleet island add`` (written) refuses a verl island whose pinned layout cannot
    be planned here; ``--dry-run`` passes with the WARN."""
    import types

    from test_fleet_island_changes import prepare
    from yeto.launch.pool import desired as dz
    from yeto.rl import fleet
    from yeto.rl.engine.object_store import DirectoryObjectStore

    monkeypatch.setattr(fragment_preflight, "local_hf_config", lambda model, revision: None)
    monkeypatch.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test-secret")
    cfg = up.load_config(str(_write(tmp_path, verl_doc(tmp_path))))
    plan = fleet.plan_of(cfg)
    objects = DirectoryObjectStore(tmp_path / "store")
    store = dz.DesiredStore(objects)
    pools = [{"kind": "sampler_pool", "name": p.name, "island": i.index, "pool": {"provider": "modal", "target": 1}}
             for i in plan.islands for p in i.pools]
    store.apply({"members": [*il.trainer_members(cfg, plan), *pools]}, expect=0)
    monkeypatch.setattr("yeto.launch.pool.fleet_cli._store",
                        lambda config: (types.SimpleNamespace(cfg=cfg, _objects=lambda: objects), store))
    samplers = tmp_path / "north.samplers.json"
    samplers.write_text(json.dumps([dict(cfg["islands"][1]["samplers"][0], name="sglang-north")]))

    def add(*extra: str) -> int:
        from yeto.cli import build_parser
        from yeto.rl import fleet_islands

        args = build_parser().parse_args(["fleet", "island", "add", "--name", "north", "--gpu", "modal:1x2xh100",
                                          "--usd-per-h", "8", "--samplers", str(samplers), "--config", "run.json",
                                          *extra])
        return fleet_islands.main(args, prepare=prepare)

    capsys.readouterr()
    assert add("--dry-run") == 0
    err = capsys.readouterr().err
    assert f"[launcher] WARN: {UNVERIFIED}: " in err and "matrix merge rda" in err
    assert add() == 2
    err = capsys.readouterr().err
    assert f"fleet island: refused: {UNVERIFIED}: " in err and "YETO_PREFLIGHT_LAYOUT=warn" in err
    assert store.read()["generation"] == 1  # nothing written


def _write(tmp_path, doc):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(doc))
    return path


# -- fetching config.json by itself (fragment_preflight._fetch_config) ----------------------

def _args(model: str, *, revision: str | None = REVISION, source: str | None = None, volume: str | None = None):
    import types

    return types.SimpleNamespace(model=model, model_revision=revision, model_store_source=source,
                                 modal_model_volume=volume)


def _store(files: dict):
    def store(uri):
        return list(files), files.get
    return store


def test_the_config_is_fetched_from_the_hub_at_the_pinned_revision():
    asked = []

    def hub(repo, name, revision):
        asked.append((repo, name, revision))
        return RAW

    assert fragment_preflight._fetch_config(_args("Qwen/Qwen3-4B"), hub=hub) == (
        RAW, f"hf://Qwen/Qwen3-4B@{REVISION}/config.json")
    assert asked == [("Qwen/Qwen3-4B", "config.json", REVISION)]  # that one file, at the pinned commit


def test_a_store_model_and_a_model_store_source_give_their_config():
    store = _store({"config.json": RAW, "model.safetensors": b""})
    assert fragment_preflight._fetch_config(_args("s3://bkt/models/q"), store=store) == (
        RAW, "s3://bkt/models/q/config.json")
    no_hub = lambda *a: (_ for _ in ()).throw(AssertionError("the store source comes first"))  # noqa: E731
    assert fragment_preflight._fetch_config(_args("Qwen/Qwen3-4B", source="s3://bkt/q/"), hub=no_hub,
                                            store=store) == (RAW, "s3://bkt/q/config.json")
    raw, why = fragment_preflight._fetch_config(_args("s3://bkt/models/q"), store=_store({}))
    assert raw is None and why == "s3://bkt/models/q has no config.json"


def test_the_model_volume_snapshot_is_the_fallback_and_every_failure_is_named():
    def offline(repo, name, revision):
        raise OSError("no network")

    snaps = []

    def volume(vol, snap):
        snaps.append((vol, snap))
        return ["config.json", "MANIFEST.json"], {"config.json": RAW}.get

    assert fragment_preflight._fetch_config(_args("Qwen/Qwen3-4B", volume="models"), hub=offline, volume=volume) == (
        RAW, f"modal volume models:/hf/Qwen3-4B/{REVISION[:8]}/config.json")
    assert snaps == [("models", f"hf/Qwen3-4B/{REVISION[:8]}")]
    raw, why = fragment_preflight._fetch_config(_args("Qwen/Qwen3-4B", volume="models"), hub=offline,
                                                volume=lambda vol, snap: None)
    assert raw is None
    assert why == (f"the Hub did not give hf://Qwen/Qwen3-4B@{REVISION}/config.json (OSError: no network); "
                   f"modal volume models has no hf/Qwen3-4B/{REVISION[:8]}/config.json")
    raw, why = fragment_preflight._fetch_config(_args("/data/models/q"))
    assert raw is None and why == "--model /data/models/q is not a local directory with config.json on this host"


def test_the_config_is_fetched_once_per_model_and_revision(monkeypatch):
    """Every launch of a fleet is prepared (the syncer's and each island's): one fetch."""
    calls = []

    def fetch(args, **probes):
        calls.append(args.model_revision)
        return RAW, "hf://x"

    monkeypatch.setattr(fragment_preflight, "_FETCHED", {})
    monkeypatch.setattr(fragment_preflight, "_fetch_config", fetch)
    for _ in range(3):
        assert REAL_REMOTE_HF_CONFIG(_args("Qwen/Qwen3-4B")) == (RAW, "hf://x")
    REAL_REMOTE_HF_CONFIG(_args("Qwen/Qwen3-4B", revision="b" * 40))
    assert calls == [REVISION, "b" * 40]
