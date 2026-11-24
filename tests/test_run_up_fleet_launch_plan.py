"""The trainer argv an island fleet compiles to (:mod:`yeto.rl.fleet`) is a launch plan
of K synced trainers behind one syncer, each on its own store sub-directory
``trainer<k>`` with the buffer shards of its island's pools (``yeto launch
--dry-run`` of the documented example; no cloud, no Ray)."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401
from yeto.rl import fleet, up
from yeto.rl.adapters.miles import island_entry
from yeto.rl.engine import sample_roles

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "fleet" / "run.two-islands.json"


@pytest.fixture(autouse=True)
def _island_env(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "fleet-p2-test-key")
    monkeypatch.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test-secret")


def _example(tmp_path) -> dict:
    text = EXAMPLE.read_text()
    for placeholder, value in (("<bucket>", "b"), ("<run>", "r"), ("<store.env>", str(tmp_path / "store.env")),
                               ("<sampler-spec.vllm.json>", str(tmp_path / "spec.json")),
                               ("<sampler-spec.sglang.json>", str(tmp_path / "spec.json")),
                               ("<sampler-image>", "registry/sampler@sha256:" + "0" * 64),
                               ("<modal-secret>", "store-secret"), ("<volume>", "weights")):
        text = text.replace(placeholder, value)
    (tmp_path / "store.env").write_text("R2_BUCKET=b\n")
    (tmp_path / "spec.json").write_text(json.dumps({"sampler_id": "x", "sampling": {"n": 8}}))
    path = tmp_path / "run.json"
    path.write_text(text)
    return up.load_config(str(path))


def test_the_compiled_argv_launches_k_trainers_on_their_own_sub_stores(tmp_path):
    from yeto.cli import parse_args
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    cfg = _example(tmp_path)
    plan = fleet.plan_of(cfg)
    args = parse_args(cfg["trainer"]["argv"])
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    args.controller = "local"
    _prepare_rl_args(args)
    assert sample_roles.is_external_pool_group(args) and sample_roles.follows_trainer(args)
    launch = dry_run_plan(args)
    assert "--learners 2 " in launch["role_syncers"][0]["command"]
    trainers = [i for i in launch["island_requests"] if i.get("sample_role") == "trainer"]
    assert len(trainers) == len(plan.islands) == 2
    for k, (request, island) in enumerate(zip(trainers, plan.islands)):
        tail = request["learner_command"].split("-m yeto.rl.adapters.miles.island_entry", 1)[1]
        tokens = [{"$LEARNER_ID": str(k), "$SYNCER_ADDR": "10.0.0.1:29400"}.get(t, t) for t in shlex.split(tail)]
        parsed = island_entry.parse_args(tokens)
        assert str(parsed.rl_sample_store).endswith(f"/{island.substore}"), parsed.rl_sample_store
        shards = parsed.rl_buffer_shards
        assert (shards.split(",") if isinstance(shards, str) else list(shards)) == island.buffer_shards()
        assert parsed.rl_sample_lanes == 1
        assert int(parsed.rl_release_every or 1) == (island.release_every or 1)
