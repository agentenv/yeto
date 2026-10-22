"""S27 P1: the RL outer-merge settings the presets fixed are explicit options. No Ray.

Each ``--rl-sync-*`` option left at its default keeps the preset's value, so
the syncer and island commands stay byte-for-byte the previous ones; a set
option reaches the syncer command (and, for the merge mode, the island and
the exporter's fragment layout).
"""

from __future__ import annotations

import shlex

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401

TWO_ISLANDS = "modal:2xh100,modal:2xh100"
THREE_ISLANDS = "modal:2xh100,modal:2xh100,modal:2xh100"
DECOUPLED = ("--rl-sync-preset", "decoupled", "--fragments", "4", "--pipeline", "2",
             "--local-rl-rounds-per-sync", "3")
FULL_PARAMETER = ("--rl-sync-preset", "full-parameter", "--rl-placement", "fixed-partition",
                  "--rl-rollout-gpus", "1", "--rl-parameter-layout-sha256", "e" * 64,
                  "--fragments", "4")
SYNCER_FLAGS = ("--quorum", "--grace-ms", "--max-base-lag", "--learner-weight",
                "--outer-lr", "--outer-momentum")


def _plan(extra=(), gpu=TWO_ISLANDS):
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _launcher_args("ports", tuple(extra), gpu=gpu)
    args.controller = "local"
    _prepare_rl_args(args)
    return args, dry_run_plan(args)


def _syncer_flags(plan) -> dict[str, str]:
    [actor] = plan["role_syncers"]
    tokens = shlex.split(actor["command"])
    return {flag: tokens[tokens.index(flag) + 1] for flag in SYNCER_FLAGS}


def _island_commands(plan) -> list[str]:
    return [island["learner_command"] for island in plan["island_requests"]]


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ((), ("2", "0", "0", "equal", "1.0", "0.0")),
        (DECOUPLED, ("2", "0", "0", "equal", "0.7", "0.9")),
        (FULL_PARAMETER, ("2", "0", "0", "equal", "1.0", "0.0")),
    ],
    ids=["strict-avg", "decoupled", "full-parameter"],
)
def test_defaults_keep_each_presets_previous_syncer_values(preset, expected):
    _, plan = _plan(preset)
    assert _syncer_flags(plan) == dict(zip(SYNCER_FLAGS, expected))
    assert all("--rl-sync-matrix-merge" not in command for command in _island_commands(plan))


def test_full_parameter_profile_keeps_the_strict_roster_by_default():
    from yeto import launcher

    args, _ = _plan(FULL_PARAMETER)
    profile = launcher.rl_syncer_profile(args, 2)
    assert (profile.quorum, profile.grace_ms, profile.max_base_lag, profile.learner_weight,
            profile.outer_lr, profile.outer_momentum) == (2, 0, 0, "equal", 1.0, 0.0)


@pytest.mark.parametrize("preset", [(), DECOUPLED], ids=["strict-avg", "decoupled"])
def test_every_option_reaches_the_syncer_command(preset):
    args, plan = _plan(preset + (
        "--rl-sync-quorum", "1", "--rl-sync-grace-ms", "750",
        "--rl-sync-max-base-lag", "2", "--rl-sync-learner-weight", "tokens2-over-steps",
        "--rl-sync-outer-lr", "0.5", "--rl-sync-outer-momentum", "0.8",
    ), gpu=THREE_ISLANDS)
    assert _syncer_flags(plan) == {
        "--quorum": "1", "--grace-ms": "750", "--max-base-lag": "2",
        "--learner-weight": "tokens2-over-steps", "--outer-lr": "0.5", "--outer-momentum": "0.8",
    }
    assert (args.quorum, args.grace_ms, args.outer_lr, args.outer_momentum) == (1, 750, 0.5, 0.8)
    [actor] = plan["role_syncers"]
    assert "--learners 3" in actor["command"]


@pytest.mark.parametrize(
    ("option", "value", "flag"),
    [
        ("--rl-sync-quorum", "2", "--quorum"),
        ("--rl-sync-grace-ms", "500", "--grace-ms"),
        ("--rl-sync-max-base-lag", "1", "--max-base-lag"),
        ("--rl-sync-learner-weight", "tokens2-over-steps", "--learner-weight"),
        ("--rl-sync-outer-lr", "0.9", "--outer-lr"),
        ("--rl-sync-outer-momentum", "0.5", "--outer-momentum"),
    ],
)
def test_one_option_changes_only_its_syncer_flag(option, value, flag):
    _, baseline = _plan(gpu=THREE_ISLANDS)
    _, plan = _plan((option, value), gpu=THREE_ISLANDS)
    expected = dict(_syncer_flags(baseline), **{flag: value})
    assert _syncer_flags(plan) == expected
    assert _island_commands(plan) == _island_commands(baseline)


def test_rda_matrix_merge_reaches_the_decoupled_islands():
    from yeto.rl.adapters.miles import island_entry

    _, plan = _plan(DECOUPLED + ("--rl-sync-matrix-merge", "rda"))
    for command in _island_commands(plan):
        tokens = shlex.split(command.split("-m yeto.rl.adapters.miles.island_entry", 1)[1])
        tokens = [{"$SYNCER_ADDR": "127.0.0.1:29400", "$LEARNER_ID": "1"}.get(t, t)
                  for t in tokens]
        parsed = island_entry.parse_args(tokens)
        assert (parsed.sync_preset, parsed.rl_sync_matrix_merge) == ("decoupled", "rda")


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (FULL_PARAMETER + ("--rl-sync-quorum", "1"), "strict-avg or decoupled"),
        (FULL_PARAMETER + ("--rl-sync-grace-ms", "100"), "strict-avg or decoupled"),
        (FULL_PARAMETER + ("--rl-sync-max-base-lag", "1"), "strict-avg or decoupled"),
        (FULL_PARAMETER + ("--rl-sync-learner-weight", "tokens2-over-steps"),
         "strict-avg or decoupled"),
        (FULL_PARAMETER + ("--rl-sync-outer-lr", "0.7"), "strict-avg or decoupled"),
        (FULL_PARAMETER + ("--rl-sync-outer-momentum", "0.9"), "strict-avg or decoupled"),
        (FULL_PARAMETER + ("--rl-sync-matrix-merge", "rda"), "strict-avg or decoupled"),
        (("--rl-sync-matrix-merge", "rda"), "only applies to --rl-sync-preset decoupled"),
        (("--rl-sync-quorum", "1", "--experimental-rl-sync", "--fragments", "1",
          "--pipeline", "1", "--merge-alpha", "0", "--wire-dtype", "f32"),
         "experimental-rl-sync"),
        (("--rl-sync-quorum", "1", "--rl-island-scheduling", "elastic"), "legacy"),
        (("--rl-sync-quorum", "0"), "between 1 and the 2 islands"),
        (("--rl-sync-quorum", "3"), "between 1 and the 2 islands"),
        (("--rl-sync-grace-ms", "-1"), "grace-ms must be >= 0"),
        (("--rl-sync-max-base-lag", "-1"), "max-base-lag must be >= 0"),
        (("--rl-sync-outer-lr", "0"), "outer-lr must be positive"),
        (("--rl-sync-outer-momentum", "1.0"), r"outer-momentum must be in \[0, 1\)"),
    ],
)
def test_refusals(extra, message, monkeypatch):
    from yeto.launcher import _prepare_rl_args

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")
    args = _launcher_args("ports", extra, gpu=TWO_ISLANDS)
    with pytest.raises(ValueError, match=message):
        _prepare_rl_args(args)


# --- the decoupled fragment layout -------------------------------------------

def _specs():
    from yeto.rl.core import CanonicalTensorSpec

    shapes = {
        "base_model.model.embed_tokens.lora_A.weight": (8, 64),
        "base_model.model.layers.0.q_proj.lora_A.weight": (8, 32),
        "base_model.model.layers.0.q_proj.lora_B.weight": (32, 8),
        "base_model.model.layers.1.v_proj.lora_A.weight": (8, 16),
        "base_model.model.layers.1.v_proj.lora_B.weight": (16, 8),
    }
    return tuple(CanonicalTensorSpec(name, shape, "float32", shape[0] * shape[1])
                 for name, shape in shapes.items())


def test_default_layout_is_the_previous_all_avg_binpack():
    from yeto.fragments import MERGE_AVG
    from yeto.protocol import layout_fingerprint
    from yeto.rl.core import build_rl_fragment_layout

    default = build_rl_fragment_layout(_specs(), 3)
    explicit = build_rl_fragment_layout(_specs(), 3, matrix_merge="avg")
    assert {fragment.merge_mode for fragment in default.fragments} == {MERGE_AVG}
    assert layout_fingerprint(default) == layout_fingerprint(explicit)


def test_rda_layout_keeps_embedding_fragments_averaged():
    from yeto.fragments import MERGE_AVG, MERGE_RDA, is_embedding_name
    from yeto.protocol import layout_fingerprint
    from yeto.rl.core import build_rl_fragment_layout

    avg = build_rl_fragment_layout(_specs(), 3)
    rda = build_rl_fragment_layout(_specs(), 3, matrix_merge="rda")
    assert [f.tensors for f in rda.fragments] == [f.tensors for f in avg.fragments]
    modes = {
        any(is_embedding_name(name) for name, _ in fragment.tensors): fragment.merge_mode
        for fragment in rda.fragments
    }
    assert modes == {True: MERGE_AVG, False: MERGE_RDA}
    assert layout_fingerprint(rda) != layout_fingerprint(avg)


def test_layout_refuses_an_unknown_matrix_merge():
    from yeto.rl.core import build_rl_fragment_layout

    with pytest.raises(ValueError, match="matrix merge"):
        build_rl_fragment_layout(_specs(), 3, matrix_merge="iso")


def test_decoupled_bridge_config_defaults_to_avg():
    import dataclasses

    from yeto.rl.decoupled import DecoupledBridgeConfig

    [field] = [f for f in dataclasses.fields(DecoupledBridgeConfig) if f.name == "matrix_merge"]
    assert field.default == "avg"


def test_exporter_flags_default_to_the_previous_decoupled_contract():
    from yeto.rl.export import parse_args

    base = ["--checkpoint", "c", "--output-dir", "o", "--model", "m",
            "--model-revision", "r", "--lora-r", "8"]
    defaults = parse_args(base)
    assert (defaults.matrix_merge, defaults.outer_lr, defaults.outer_momentum) == ("avg", 0.7, 0.9)
    parsed = parse_args(base + ["--matrix-merge", "rda", "--outer-lr", "0.5",
                                "--outer-momentum", "0.8"])
    assert (parsed.matrix_merge, parsed.outer_lr, parsed.outer_momentum) == ("rda", 0.5, 0.8)
