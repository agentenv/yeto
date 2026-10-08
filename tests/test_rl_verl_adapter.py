"""rl-verl-backend group 1 (CPU): registry, config rules (1.3), names (1.5),
criteria (1.6), publication (1.10), bindings, identity, launcher wiring.

No verl / vLLM / Ray here: modules that need them (``trainer``, ``verl_main``,
``ports_impl`` worker functions) are exercised on the GPU (V1/V2)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import pytest
import torch

from yeto.rl.adapters.verl import config as vconf
from yeto.rl.adapters.verl import identity as verl_identity
from yeto.rl.adapters.verl import param_names as pn
from yeto.rl.adapters.verl import publish as pub
from yeto.rl.adapters.verl.pins import VERL_COMMIT, VERL_IMAGE
from yeto.rl.engine import backends, mismatch_criteria as mc

S16_DUMPS = Path("/home/michael/work/s1-runs/s16-verl-mismatch-20261008/s16-verl-mismatch-20261008")


def _cfg(**kw):
    base = dict(model_path="/m", train_file="/t", val_file="/v", out_dir="/o")
    base.update(kw)
    return vconf.VerlRunConfig(**base)


# ---------------------------------------------------------------- registry / roles
def test_verl_is_registered_with_the_roles_neutral_code_uses():
    entry = backends.get("verl")
    for role in ("rollout_meta", "identity", "binding", "run_config_rules", "launch_flags", "entry",
                 "elastic_hook", "image"):
        assert backends.module(role, "verl").__name__.startswith("yeto.rl.adapters.verl.")
    assert "verl" in backends.registered() and entry.package == "yeto.rl.adapters.verl"
    with pytest.raises(backends.UnknownBackend, match="没有提供"):
        backends.module("harness_glue", "verl")


def test_launcher_defaults_and_accepts_the_modal_built_verl_image():
    from yeto import launcher

    args = argparse.Namespace(training_mode="rl", rl_backend="verl", rl_image=None)
    launcher.resolve_default_rl_image(args)
    assert args.rl_image == VERL_IMAGE == f"verl-build:{VERL_COMMIT}"
    assert launcher._rl_backend_image_ok(args)
    args.rl_image = "verl-build:" + "b" * 40  # another commit is refused
    assert not launcher._rl_backend_image_ok(args)
    flags = launcher._rl_backend_module(args, "launch_flags")
    assert flags.ISLAND_ENTRY_MODULE == "yeto.rl.adapters.verl.island_entry"
    assert flags.NEEDS_MILES_SOURCE is False
    miles = argparse.Namespace(training_mode="rl", rl_backend="miles", rl_image=None)
    assert launcher._rl_backend_module(miles, "launch_flags").ISLAND_ENTRY_MODULE == \
        "yeto.rl.adapters.miles.island_entry"


def test_modal_runner_passes_engine_built_image_refs_through():
    from yeto.modal_runner import ENGINE_BUILT_IMAGE_RE, image_ref_from_rl_image

    assert image_ref_from_rl_image(VERL_IMAGE) == VERL_IMAGE
    assert ENGINE_BUILT_IMAGE_RE.fullmatch(VERL_IMAGE)["backend"] == "verl"
    with pytest.raises(ValueError):
        image_ref_from_rl_image("verl-build:latest")


def test_verl_run_config_rules_refuse_multi_gpu_and_megatron_options():
    rules = backends.module("run_config_rules", "verl")
    rules.check_trainer_parallel(1, 1, 1)
    for bad in ((2, 1, 1), (1, 2, 1)):
        with pytest.raises(ValueError):
            rules.check_trainer_parallel(*bad)
    with pytest.raises(ValueError):
        rules.resolve_ref_load(argparse.Namespace(megatron_ref_load="/x"), "/m")
    with pytest.raises(ValueError):
        backends.module("launch_flags", "verl").cross_node_flags(True, False)


# ---------------------------------------------------------------- 1.3 config rules
@pytest.mark.parametrize("kw,match", [
    (dict(lora_rank=32, lora_alpha=16), "lora_alpha"),
    (dict(engine_commit="0" * 40), "commit"),
    (dict(bypass_mode=True, criteria_enabled=True), "旁路"),
    (dict(full_determinism=True, enforce_eager=True, mode="train"), "full_determinism"),
    (dict(full_determinism=True, enforce_eager=False, mode="diagnose"), "enforce_eager"),
    (dict(correction="mis"), "不支持此修正"),
    (dict(correction="opsm"), "不支持此修正"),
    (dict(correction="tis", tis_lower=0.5), "下界"),
    (dict(correction="icepop"), "icepop"),
    (dict(top_p=0.9), "top_p"),
    (dict(top_k=50), "top_p/top_k"),
    (dict(temperature=0.7), "温度"),
    (dict(n_gpus=2), "1 张卡"),
])
def test_config_rules_refuse(kw, match):
    with pytest.raises(vconf.VerlConfigError, match=match):
        vconf.validate(_cfg(**kw))


@pytest.mark.parametrize("kw", [
    dict(),  # defaults: T=1, top_p=1, top_k=-1, TIS upper 2.0 lower 0
    dict(correction="none"),
    dict(bypass_mode=True, criteria_enabled=False),
    dict(mode="diagnose", full_determinism=True, enforce_eager=True),
    dict(top_p=0.9, allow_sampling_shift=True),
])
def test_config_rules_accept(kw):
    vconf.validate(_cfg(**kw))


def test_defaults_are_the_decided_sampling_and_lora():
    cfg = _cfg()
    assert (cfg.temperature, cfg.top_p, cfg.top_k) == (1.0, 1.0, -1)
    assert cfg.lora_alpha == cfg.lora_rank and cfg.correction == "tis" and cfg.tis_lower == 0.0
    o = vconf.build_overrides(cfg)
    assert "trainer.v1.trainer_mode=yeto_sync" in o
    assert "actor_rollout_ref.rollout.calculate_log_probs=True" in o
    assert "algorithm.rollout_correction.bypass_mode=False" in o
    assert len(o) == len(set(k.split("=")[0] for k in o))  # one value per key


def test_chat_template_kwargs_are_added_as_new_keys():
    o = vconf.build_overrides(_cfg(chat_template_kwargs={"enable_thinking": False}))
    assert "+data.apply_chat_template_kwargs.enable_thinking=False" in o


# ---------------------------------------------------------------- 1.5 names
def test_qwen3_names_map_both_ways_and_vllm_packing_unpacks():
    names = pn.expected_qwen3_names(28)
    assert len(names) == 28 * 7 * 2 == 392
    for name in names:
        train = pn.canonical_to_train(name)
        assert ".lora_A.default.weight" in train or ".lora_B.default.weight" in train
        assert pn.train_to_canonical(train) == name
        assert pn.train_to_canonical(name) == name  # get_peft_model_state_dict form
    assert pn.vllm_to_canonical("model.layers.3.self_attn.qkv_proj", "A", 1) == \
        "base_model.model.model.layers.3.self_attn.k_proj.lora_A.weight"
    assert pn.vllm_to_canonical("model.layers.3.mlp.gate_up_proj", "B", 1) == \
        "base_model.model.model.layers.3.mlp.up_proj.lora_B.weight"
    assert pn.vllm_to_canonical("model.layers.0.mlp.down_proj", "A") == \
        "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"
    with pytest.raises(pn.ParamNameError):
        pn.vllm_to_canonical("model.layers.3.self_attn.qkv_proj", "A")
    with pytest.raises(pn.ParamNameError):
        pn.train_to_canonical("base_model.model.model.layers.0.mlp.up_proj.lora_A.other.weight")


def test_param_map_hash_is_stable_and_feeds_the_identity():
    ident = verl_identity.backend_identity()
    assert ident.engine == "verl" and ident.engine_commit == VERL_COMMIT
    assert ident.param_map_sha256 == verl_identity.PARAM_MAP_SHA256
    # the hash is pinned: changing the map is a deliberate identity change
    from yeto.rl.engine.backend_identity import param_map_sha256

    assert param_map_sha256(dict(pn.PARAM_MAP)) == verl_identity.PARAM_MAP_SHA256
    from yeto.rl.adapters.miles import identity as miles_identity

    assert ident.sha256() != miles_identity.backend_identity("ports").sha256()


def test_miles_and_verl_session_contracts_differ_for_the_same_layout():
    from yeto.rl.adapters.miles import identity as miles_identity
    from yeto.rl.engine.backend_identity import session_contract_hash

    fp = bytes(range(32))
    a = session_contract_hash(fp, miles_identity.backend_identity("ports").sha256())
    b = session_contract_hash(fp, verl_identity.backend_identity().sha256())
    c = session_contract_hash(fp, verl_identity.backend_identity().sha256())
    assert a != b and b == c


# ---------------------------------------------------------------- 1.6 criteria
def test_criteria_match_hand_computed_values():
    train = torch.tensor([[-1.0, -2.0, 0.0], [-0.5, -0.5, -0.5]])
    infer = torch.tensor([[-1.1, -1.0, 9.0], [-0.5, -0.4, -0.5]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    out = mc.compute(train, infer, mask)
    d0 = torch.tensor([0.1, -1.0])
    d1 = torch.tensor([0.0, -0.1, 0.0])
    want_abs = (d0.abs().mean() + d1.abs().mean()) / 2
    assert math.isclose(out["abs_diff"], float(want_abs), rel_tol=1e-6)
    want_signed = float((d0.sum() + d1.sum()) / 5)
    assert math.isclose(out["signed_mean"], want_signed, rel_tol=1e-6)
    k3 = lambda x: math.exp(x) - 1 - x  # noqa: E731
    want_k3 = ((k3(0.1) + k3(-1.0)) / 2 + (k3(0.0) + k3(-0.1) + k3(0.0)) / 3) / 2
    assert math.isclose(out["k3"], want_k3, rel_tol=1e-5)
    assert out["tis_clipfrac"] == 0.0  # exp(0.1) < 2
    assert out["n_tokens"] == 5 and out["nonfinite_tokens"] == 0


def test_signed_criterion_alarms_on_one_sided_shift_k3_does_not():
    torch.manual_seed(0)
    infer = -torch.rand(16, 64)
    mask = torch.ones(16, 64)
    shifted = mc.compute(infer - 0.02, infer, mask)  # one-sided shift, |diff| under the abs limit
    limits = mc.thresholds_for(("verl", "fsdp2", "vllm-0.29.0", "H100"))
    alarms = mc.judge(shifted, limits)
    assert alarms == ["signed_mean"]
    assert mc.judge(mc.compute(infer, infer, mask), limits) == []
    with pytest.raises(mc.Uncalibrated, match="未标定"):
        mc.thresholds_for(("verl", "fsdp2", "vllm-0.29.0", "Ascend910B"))


def _s16(arm: str, step: int):
    path = S16_DUMPS / arm / "dump" / f"step{step:03d}.pt"
    if not path.is_file():
        pytest.skip(f"S16 raw data not on this machine: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.mark.parametrize("arm,step,abs_diff,k3,signed", [
    ("A1_base_lr0", 1, 0.0176, 0.00080, -0.0009),
    ("A1_base_lr0", 2, 0.0172, 0.00078, -0.0009),
    ("D_topp09_lr0", 1, 0.0354, 0.00165, -0.032),
    ("D_topp09_lr0", 2, 0.0355, 0.00167, -0.032),
])
def test_s16_raw_data_reproduces_the_published_table(arm, step, abs_diff, k3, signed):
    """1.6 acceptance: VERL-MODAL-CHECK-S16.md 7.2 rows A1 and D; D alarms on the signed criterion."""
    rec = _s16(arm, step)
    out = mc.compute(rec["old_log_probs"], rec["rollout_log_probs"], rec["response_mask"])
    assert abs(out["abs_diff"] - abs_diff) < 0.00051
    assert abs(out["k3"] - k3) < 0.000051
    assert abs(out["signed_mean"] - signed) < (0.0006 if arm.startswith("A") else 0.0025)
    alarms = mc.judge(out, mc.thresholds_for(("verl", "fsdp2", "vllm-0.29.0", "H100")))
    assert ("signed_mean" in alarms) == arm.startswith("D")
    # k3 / tis_clipfrac cannot see the top_p shift (S16 7.3 point 3); abs_diff doubles past 0.03
    assert alarms == ([] if arm.startswith("A") else ["abs_diff", "signed_mean"])


# ---------------------------------------------------------------- 1.10 publication
def _lora():
    torch.manual_seed(1)
    return {n: torch.randn(4, 8) for n in pn.expected_qwen3_names(1)}


def test_checksum_compare_verified_mismatch_and_unverifiable():
    tensors = _lora()
    sent = pub.lora_checksum(tensors)
    same = pub.lora_checksum({k: v.clone() for k, v in tensors.items()})
    assert pub.compare(3, sent, same).status == pub.VERIFIED
    bad = dict(tensors)
    first = sorted(bad)[0]
    bad[first] = bad[first] + 1.0
    check = pub.compare(3, sent, pub.lora_checksum(bad))
    assert check.status == pub.MISMATCH and check.mismatched == (first,)
    assert pub.compare(3, sent, None).status == pub.UNVERIFIABLE
    assert pub.compare(3, sent, {"error": "boom"}).detail == "boom"
    assert check.to_event()["level"] == "inference-received"


def test_checksum_is_over_bf16_values():
    t = {"base_model.model.x.lora_A.weight": torch.tensor([[1.0, 1.0 + 2 ** -12]])}
    u = {"base_model.model.x.lora_A.weight": torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16)}
    assert pub.lora_checksum(t)["total"] == pub.lora_checksum(u)["total"]


def test_disk_transport_deletes_old_versions_after_a_successful_load(tmp_path):
    def save(tensors, path):
        torch.save(dict(tensors), path / "adapter.pt")

    disk = pub.DiskTransport(tmp_path, save)
    loaded = []
    for version in (1, 2):
        disk.stage(version, _lora())
        assert disk.commit(version, lambda p: loaded.append(p.name) or True)
    assert loaded == ["v00000001", "v00000002"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v00000002"]
    disk.stage(3, _lora())
    assert not disk.commit(3, lambda p: False)  # failed load keeps the old one
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v00000002", "v00000003"]
    assert pub.transport_for("memory") == "memory"
    with pytest.raises(ValueError):
        pub.transport_for("nccl")


def test_vllm_readback_maps_a_packed_lora_model_to_canonical_names(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from yeto.rl.adapters.verl import vllm_readback as rb

    torch.manual_seed(2)
    q, k, v = (torch.randn(4, 8) for _ in range(3))
    bq, bk, bv = (torch.randn(8, 4) for _ in range(3))
    lora_model = SimpleNamespace(loras={
        "model.layers.0.self_attn.qkv_proj": SimpleNamespace(lora_a=[q, k, v], lora_b=[bq, bk, bv]),
        "model.layers.0.self_attn.o_proj": SimpleNamespace(lora_a=q.t().contiguous(), lora_b=bq),
    })
    manager = SimpleNamespace(_adapter_manager=SimpleNamespace(get_adapter=lambda i: lora_model))
    worker = SimpleNamespace(model_runner=SimpleNamespace(lora_manager=manager))
    monkeypatch.setenv(rb.READBACK_ENV, str(tmp_path))
    (tmp_path / "expected_version").write_text("5")
    pre = "base_model.model.model.layers.0.self_attn."
    shapes = {pre + "o_proj.lora_A.weight": [4, 8]}
    (tmp_path / "expected_shapes.json").write_text(json.dumps(shapes))
    rb.after_add_lora(worker, 123)
    out = json.loads((tmp_path / "readback-v5.json").read_text())
    want = pub.lora_checksum({pre + "q_proj.lora_A.weight": q, pre + "k_proj.lora_A.weight": k,
                              pre + "v_proj.lora_A.weight": v, pre + "q_proj.lora_B.weight": bq,
                              pre + "k_proj.lora_B.weight": bk, pre + "v_proj.lora_B.weight": bv,
                              pre + "o_proj.lora_A.weight": q, pre + "o_proj.lora_B.weight": bq})
    assert out["total"] == want["total"] and json.loads(out["note"])["transposed"] == 1
    # a broken manager is reported, never raised
    rb.after_add_lora(SimpleNamespace(), 123)
    assert "error" in json.loads((tmp_path / "readback-v5.json").read_text())


def test_patch_verl_inserts_the_hook_once(tmp_path):
    from yeto.rl.adapters.verl import patch_verl

    target = tmp_path / patch_verl.TARGET
    target.parent.mkdir(parents=True)
    target.write_text("def f(self):\n" + patch_verl.ANCHOR + "            return 1\n")
    patch_verl.apply(tmp_path)
    patch_verl.apply(tmp_path)  # idempotent
    text = target.read_text()
    assert text.count("_yeto_after_add_lora(self, VLLM_LORA_INT_ID)") == 1
    target.write_text("nothing here\n")
    with pytest.raises(SystemExit):
        patch_verl.apply(tmp_path)


# ---------------------------------------------------------------- binding / image / entry
def test_three_way_binding_check_passes_and_reports_skips():
    out = backends.module("binding", "verl").check_bindings(verl_available=False)
    assert all(row["ok"] for row in out["config"].values())
    assert out["call"]["equal"] and out["call"]["via_verl"] == 1.0
    assert "skipped" in out["identity"]["verl_adv_estimator_grpo"]
    assert len(out["identity"]["yeto_reward"]["source_sha256"]) == 64


def test_unmapped_reward_and_algorithm_fields_are_refused():
    from yeto.rl.adapters.verl import island_entry, reward_fn

    with pytest.raises(reward_fn.UnmappedReward):
        reward_fn.function_for("my_reward:score")
    golden = json.loads(Path("tests/golden/decoupling/grpo_tis.json").read_text())["algorithm_spec"]
    assert island_entry.algorithm_options(golden) == {"correction": "tis", "tis_upper": 2.0,
                                                      "tis_lower": 0.0}
    assert island_entry.algorithm_options(None)["correction"] == "none"


def test_image_recipe_is_pinned():
    from yeto.rl.adapters.verl import image

    cmds = image.build_commands()
    assert VERL_COMMIT in cmds[0] and "uv sync --frozen --all-packages --extra vllm --extra fsdp" in cmds[1]
    with pytest.raises(ValueError):
        image.build_commands("c" * 40)


def test_verl_capabilities_accept_default_and_tis_specs_only():
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.capabilities import CapabilityMismatch

    caps = backends.module("entry", "verl").verl_capabilities("sha256:" + "0" * 64)
    golden = json.loads(Path("tests/golden/decoupling/grpo_tis.json").read_text())["algorithm_spec"]
    for spec in (AlgorithmSpec(), AlgorithmSpec.from_dict(golden)):
        caps.check(layout="lora", placement="colocated", execution_mode="colocated-serial", algorithm=spec)
    with pytest.raises(CapabilityMismatch):
        caps.check(layout="lora", placement="fixed-partition", execution_mode="colocated-serial",
                   algorithm=AlgorithmSpec())
