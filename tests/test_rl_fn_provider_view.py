"""try22 / FN-IMAGE-PLAN option (c): the Flash-Next learner path never calls
megatron.bridge AutoBridge; it registers Miles' qwen4_exp HF aliases and builds
a provider view from the real ``config.json`` (no AutoConfig / AutoBridge mock).

Every probe runs in a subprocess: ``AutoConfig.register`` is process-global, so
the "before the fix" assertion needs a clean transformers, and the Miles
fixture package must not leak into other tests' ``miles`` stubs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("transformers")

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
CONFIGS = {
    "4layer": FIXTURES / "qwen3_8_flash_next_4layer_config.json",  # CharyZeng@d19a6b60
    "full": FIXTURES / "qwen3_8_flash_next_full_config.json",  # Qwen@de4b8e4d
}
MILES_FIXTURE = FIXTURES / "miles_c35702e_hf_config"
# optional real Miles c35702e checkout (git worktree add /tmp/miles-c357 c35702e)
MILES_CHECKOUT = Path(os.environ.get("YETO_TEST_MILES_C35702E", "/tmp/miles-c357"))
TRY22_ERROR = (
    "model type `qwen4_exp` but Transformers does not recognize this architecture"
)


@pytest.fixture(scope="module")
def snapshots(tmp_path_factory):
    out = {}
    for variant, src in CONFIGS.items():
        d = tmp_path_factory.mktemp(f"fn-{variant}")
        (d / "config.json").write_bytes(src.read_bytes())
        out[variant] = d
    return out


def _probe(body: str, *, miles_root: Path | None = MILES_FIXTURE) -> dict:
    path = [str(REPO)] + ([str(miles_root)] if miles_root else [])
    script = (
        "import json, sys\n"
        f"sys.path[:0] = {path!r}\n"
        "sys.modules['megatron'] = None  # any megatron.bridge import fails loudly\n"
        + textwrap.dedent(body)
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600,
        env={**os.environ, "TRANSFORMERS_VERBOSITY": "error"},
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_unfixed_call_fails_like_try22(snapshots):
    """Before the fix: the bare AutoConfig resolution (what AutoBridge.from_hf_pretrained
    does first) rejects qwen4_exp with the try22 message."""
    out = _probe(f"""
        import transformers
        from transformers import AutoConfig
        try:
            AutoConfig.from_pretrained({str(snapshots['4layer'])!r})
            err = None
        except ValueError as exc:
            err = str(exc)
        print(json.dumps({{"version": transformers.__version__, "err": err}}))
    """)
    assert out["err"] is not None and TRY22_ERROR in out["err"], out


def _view_probe(snapshot: Path, model: str) -> str:
    return f"""
        import dataclasses
        from types import SimpleNamespace
        from yeto.rl import flash_next_provider as fnp
        args = SimpleNamespace(model={model!r})
        assert fnp.is_flash_next(args, {str(snapshot)!r})
        assert fnp.is_flash_next(SimpleNamespace(model="/x/local"), {str(snapshot)!r})
        view = fnp.flash_next_provider(args, {str(snapshot)!r})
        print(json.dumps({{"cls": type(view).__name__, **dataclasses.asdict(view)}}))
    """


def _flag(argv, name):
    return argv[argv.index(name) + 1]


@pytest.mark.parametrize(
    "variant, model",
    [("4layer", "CharyZeng/Qwen3.8-Flash-Next-4layer"), ("full", "Qwen/Qwen3.8-Flash-Next")],
)
def test_view_matches_profile_and_miles_model_args(snapshots, variant, model):
    from yeto.rl.profiles import qwen3_8_next as q

    view = _probe(_view_probe(snapshots[variant], model))
    argv = list(q.model_args(variant))  # verbatim scripts/models/qwen3.8-flash-next.py
    assert view["cls"] == "Qwen4ExpModelProvider"
    assert view["num_layers"] == q.NUM_LAYERS[variant] == int(_flag(argv, "--num-layers"))
    assert view["hidden_size"] == int(_flag(argv, "--hidden-size")) == 2560
    assert view["num_attention_heads"] == int(_flag(argv, "--num-attention-heads")) == 24
    assert view["num_query_groups"] == int(_flag(argv, "--num-query-groups")) == 2
    assert view["kv_channels"] == int(_flag(argv, "--kv-channels")) == 256
    assert view["ffn_hidden_size"] == int(_flag(argv, "--ffn-hidden-size")) == 640
    assert view["moe_ffn_hidden_size"] == int(_flag(argv, "--moe-ffn-hidden-size")) == 640
    assert view["moe_shared_expert_intermediate_size"] == int(
        _flag(argv, "--moe-shared-expert-intermediate-size")) == 640
    assert view["num_moe_experts"] == int(_flag(argv, "--num-experts")) == 512
    assert view["moe_router_topk"] == int(_flag(argv, "--moe-router-topk")) == 10
    # every layer MoE: Miles moe_layer_freq(nlayers, first_k_dense_replace=0) = [1]*n
    assert _flag(argv, "--moe-layer-freq") == "[" + ",".join(["1"] * view["num_layers"]) + "]"
    assert view["moe_layer_freq"] == 1
    assert view["vocab_size"] == int(_flag(argv, "--vocab-size")) == 248320
    assert view["layernorm_epsilon"] == float(_flag(argv, "--norm-epsilon")) == 1e-6
    assert view["rotary_base"] == int(_flag(argv, "--rotary-base")) == 10000000
    assert view["rotary_percent"] == float(_flag(argv, "--rotary-percent")) == 0.25
    assert view["normalization"] == _flag(argv, "--normalization") == "RMSNorm"
    assert view["position_embedding_type"] == _flag(argv, "--position-embedding-type") == "rope"
    assert view["share_embeddings_and_output_weights"] is False
    assert "--untie-embeddings-and-output-weights" in argv
    assert view["add_bias_linear"] is False and "--disable-bias-linear" in argv
    assert view["qk_layernorm"] is True and "--qk-layernorm" in argv
    assert view["gated_linear_unit"] is True and "--swiglu" in argv
    assert view["attention_output_gate"] is True and "--attention-output-gate" in argv
    assert "--group-query-attention" in argv and view["num_query_groups"] < view["num_attention_heads"]
    # attention types: GDN x3 + full attention every 4th layer (Miles qwen3_8_next spec)
    assert view["experimental_attention_variant"] == "gated_delta_net"
    assert view["multi_latent_attention"] is False
    n = view["num_layers"]
    assert view["layer_types"] == [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(n)
    ]
    assert view["max_position_embeddings"] == view["seq_length"] == 262144


def test_view_matches_fp_fn_provider(snapshots):
    """The attestation fingerprint provider (fp_fn.fn_provider) and the learner's
    view agree on every field the former sets."""
    sys.path.insert(0, str(REPO / "tests" / "multinode_gpu"))
    try:
        import fp_fn
    finally:
        sys.path.pop(0)
    for variant, model, layers in (
        ("4layer", "CharyZeng/Qwen3.8-Flash-Next-4layer", 4),
        ("full", "Qwen/Qwen3.8-Flash-Next", 48),
    ):
        view = _probe(_view_probe(snapshots[variant], model))
        ref = vars(fp_fn.fn_provider(layers))
        for name, value in ref.items():
            if name == "activation_func":
                continue
            assert view[name] == value, (variant, name, view[name], value)
        assert view["cls"] == type(fp_fn.fn_provider(layers)).__name__


def test_name_and_shape_variant_must_agree(snapshots):
    body = f"""
        from types import SimpleNamespace
        from yeto.rl import flash_next_provider as fnp
        try:
            fnp.flash_next_provider(SimpleNamespace(model="Qwen/Qwen3.8-Flash-Next"),
                                    {str(snapshots['4layer'])!r})
            err = None
        except ValueError as exc:
            err = str(exc)
        print(json.dumps({{"err": err}}))
    """
    assert "does not match" in (_probe(body)["err"] or "")


def test_missing_miles_alias_registration_is_explicit(snapshots):
    body = f"""
        from types import SimpleNamespace
        sys.modules['miles'] = None
        from yeto.rl import flash_next_provider as fnp
        try:
            fnp.flash_next_provider(SimpleNamespace(model="CharyZeng/Qwen3.8-Flash-Next-4layer"),
                                    {str(snapshots['4layer'])!r})
            err = None
        except RuntimeError as exc:
            err = str(exc)
        print(json.dumps({{"err": err}}))
    """
    err = _probe(body, miles_root=None)["err"] or ""
    assert "register_hf_config_aliases" in err and "AutoBridge" in err


def test_non_flash_next_models_are_not_routed(tmp_path):
    from types import SimpleNamespace

    from yeto.rl import flash_next_provider as fnp

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))
    assert not fnp.is_flash_next(SimpleNamespace(model="Qwen/Qwen3.5-35B-A3B"), tmp_path)
    assert not fnp.is_flash_next(SimpleNamespace(model="Qwen/Qwen3-0.6B"), tmp_path / "nope")


def test_run_miles_fn8s_reaches_ports_argv_without_megatron(snapshots):
    """run_miles for the fnrun.sh fn8s learner args: no megatron import, real
    AutoConfig + Miles aliases, resolved recipe qwen3_8_next and the native
    Flash-Next argv (stopped at parse_miles_args, which needs the Miles parser)."""
    body = f"""
        import tempfile
        from pathlib import Path
        sys.path.insert(0, {str(REPO / 'tests')!r})
        sys.path.insert(0, {str(REPO / 'tests' / 'multinode_gpu')!r})
        import _pytest.monkeypatch as _m
        from rl_e2e_launch import island_run, learner_from_run
        from fp_fn import fnrun_cli
        from yeto.rl import learner
        from yeto.rl.engine import run_config
        from yeto.rl.engine.miles_adapter import config as mac
        from yeto.rl.engine.miles_adapter import state as mas

        mp = _m.MonkeyPatch()
        run = island_run(tuple(fnrun_cli("fn8s")), mp)
        args, _env = learner_from_run(run, Path(tempfile.mkdtemp()) / "home")
        mp.setattr(run_config, "_resolve_ref_load",
                   lambda a, model_path: a.megatron_ref_load or str(model_path))
        captured = {{}}
        real_resolve = run_config.resolve_rl_run_config

        def resolve(*a, **kw):
            rc = real_resolve(*a, **kw)
            captured["provider"] = type(kw["provider"]).__name__
            captured["targets"] = list(kw["target_modules"])
            captured["recipe"] = rc.model_recipe.name
            return rc

        class Stop(Exception):
            pass

        def parse(launch):
            captured["argv"] = list(launch.argv)
            raise Stop

        mp.setattr(run_config, "resolve_rl_run_config", resolve)
        mp.setattr(mac, "parse_miles_args", parse)
        mp.setattr(mas, "require_run_plugin", lambda: None)  # needs Miles' Ray trainer group
        try:
            learner.run_miles(args, model_path={str(snapshots['4layer'])!r},
                              prompt_path="/root/yeto-rl/prompts.jsonl",
                              yeto_policy_sync=True)
        except ValueError as exc:
            captured["sync_err"] = str(exc)
        captured["sync_reached_resolve"] = "recipe" in captured
        try:
            learner.run_miles(args, model_path={str(snapshots['4layer'])!r},
                              prompt_path="/root/yeto-rl/prompts.jsonl",
                              yeto_policy_sync=False)
        except Stop:
            pass
        captured["megatron_loaded"] = any(
            m == "megatron" or m.startswith("megatron.")
            for m, v in sys.modules.items() if v is not None)
        captured["model"] = args.model
        print(json.dumps(captured))
    """
    out = _probe(body)
    assert "needs --rl-single-island-no-sync" in out.get("sync_err", ""), out
    assert out["sync_reached_resolve"] is False
    assert out["megatron_loaded"] is False
    assert out["provider"] == "Qwen4ExpModelProvider"
    assert out["recipe"] == "qwen3_8_next"
    argv = out["argv"]
    assert _flag(argv, "--model-name") == "qwen4_exp"
    assert _flag(argv, "--num-layers") == "4" and _flag(argv, "--num-experts") == "512"
    assert _flag(argv, "--megatron-to-hf-mode") == "raw"
    assert "--custom-model-provider-path" in argv


@pytest.mark.skipif(not (MILES_CHECKOUT / "miles/utils/hf_utils/config.py").is_file(),
                    reason="no Miles c35702e checkout")
def test_fixture_matches_real_miles_c35702e(snapshots):
    body = """
        from miles.utils.hf_utils import config as c
        print(json.dumps([[a.model_type, a.base_module, a.base_class, a.compat_class_name,
                           list(a.auto_model_classes), a.override_hf_native]
                          for a in c._CONFIG_ALIASES if a.model_type.startswith("qwen4_exp")]))
    """
    assert _probe(body, miles_root=MILES_CHECKOUT) == _probe(body)
    view = _probe(_view_probe(snapshots["4layer"], "CharyZeng/Qwen3.8-Flash-Next-4layer"),
                  miles_root=MILES_CHECKOUT)
    assert view["num_layers"] == 4 and view["num_moe_experts"] == 512
