"""G2: on the real ports path the ExecutionProfile is ``miles-lora-partitioned-serial``;
Flash-Next full must still get its declared edges (by model fingerprint), and
"declared AND certified (rollout-only)" stays the only way an edge is kept."""
from types import SimpleNamespace

from yeto.rl.elastic_benchmark.capabilities import attestation_from_dict
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.miles_adapter import config as mc
from yeto.rl.engine.miles_adapter.elastic_hook import _declared_edges, is_flash_next_full
from yeto.rl.profiles import qwen3_8_next as q

from test_rl_fn_ports_recipe import _fn_config

DECL = q.flash_next_elastic_declaration()
SMALL, BIG = list(DECL["configs"])[:2]
REAL = SimpleNamespace(name="miles-lora-partitioned-serial")


def _parsed_from_ports_argv(layers=48):
    """What Miles' parser sees for the fingerprint fields (int-typed like upstream)."""
    argv = list(mc.translate_run_config(_fn_config(layers=layers), AlgorithmSpec()).argv)
    get = lambda f: argv[argv.index(f) + 1]  # noqa: E731
    return SimpleNamespace(model_name=get("--model-name"),
                           custom_model_provider_path=get("--custom-model-provider-path"),
                           num_layers=int(get("--num-layers")), num_experts=int(get("--num-experts")))


def _att(kind):
    return attestation_from_dict({"auto_controller": True, "certified_edges": [
        {"source": SMALL, "target": BIG, "kind": kind},
        {"source": BIG, "target": SMALL, "kind": kind}]})


def test_real_profile_name_recognised_by_fingerprint():
    ma = _parsed_from_ports_argv()
    assert is_flash_next_full(REAL, ma)
    assert not is_flash_next_full(REAL, None)
    assert not is_flash_next_full(REAL, _parsed_from_ports_argv(layers=4))  # 4layer: no declaration
    assert not is_flash_next_full(REAL, SimpleNamespace(model_name="qwen3_5", num_layers=48))
    assert is_flash_next_full(q.flash_next_execution_profile(), None)  # name prefix still works


def test_declared_and_certified_both_required():
    ma = _parsed_from_ports_argv()
    configs = DECL["configs"]
    got = _declared_edges(REAL, configs, _att(q.ELASTIC_EDGE_KIND), miles_args=ma)
    assert got == frozenset({(SMALL, BIG), (BIG, SMALL)})
    assert _declared_edges(REAL, configs, _att("trainer-dp"), miles_args=ma) == frozenset()
    assert _declared_edges(REAL, configs, None, miles_args=ma) == frozenset()
    # a non-FN run keeps attestation-only behaviour
    assert _declared_edges(REAL, configs, _att(q.ELASTIC_EDGE_KIND), miles_args=None) is None
