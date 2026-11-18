"""YETO_MILES_TRAIN_EXTRA_ARGV allow-list (MFU measurement flags, Nemotron-H only)."""
import pytest

from yeto.rl.adapters.miles.config import MilesConfigError, train_extra_argv


def test_empty_is_noop_for_any_recipe():
    assert train_extra_argv("generic", "") == ()
    assert train_extra_argv("nemotron_h", "  ") == ()


def test_allowed_flags_pass_through_in_order():
    raw = ("--recompute-granularity selective --recompute-modules moe layernorm "
           "--moe-token-dispatcher-type alltoall --moe-permute-fusion")
    assert train_extra_argv("nemotron_h", raw) == tuple(raw.split())


@pytest.mark.parametrize("raw", ["--tensor-model-parallel-size 1", "selective",
                                 "--moe-permute-fusion --expert-model-parallel-size 4"])
def test_layout_or_unknown_flags_are_refused(raw):
    with pytest.raises(MilesConfigError):
        train_extra_argv("nemotron_h", raw)


def test_other_recipes_refuse_a_set_value():
    with pytest.raises(MilesConfigError):
        train_extra_argv("generic", "--moe-permute-fusion")


def test_commas_split_like_spaces():
    assert train_extra_argv("nemotron_h", "--recompute-modules,moe,layernorm") == (
        "--recompute-modules", "moe", "layernorm")


def test_log_prob_chunking_is_allowed():
    # the colocated agentic run ran out of memory in the full fp32 log-prob block
    assert train_extra_argv("nemotron_h", "--log-probs-chunk-size,4096") == ("--log-probs-chunk-size", "4096")


def test_kimi_k31_takes_chunking_and_profiler_flags():
    raw = "--log-probs-chunk-size,4096 --use-pytorch-profiler --profile-step-start 2 --profile-step-end 3"
    assert train_extra_argv("kimi_k31", raw) == tuple(raw.replace(",", " ").split())


def test_kimi_k31_refuses_layout_flags():
    with pytest.raises(MilesConfigError):
        train_extra_argv("kimi_k31", "--context-parallel-size 2")


def test_kimi_k31_takes_the_first_stage_grad_hook():
    from yeto.rl.adapters.miles.models.kimi_k31 import first_stage_grad

    path = f"{first_stage_grad.__name__}.before_train_step"
    raw = f"--log-probs-chunk-size 16384 --custom-megatron-before-train-step-hook-path {path}"
    assert train_extra_argv("kimi_k31", raw)[-1] == path


def test_kimi_k31_takes_an_uneven_pipeline_split():
    raw = "--decoder-first-pipeline-num-layers 36 --decoder-last-pipeline-num-layers 27"
    assert train_extra_argv("kimi_k31", raw) == tuple(raw.split())


def test_an_extra_argv_split_is_checked_like_a_run_config_split():
    from yeto.rl.adapters.miles.config import check_pipeline_split

    check_pipeline_split(93, 3, None, ("--decoder-first-pipeline-num-layers", "36",
                                       "--decoder-last-pipeline-num-layers", "27"))
    with pytest.raises(MilesConfigError):
        check_pipeline_split(93, 3, None, ("--decoder-first-pipeline-num-layers", "36",
                                           "--decoder-last-pipeline-num-layers", "60"))


# k31-mfu16-x3 (launch.sh PROF_STACK=1): the island refused this value after a two-node start
X3_EXTRA = ("--use-pytorch-profiler --profile-target train_overall --profile-step-start 3 "
            "--profile-step-end 4 --profile-ranks 0 8 --tensorboard-dir /yeto-tape/traces/k31-mfu16-x3 "
            "--pytorch-profiler-collect-callstack")


def test_kimi_k31_takes_the_profiler_record_flags():
    assert train_extra_argv("kimi_k31", X3_EXTRA) == tuple(X3_EXTRA.split())
    assert train_extra_argv("kimi_k31", "--use-pytorch-profiler --pytorch-profiler-collect-shapes")[-1] == (
        "--pytorch-profiler-collect-shapes")


def test_the_launcher_applies_the_island_allow_list_to_the_modal_env():
    from yeto.rl.adapters.miles.config import TRAIN_EXTRA_ENV
    from yeto.rl.adapters.miles.run_config_rules import check_train_extra_env

    pairs = ["OMP_NUM_THREADS=8", f"{TRAIN_EXTRA_ENV}={X3_EXTRA}"]
    check_train_extra_env(recipe="kimi_k31", env_pairs=pairs)
    check_train_extra_env(recipe="kimi_k31", env_pairs=None)
    check_train_extra_env(recipe="generic", env_pairs=["OMP_NUM_THREADS=8", f"{TRAIN_EXTRA_ENV}="])
    with pytest.raises(MilesConfigError, match="flags not allowed.*--expert-model-parallel-size"):
        check_train_extra_env(recipe="kimi_k31",
                              env_pairs=[f"{TRAIN_EXTRA_ENV}=--use-pytorch-profiler --expert-model-parallel-size 8"])
    with pytest.raises(MilesConfigError, match="only for"):
        check_train_extra_env(recipe=None, env_pairs=[f"{TRAIN_EXTRA_ENV}=--moe-permute-fusion"])
