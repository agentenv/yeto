"""``--rl-log-probs-chunk-size``: the trainer's log-prob row chunks for every recipe.

Score centering (and the policy loss) score each response's rows over the whole
vocabulary; one unchunked fp32 ``[rows, vocab / TP]`` block per response ran a
Qwen3-4B score-centering step out of memory on 1 x H100 (segmented samples,
32k packed rows, a 151k vocabulary). Miles chunks the rows with
``--log-probs-chunk-size``, which only the nemotron_h / kimi_k31 recipes could set
(``YETO_MILES_TRAIN_EXTRA_ARGV``). The option is now a yeto flag with an automatic
default; the chunked values and gradients are bitwise those of the unchunked ones.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.config_errors import ConfigError
from yeto.rl.adapters.miles import config as mc
from yeto.rl.engine import run_config as rc
from yeto.rl.engine.algorithm import AlgorithmSpec

FIXTURE = Path(__file__).parent / "fixtures" / "miles_score_centering" / "selected_log_probs.py"
INSTALLED = "miles.backends.training_utils.loss.hub.score_centering.selected_log_probs"


def _fixture_module():
    spec = importlib.util.spec_from_file_location("miles_selected_log_probs_fixture", FIXTURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _installed_module():
    try:
        return importlib.import_module(INSTALLED)
    except Exception as error:  # noqa: BLE001 - any import failure: no Miles here
        pytest.skip(f"Miles does not import here ({type(error).__name__}); the fixture copy covers CPU")


IMPLEMENTATIONS = {"pinned-copy": _fixture_module, "installed": _installed_module}


# ------------------------------------------------------------------ values do not change


@pytest.mark.parametrize("implementation", sorted(IMPLEMENTATIONS))
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("sampling_support", [False, True])
def test_chunked_log_probs_entropy_and_gradients_equal_the_unchunked_ones(implementation, dtype,
                                                                        sampling_support):
    torch = pytest.importorskip("torch")
    module = IMPLEMENTATIONS[implementation]()
    torch.manual_seed(0)
    rows, width, vocab, candidates = 53, 1031, 1000, 5  # padded vocabulary (width > vocab)
    logits = (torch.randn(rows, width) * 3).to(getattr(torch, dtype))
    ids = torch.randint(0, vocab, (rows, 1 + candidates))
    ids[::7, 3:] = -1  # padded candidate slots

    def run(chunk_size):
        x = logits.clone().requires_grad_(True)
        selected, entropy = module.selected_log_probs_and_entropy(
            x, ids, vocab_size=vocab, temperature=0.7, chunk_size=chunk_size, with_entropy=True,
            sampling_support=sampling_support)
        weights = torch.linspace(-1.0, 1.0, selected.numel()).view_as(selected)
        loss = (torch.nan_to_num(selected, neginf=0.0) * weights).sum() + 0.3 * entropy.sum()
        loss.backward()
        return selected.detach(), entropy.detach(), x.grad.detach()

    reference = run(-1)  # Miles' default: one block of all rows
    for chunk_size in (1, 4, 16, rows - 1, rows, rc.LOG_PROBS_CHUNK_AUTO_ROWS):
        for name, expected, got in zip(("log-probs", "entropy", "gradient"), reference, run(chunk_size)):
            assert torch.equal(expected, got), f"{name} differs at chunk size {chunk_size}"


def test_the_fixture_is_the_pinned_miles_file_where_miles_imports():
    module = _installed_module()
    installed = Path(module.__file__).read_text()
    copy = FIXTURE.read_text()
    body = copy[copy.index('"""Selected-token'):]
    assert body == installed


# ------------------------------------------------------------------ option resolution


def test_auto_chunks_above_the_auto_rows_and_is_off_below():
    rows = rc.LOG_PROBS_CHUNK_AUTO_ROWS
    assert rc.resolve_log_probs_chunk_size("auto", 32768) == rows
    assert rc.resolve_log_probs_chunk_size(None, rows + 1) == rows
    # a response of at most that many rows is one chunk either way: the argv stays as it was
    assert rc.resolve_log_probs_chunk_size("auto", rows) == 0
    assert rc.resolve_log_probs_chunk_size("auto", 128) == 0


def test_zero_turns_chunking_off_and_a_number_is_kept():
    assert rc.resolve_log_probs_chunk_size(0, 262144) == 0
    assert rc.resolve_log_probs_chunk_size(1024, 128) == 1024
    assert rc.resolve_log_probs_chunk_size("2048", 32768) == 2048


@pytest.mark.parametrize("bad", ["-1", "on", "1.5", ""])
def test_bad_values_are_refused(bad):
    with pytest.raises(ValueError, match="auto, 0 \\(off\\) or a positive"):
        rc.log_probs_chunk_size_arg(bad)
    with pytest.raises(ConfigError):
        rc.resolve_log_probs_chunk_size(bad, 32768)


def test_train_memory_resolves_the_option():
    def memory(**kw):
        return rc._train_memory(SimpleNamespace(seq_len=32768, **kw)).log_probs_chunk_size

    assert memory() == rc.LOG_PROBS_CHUNK_AUTO_ROWS  # an island args copy without the flag
    assert memory(rl_log_probs_chunk_size="auto") == rc.LOG_PROBS_CHUNK_AUTO_ROWS
    assert memory(rl_log_probs_chunk_size=0) == 0
    assert memory(rl_log_probs_chunk_size=8192) == 8192


# ------------------------------------------------------------------ Miles argv


def _config(chunk: int, **memory):
    from test_rl_miles_adapter_config import make_config

    return dataclasses.replace(make_config(), train_memory=rc.TrainMemoryConfig(
        log_probs_chunk_size=chunk, **memory))


def test_generic_recipe_argv_carries_the_chunk_size():
    argv = list(mc.translate_run_config(_config(4096), AlgorithmSpec()).argv)
    at = argv.index("--log-probs-chunk-size")
    assert argv[at + 1] == "4096" and argv.count("--log-probs-chunk-size") == 1


def test_chunking_off_leaves_the_argv_as_it_was():
    off = mc.translate_run_config(_config(0), AlgorithmSpec()).argv
    assert "--log-probs-chunk-size" not in off
    default = mc.translate_run_config(dataclasses.replace(_config(0), train_memory=rc.TrainMemoryConfig()),
                                      AlgorithmSpec()).argv
    assert off == default


def test_train_memory_argv_appends_the_chunk_flag_last():
    memory = rc.TrainMemoryConfig(max_tokens_per_gpu=32768, log_probs_chunk_size=4096)
    assert mc.train_memory_argv(memory) == (
        "--use-dynamic-batch-size", "--max-tokens-per-gpu", "32768", "--log-probs-chunk-size", "4096")


def test_an_island_environment_value_replaces_the_flag():
    """``YETO_MILES_TRAIN_EXTRA_ARGV=--log-probs-chunk-size N`` (nemotron_h / kimi_k31
    launch kits) wins over --rl-log-probs-chunk-size: one flag in the argv."""
    values = ["train.py", "--max-tokens-per-gpu", "32768", "--log-probs-chunk-size", "4096", "--bf16"]
    assert mc.drop_flag_value(values, "--log-probs-chunk-size") == [
        "train.py", "--max-tokens-per-gpu", "32768", "--bf16"]
    assert mc.drop_flag_value(values, "--absent") == values


# ------------------------------------------------------------------ launcher flag


def _flag_type():
    from yeto.rl.adapters.miles.run_flags import RL_FLAGS

    for flags in RL_FLAGS.values():
        for flag in flags:
            if "--rl-log-probs-chunk-size" in flag.options:
                return flag
    raise AssertionError("--rl-log-probs-chunk-size is not a Miles flag")


def test_the_launcher_flag_parses_auto_zero_and_rows():
    flag = _flag_type()
    parse = flag.kwargs["type"]
    assert flag.kwargs["default"] == "auto"
    assert parse("auto") == "auto" and parse("AUTO") == "auto"
    assert parse("0") == 0 and parse("16384") == 16384
    with pytest.raises(argparse.ArgumentTypeError):
        parse("-2")


@pytest.mark.parametrize("cli, expected", [((), "auto"), (("--rl-log-probs-chunk-size", "1024"), 1024),
                                           (("--rl-log-probs-chunk-size", "0"), 0)])
def test_the_island_receives_the_launcher_value(monkeypatch, tmp_path, cli, expected):
    """The real launcher -> island run script -> the island's own parser. ``auto`` is not
    forwarded: the island resolves it from its --seq-len."""
    from rl_e2e_launch import island_run, learner_from_run

    run = island_run(cli, monkeypatch)
    assert ("--rl-log-probs-chunk-size" in run) == (expected != "auto")
    learner, _env = learner_from_run(run, tmp_path / "home")
    assert learner.rl_log_probs_chunk_size == expected


def test_the_verl_backend_refuses_an_explicit_value_with_its_reason():
    from yeto.rl.adapters.verl import launch_flags

    refused = {option: off for _dest, off, option in launch_flags.IGNORED_ON_VERL_FULL_PARAMETER}
    assert refused["--rl-log-probs-chunk-size"] == "auto"
    assert "--rl-log-probs-chunk-size" in launch_flags.NOT_MAPPED_REASONS
