"""S27 G-Y: YETO_MILES_TRAIN_EXTRA_ARGV allow-list (MFU measurement flags, Nemotron-H only)."""
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
