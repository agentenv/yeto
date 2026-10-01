"""Dr.GRPO constant-denominator pg_loss reducer (change ``rl-algo-grpo-knobs``, design D2).

Vendored from radixark/miles ``9e4260d``
``examples/experimental/DrGRPO/custom_reducer.py``
(git blob ``96390ac33e6f2e896d0ef5ebe9c3a4843a4fc8de``, source SHA256
``2fe93181ce9cb5e6f0041f9505244de6ef1a1697604f5375e51a4af5b81a1f86``;
``tests/test_rl_grpo_knobs_reducer.py`` recomputes both from
``git show 9e4260d:<path>`` in the miles-next checkout).

The only change: the divisor is not the module constant ``DIVISOR = 1000.0``
but ``loss.constant_denominator`` of the AlgorithmSpec, read from
``args.yeto_algo_plugins`` (hash-checked, design D5). ``args`` is the Miles
namespace Megatron keeps as its global args (Miles ``initialize.py``
``set_args(args)``); a missing denominator raises. yeto rejects
``loss.aggregation='constant'`` with ``--calculate-per-token-loss`` and with
CP>1 before any GPU process; the CP assertion of the original is kept.

Selected with ``--custom-pg-loss-reducer-function-path
yeto.rl.algos.reducers.constant_denominator_reducer``; only pg_loss uses it
(clipfrac / KL / entropy keep the default reducer, Miles semantics).
"""

from collections.abc import Callable

from .reward_pipeline import read_plugins

SOURCE_REPOSITORY = "radixark/miles"
SOURCE_COMMIT = "9e4260d"
SOURCE_PATH = "examples/experimental/DrGRPO/custom_reducer.py"
SOURCE_BLOB = "96390ac33e6f2e896d0ef5ebe9c3a4843a4fc8de"
SOURCE_SHA256 = "2fe93181ce9cb5e6f0041f9505244de6ef1a1697604f5375e51a4af5b81a1f86"


def _global_args():
    from megatron.training.global_vars import get_args

    return get_args()


def _cp_world_size() -> int:
    from megatron.core import mpu

    return mpu.get_context_parallel_world_size()


def configured_denominator(args) -> float:
    config = read_plugins(args)
    reducer = config.get("reducer") or {}
    denominator = reducer.get("denominator")
    if denominator is None:
        raise RuntimeError(
            "yeto constant-denominator reducer: args.yeto_algo_plugins has no "
            "reducer.denominator (loss.constant_denominator of the AlgorithmSpec)"
        )
    denominator = float(denominator)
    if not denominator > 0:
        raise RuntimeError(f"yeto constant-denominator reducer: denominator {denominator} <= 0")
    return denominator


def constant_denominator_reducer(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list,
    calculate_per_token_loss: bool = False,
    *,
    args=None,
) -> Callable:
    """Miles pg_loss reducer: sum of masked token losses / D (D from the spec)."""

    assert _cp_world_size() == 1, "This custom reducer only supports cp_size == 1"
    if calculate_per_token_loss:
        # yeto refuses this combination before launch (design D2); the
        # original example degrades to a plain token sum here.
        raise RuntimeError(
            "loss.aggregation='constant' cannot be combined with --calculate-per-token-loss"
        )
    divisor = configured_denominator(args if args is not None else _global_args())

    def sum_of_sample_mean(x):
        return sum(
            [
                (x_i * loss_mask_i).sum() / divisor
                for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
            ]
        )

    return sum_of_sample_mean
