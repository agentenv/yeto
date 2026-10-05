"""D2 learner wiring: ``--rl-recommend-mode`` / ``--rl-edge-costs-path`` /
``--rl-elastic-window-s`` -> :class:`yeto.rl.engine.elastic.ElasticHook`.

The learner copies the flags onto ``miles_args`` (``yeto_rl_recommend_mode``,
``yeto_rl_edge_costs_path``, ``yeto_rl_elastic_window_s``).  A hook is built
only when the driver observes, the run is elastic (there is a controller) and
the mode is not ``disabled``; otherwise ``None`` and the island is unchanged.

* configs: the controller's resource configs (the run's elastic declaration);
* declared edges: the Flash-Next full profile intersects with
  ``flash_next_elastic_declaration``; other profiles use attestation edges only;
* events: the driver feeds every observe event to ``hook.feed`` (in-memory,
  independent of where the Miles tape lives);
* mode: ``controller.set_recommend_mode`` at startup; ``auto`` without the
  ``auto_controller`` capability is Rejected and falls back to ``recommend``;
* restart: the last journaled ``auto_state`` is restored before the first step.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_WINDOW_S = 300.0


def recommend_mode_of(miles_args: Any) -> str:
    return str(getattr(miles_args, "yeto_rl_recommend_mode", None) or "disabled")


def _declared_edges(profile: Any, configs: Any):
    from yeto.rl.profiles import qwen3_8_next as q

    if not str(getattr(profile, "name", None) or "").startswith(q.PROFILE_NAME_FULL):
        return None
    edges = q.flash_next_elastic_declaration()["declared_edges"]
    return frozenset(e for e in edges if e[0] in configs and e[1] in configs)


def apply_recommend_mode(controller: Any, mode: str) -> str:
    """Set the startup mode; auto refused by the attestation -> recommend."""
    from ..controller import Rejected

    try:
        controller.set_recommend_mode(mode, reason="learner --rl-recommend-mode")
        return mode
    except Rejected as exc:
        if mode != "auto":
            raise
        log.warning("--rl-recommend-mode auto refused (%s); falling back to recommend", exc)
        controller.set_recommend_mode("recommend", reason=f"auto refused at startup: {exc}")
        return "recommend"


def elastic_hook_for(miles_args: Any, *, controller: Any, profile: Any, observe: bool,
                     clock: Any = None) -> Any:
    mode = recommend_mode_of(miles_args)
    if mode == "disabled" or not observe or controller is None:
        return None
    from ..auto import AutoController
    from ..elastic import ElasticHook
    from ..recommend import Recommender

    kw = {"clock": clock} if clock is not None else {}
    window = getattr(miles_args, "yeto_rl_elastic_window_s", None) or DEFAULT_WINDOW_S
    hook = ElasticHook(
        dict(controller.configs), window_s=float(window),
        total_rounds=getattr(miles_args, "num_rollout", None),
        edge_costs_path=getattr(miles_args, "yeto_rl_edge_costs_path", None) or None,
        declared_edges=_declared_edges(profile, controller.configs),
        recommender=Recommender(**kw),
        auto=AutoController(Recommender(**kw), **kw),
    )
    applied = apply_recommend_mode(controller, mode)
    hook._restore(controller)  # restart: journaled auto_state before the first step
    log.info("elastic hook: mode=%s window_s=%s costs=%s", applied, hook.window_s,
             hook.edge_costs_path)
    return hook


RECOMMEND_MODE_CHOICES = ("disabled", "manual", "recommend", "auto")


def check_recommend_flags(args: Any) -> None:
    """Launcher and learner share this check (args from either parser)."""
    mode = getattr(args, "rl_recommend_mode", None) or "disabled"
    window = getattr(args, "rl_elastic_window_s", None)
    costs = getattr(args, "rl_edge_costs_path", None)
    if mode not in RECOMMEND_MODE_CHOICES:
        raise ValueError(f"--rl-recommend-mode must be one of {RECOMMEND_MODE_CHOICES}")
    if window is not None and not window > 0:
        raise ValueError("--rl-elastic-window-s must be positive")
    if mode == "disabled" and window is None and not costs:
        return
    needs = [f for f, on in (("--rl-elastic", getattr(args, "rl_elastic", False)),
                             ("--rl-observe-timeline", getattr(args, "rl_observe_timeline", False)))
             if not on]
    if needs:
        raise ValueError("--rl-recommend-mode / --rl-edge-costs-path / --rl-elastic-window-s need "
                         + ", ".join(needs))


def recommend_flags(args: Any) -> str:
    """Launcher -> learner argv fragment (empty when nothing was given)."""
    import shlex

    out = ""
    mode = getattr(args, "rl_recommend_mode", None)
    if mode and mode != "disabled":
        out += f" --rl-recommend-mode {mode}"
    if getattr(args, "rl_edge_costs_path", None):
        out += f" --rl-edge-costs-path {shlex.quote(str(args.rl_edge_costs_path))}"
    if getattr(args, "rl_elastic_window_s", None) is not None:
        out += f" --rl-elastic-window-s {float(args.rl_elastic_window_s)!r}"
    return out


def apply_recommend_flags(args: Any, miles_args: Any) -> None:
    mode = getattr(args, "rl_recommend_mode", None)
    if mode and mode != "disabled":
        miles_args.yeto_rl_recommend_mode = mode
    if getattr(args, "rl_edge_costs_path", None):
        miles_args.yeto_rl_edge_costs_path = str(args.rl_edge_costs_path)
    if getattr(args, "rl_elastic_window_s", None) is not None:
        miles_args.yeto_rl_elastic_window_s = float(args.rl_elastic_window_s)


def add_recommend_arguments(parser: Any) -> None:
    parser.add_argument("--rl-recommend-mode", choices=RECOMMEND_MODE_CHOICES, default=None,
                        help="ports+--rl-elastic+--rl-observe-timeline: D2 recommend/auto "
                        "mode at startup (default disabled)")
    parser.add_argument("--rl-edge-costs-path", default=None,
                        help="5.7 transition cost table for recommend/auto (missing -> hold)")
    parser.add_argument("--rl-elastic-window-s", type=float, default=None,
                        help=f"load window seconds (default {DEFAULT_WINDOW_S:g})")
