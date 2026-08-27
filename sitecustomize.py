"""Process-wide hooks required by explicitly attested Yeto RL recipes."""

from __future__ import annotations

import os
import time


if os.environ.get("YETO_DSV4_EXPERT_CLONE") == "1":
    from yeto.rl.sglang_deepseek_v4_clone import install

    install()

if os.environ.get("YETO_DSV4_CLONE_ONLY_LORA") == "1":
    from yeto.rl.deepseek_v4_clone_lora import install

    install()

if os.environ.get("YETO_DSV4_EXPERT_FULL") == "1":
    from yeto.rl.deepseek_v4_expert_full_runtime import install

    install()


def _install_tms_post_pause_hold() -> None:
    """Hold a Miles trainer actor after TMS pause for crash attribution.

    ``TMS_INIT_ENABLE`` is injected only into offloaded Megatron trainer Ray
    workers.  Requiring it here avoids importing Miles in the controller,
    rollout actors, SGLang workers, or ordinary Yeto commands.  The wrapper is
    deliberately outside Miles' ``with_logs`` decorator: a successful
    ``sleep phase=end`` followed by ``hold phase=start`` proves pause returned,
    while the matching end marker proves the paused process survived the full
    diagnostic window before the Ray method returned.
    """

    raw = os.environ.get("YETO_TMS_POST_PAUSE_IDLE_S")
    if raw is None or os.environ.get("TMS_INIT_ENABLE") != "1":
        return
    seconds = float(raw)
    if not 0 <= seconds <= 300:
        raise RuntimeError("YETO_TMS_POST_PAUSE_IDLE_S must be in [0, 300]")

    from miles.backends.megatron_utils.actor import MegatronTrainRayActor

    original = MegatronTrainRayActor.sleep
    if getattr(original, "_yeto_post_pause_hold", False):
        return

    def sleep_with_post_pause_hold(self):
        result = original(self)
        rank = os.environ.get("RANK", "unknown")
        print(
            "[yeto-tms-post-pause] "
            f"phase=start pid={os.getpid()} rank={rank} seconds={seconds:g}",
            flush=True,
        )
        time.sleep(seconds)
        print(
            "[yeto-tms-post-pause] "
            f"phase=end pid={os.getpid()} rank={rank} seconds={seconds:g}",
            flush=True,
        )
        return result

    sleep_with_post_pause_hold._yeto_post_pause_hold = True
    sleep_with_post_pause_hold.__name__ = original.__name__
    sleep_with_post_pause_hold.__qualname__ = original.__qualname__
    MegatronTrainRayActor.sleep = sleep_with_post_pause_hold


_install_tms_post_pause_hold()
