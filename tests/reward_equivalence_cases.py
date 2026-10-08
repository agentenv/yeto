"""Recorded sample set for the reward / filter equivalence test (decoupling task 3.2).

``python tests/reward_equivalence_cases.py --write`` runs the Miles-facing
reward and filter entry points on a fixed set of fake Miles samples and stores
every observable result (return value, sample status, sample metadata, filter
keep/reason, filter state) in ``tests/golden/rewards/miles_entry_points.json``.
The baseline file was written with the code *before* the neutral refactor
(commit c5d2c090); ``test_rl_neutral_rewards.py`` replays the same cases on the
current code and requires identical output.

CPU only.  Miles is not imported: math grading uses the test-only copy
``tests/vendor/miles_math_utils.py`` injected as
``miles.rollout.rm_hub.math_utils`` (task 3.9), and the filter result type
falls back to a namespace exactly as on a Miles-less host.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "golden" / "rewards" / "miles_entry_points.json"


class Status(enum.Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"
    FAILED = "failed"


class FakeSample:
    Status = Status

    def __init__(self, response=None, label=None, *, status=Status.COMPLETED, metadata=None,
                 index=None, reward=None):
        self.response = response
        self.label = label
        self.status = status
        self.metadata = metadata
        self.index = index
        self.reward = reward

    def snapshot(self) -> dict[str, Any]:
        return {"status": getattr(self.status, "value", self.status),
                "metadata": self.metadata}


@contextlib.contextmanager
def vendor_math_utils():
    """Expose the test-only Miles math_utils copy under its Miles module name."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    vendor = importlib.import_module("vendor.miles_math_utils")
    names = ("miles", "miles.rollout", "miles.rollout.rm_hub", "miles.rollout.rm_hub.math_utils")
    saved = {n: sys.modules.get(n) for n in names}
    for n in names[:-1]:
        sys.modules[n] = types.ModuleType(n)
    sys.modules[names[-1]] = vendor
    try:
        yield vendor
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


MATH_CASES = [
    ("The answer is \\boxed{4}.", "4"),
    ("<think>maybe \\boxed{3}</think> so \\boxed{\\frac{1}{2}}", "\\frac12"),
    ("\\boxed{0.5}", "\\frac{1}{2}"),
    ("no box here 7", "7"),
    ("\\boxed{x^2+2x+1}", "(x+1)^2"),
    ("\\boxed{12}", "\\boxed{12}"),
    ("\\boxed{5}", ""),
    ("\\boxed{5}", None),
    ("", "1"),
    ("\\boxed{1,000}", "1000"),
]
GSM8K_CASES = [
    ("so \\boxed{18}", "Janet ... #### 18"),
    ("the total is 1,234 dollars", "#### 1234"),
    ("\\boxed{17}", "#### 18"),
    ("no number", "#### 3"),
    ("\\boxed{3.0}", "3"),
    ("\\boxed{2}", None),
    (None, "#### 2"),
]
GDPO_CASES = [
    ("<think>x</think>\\boxed{18}", "steps #### 18"),
    ("\\boxed{17}", "#### 18"),
    ("18", "#### 18"),
    ("\\boxed{1,000}", "#### 1,000"),
    ("", None),
]
LENGTH_CASES = [
    ("short", Status.COMPLETED),
    ("x" * 2000, Status.TRUNCATED),
    ("y" * 300, Status.ABORTED),
    ("", Status.COMPLETED),
    (None, None),
    ("z" * 1024, "completed"),
]
# (policy version, limit, group rewards, group indices) in call order; repeats
# exercise memoized decisions and a version change resets the state.
FILTER_CALLS = [
    (1, 2, [1.0, 0.0, 1.0], [0, 1, 2]),
    (1, 2, [0.0, 0.0, 0.0], [3, 4, 5]),
    (1, 2, [1.0, 1.0, 1.0], [6, 7, 8]),
    (1, 2, [0.0, 0.0, 0.0], [3, 4, 5]),
    (1, 2, [0.5, 0.5, 0.5], [9, 10, 11]),
    (1, 2, [0.25, 0.25], [12, 13]),
    (2, None, [0.0, 0.0], [0, 1]),
    (2, None, [0.0, 0.0], [2, 3]),
    (3, 0, [1.0, 1.0], [0, 1]),
    (3, 0, [1e-12, 0.0], [2, 3]),
    (3, 0, [], []),
]


def _run(coro):
    return asyncio.run(coro)


def _metadata_seed(i: int):
    return None if i % 3 == 0 else ({} if i % 3 == 1 else {"keep": i})


def render() -> dict[str, Any]:
    from yeto.rl import filters, gsm8k_reward, length_reward, math_reward
    from yeto.rl.algos import gdpo_reward

    out: dict[str, Any] = {}
    with vendor_math_utils():
        rows = []
        for i, (resp, label) in enumerate(MATH_CASES):
            s = FakeSample(resp, label, metadata=_metadata_seed(i))
            value = _run(math_reward.reward_func(None, s))
            rows.append({"value": value, **s.snapshot()})
        out["math_reward.reward_func"] = rows
        rows = []
        for i, (resp, label) in enumerate(GDPO_CASES):
            for name in ("reward_func", "correctness_reward"):
                s = FakeSample(resp, label, metadata=_metadata_seed(i))
                value = _run(getattr(gdpo_reward, name)(None, s))
                rows.append({"entry": name, "value": value, **s.snapshot()})
            rows.append({"entry": "components", "value": gdpo_reward.components(resp, label)})
        out["gdpo_reward"] = rows
    rows = []
    for i, (resp, label) in enumerate(GSM8K_CASES):
        s = FakeSample(resp, label, metadata=_metadata_seed(i))
        value = _run(gsm8k_reward.score(None, s))
        rows.append({"value": value, **s.snapshot()})
    out["gsm8k_reward.score"] = rows
    rows = []
    for i, (resp, status) in enumerate(LENGTH_CASES):
        s = FakeSample(resp, None, status=status, metadata=_metadata_seed(i))
        value = _run(length_reward.reward_func(None, s))
        rows.append({"value": value, **s.snapshot()})
    out["length_reward.reward_func"] = rows
    args = types.SimpleNamespace()
    rows = []
    for version, limit, rewards, indices in FILTER_CALLS:
        args.yeto_rl_policy_version = version
        args.yeto_rl_dynamic_sampling_max_replacements = limit
        group = [FakeSample("r", None, index=ix, reward=r) for r, ix in zip(rewards, indices)]
        result = filters.bounded_nonzero_reward_std(args, group)
        state = args._yeto_bounded_filter_state
        rows.append({"keep": result.keep, "reason": result.reason,
                     "state": {k: state[k] for k in ("rollout_id", "rejections", "forced")},
                     "decisions": len(state["decisions"])})
    out["filters.bounded_nonzero_reward_std"] = rows
    return out


def dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=True) + "\n"


if __name__ == "__main__":
    text = dumps(render())
    if "--write" in sys.argv:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(text)
    print(text[:2000])
