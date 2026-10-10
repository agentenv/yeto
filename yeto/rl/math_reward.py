"""Rule-based math-answer reward for Miles RL (MATH-500 style rows).

A smoke-test environment, not a product reward: it needs no external
service, so it is the quickest way to check that an RL island trains at
all. Launch with

    --data HuggingFaceH4/MATH-500 --rl-prompt-column problem --rl-label-column answer
    --reward-function examples/math_reward.py:reward_func

(see examples/README.md for the full line). The response's last
`\\boxed{...}` answer, after any `</think>` block, is graded against the
label with Miles' own math graders (mathd normalisation, then sympy
equivalence). Reward is 1.0 for a correct answer and 0.0 otherwise.

``reward_func`` also records ``sample.metadata["success"]`` (bool: the final
answer is correct), the flag the critic fork's VAPO positive-example LM loss
reads (``--positive-example-source success``). The score is unchanged.
"""

from __future__ import annotations


def score(response: str, label) -> float:
    from miles.rollout.rm_hub.math_utils import (
        extract_answer,
        grade_answer_mathd,
        grade_answer_sympy,
    )

    if label is None or str(label).strip() == "":
        return 0.0
    solution = response.split("</think>")[-1]
    answer = extract_answer(solution)
    if answer is None:
        return 0.0
    truth = str(label)
    if "\\boxed" in truth:
        truth = extract_answer(truth) or truth
    return 1.0 if grade_answer_mathd(answer, truth) or grade_answer_sympy(answer, truth) else 0.0


SUCCESS_DECLARATION_ATTR = "yeto_writes_success"


def writes_success(fn):
    """Declare that reward function ``fn`` writes ``sample.metadata['success']``.

    The launch check ``positive_lm_success`` (rl-algo-critic-family) rejects
    ``loss.positive_lm_source='success'`` when the run's reward function lacks
    this declaration: the Miles fork otherwise fails on the island at the first
    critic step (S19 #6 VAPO G3 s19-vapo-g3-20261010a).
    """

    setattr(fn, SUCCESS_DECLARATION_ATTR, True)
    return fn


def set_success(sample, success: bool) -> None:
    """Write the boolean full-success flag into ``sample.metadata`` (in place)."""
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        sample.metadata = metadata
    metadata["success"] = bool(success)


@writes_success
async def reward_func(args, sample, **kwargs) -> float:
    value = score(sample.response or "", sample.label)
    set_success(sample, value == 1.0)
    return value
