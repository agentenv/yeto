"""CPU tests of the numeric final-answer reward (yeto.rl.numeric_answer_reward).

REAL_TAILS are the last 260 characters of real responses to DAPO-Math-17k rows
(the S27 task prompt "... Answer: \\boxed{$Answer} ...") from the S27 offline
cut evaluations (infra-drafts/SESSION27/logs/<run>/eval-island*/cuts/*.json,
labels from s1-runs/s27-x/eval/heldout-0.6b.jsonl). Each row stores the score of
the old ``gsm8k_grade`` and of the new grader; the old scores reproduce the
recorded rewards. NEMOTRON_HEADS are the response heads Miles printed for
Nemotron-3 Nano in s27-gr-g2/launch.stream.log ("First rollout sample" /
"Finish rollout", 512-character previews). As text cut off before an answer,
they must score 0.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from yeto.rl.rewards import registry
from yeto.rl.rewards.builtin import (
    answer_number,
    final_answer,
    gsm8k_grade,
    last_boxed,
    numeric_answer_grade,
)
from yeto.rl.rewards.types import Trajectory

# (response tail, label, old gsm8k_grade, new numeric_answer_grade)
REAL_TAILS = [
    # GK/s27-xk-x1-k1go20-b h0
    ('\n\nFrom earlier, we have $P = 2Q$. Substitute this into the equation above:\n\n$$\nQ - 12(2Q) = -416\n\\Rightarrow Q - 24Q = -416\n\\Rightarrow -23Q = -416 \\Rightarrow Q = \\frac{416}{23} = 18\n$$\n\nThen $P = 2Q = 36$\n\nSo the value of $xy + yz + zx$ is:\n\n$$\n\\boxed{36}\n$$',
     '672', 0.0, 0.0),
    # GK/s27-xk-x1-k1go20-b h46
    ("m/s}$\n- Cate's speed = $6 \\, \\text{m/s}$\n\nSince they are moving towards each other, the time it takes for Dan to reach Cate is:\n\n$$\nt = \\frac{50 \\, \\text{m}}{8 + 6} = \\frac{50}{14} = \\frac{25}{7} \\approx 3.571 \\, \\text{seconds}\n$$\n\nAnswer: \\boxed{\\frac{25}{7}}",
     '25', 1.0, 0.0),
    # GK/s27-xk-x1-k1go20-b h56
    ("\n\\frac{6050}{3}\n$$\n\nPerforming the division:\n\n$$\n\\frac{6050}{3} = 2016.666...\n$$\n\nBut since we can't have fractional triangles, the **maximum number of non-overlapping non-collinear triangles** is thus:\n\n$$\n2016\n$$\n\n---\n\n### Final answer:\n\nAnswer: \\boxed{2016}",
     '2016', 1.0, 1.0),
    # GK/s27-xk-x1-k1go20-b h94
    ('^{(0)}|} = \\frac{1}{3^7}\n$$\n\nSo:\n\n$$\n\\frac{|OA_{2014}^{(7)}|}{|OA_{2014}^{(0)}|} = \\frac{1}{3^7}\n$$\n\nWe can now write:\n\n$$\n\\boxed{\\frac{1}{3^7} = \\frac{1}{2187}}\n$$\n\nSo, $ p = 1 $, $ q = 2187 $, and $ p+q = 2188 $.\n\n---\n\n### ✅ Final Answer:\n\n$$\n\\boxed{2188}\n$$',
     '2188', 1.0, 1.0),
    # GK/s27-xk-x1-k1go20-b h115
    ('solve for $ x $:\n\n$$\n12x - 225 = 17(-x + 200).\n$$\n\nExpand the right-hand side:\n\n$$\n12x - 225 = -17x + 3400.\n$$\n\nCombine like terms:\n\n$$\n12x + 17x = 3400 + 225,\n$$\n\n$$\n29x = 3625.\n$$\n\nSolve for $ x $:\n\n$$\nx = \\frac{3625}{29}.\n$$\n\nAnswer: \\boxed{\\frac{3625}{29}}',
     '125', 0.0, 1.0),
    # GK/s27-xk-x1-k1go20-b h152
    ('\nSo assume that $ x = 1 $, $ y = 2 $ and check:\n\nFirst equation: $ 1 - 3(1)(4) = 1 - 12 = -11 \\ne 3 $\n\nTry $ x = 1 $, $ y = 1.5 \\Rightarrow -1.5 $ each time.\n\nEventually, desired $ x = a^2 $ and $ y = b^2 $ such that:\n\n$$\nx^3 - 3xy^2 = 3, \\quad y^3 - 3x^2y = 3',
     '3', 1.0, 0.0),
    # GK/s27-xk-x1-k1go20-b h224
    ('mes 25 \\times 36 \\times 49 \\times 64 \\times 81) = (1 \\times 2^2 \\times 3^2 \\times 5^2 \\times 6^2 \\times 7^2)\n$$\n\nNow this is a square of:\n\n$$\n(2 \\times 3 \\times 5 \\times 7)^2 = 210^2\n$$\n\n---\n\n### Step 3: Conclude the result\n\n$$\n\\text{Answer: } \\boxed{210^2}\n$$',
     '44100', 0.0, 1.0),
    # GK/s27-xk-x1-k1go20-b h253
    (' between two options based on the remaining value.\n\nThis behavior ensures that the player who starts (Euhan) will **always be the one forced to say it**, so **Minjune has a winning strategy** for any $N > 2019$.\n\n---\n\n### Final Answer:\n\n$$\nAnswer: \\boxed{2020}',
     '2020', 1.0, 1.0),
    # GK/s27-xk-x1-k1go20-b h2
    ('ble value** of $a$ is:\n\n$$\n\\boxed{9997}\n$$\n\n---\n\n### Step 4: Find the sum of the **smallest** and **largest** values\n\n$$\n\\text{Smallest } a = 3 \\\\\n\\text{Largest } a = 9997 \\\\\n\\text{Sum} = 3 + 9997 = \\boxed{9997 + 3 = 10000}\n$$\n\n---\n\n**Answer: $\\boxed{10000}$**',
     '10000', 1.0, 1.0),
    # GK/s27-xk-x1-k1go20-b h52
    (' = 4 $\n- $ xy = 2 $, $ x + y + (x, y) = 4 $ → So this is **not valid**\n\nHence, we discard this pair.\n\n---\n\nSo the **only valid pair** is:\n\n- $ x = 2 $, $ y = 3 $\n\n---\n\n**Answer: $ \\frac{1}{\\text{Once}} $**\n\nBut when writing Answer:\n\n---\n\nAnswer: \\boxed{(2, 3)}',
     '3', 1.0, 0.0),
    # X1/s27-x1-k1-a h251
    ('{3}{2} \\right)^2 } = \\sqrt{ \\left( \\frac{1}{2} \\right)^2 + \\left( -1 \\right)^2 } = \\frac{1}{\\sqrt{2}}\n$$\n\nSo the area of square $PQRS$ is:\n\n$$\n\\text{Area} = s^2 = \\left( \\frac{1}{\\sqrt{2}} \\right)^2 = \\frac{1}{2}\n$$\n\n### Final Answer:\n$$\n\\boxed{\\frac{1}{2}}\n$$',
     '1', 1.0, 0.0),
    # X1/s27-x1-strict20-a h50
    (' A_{\\text{large}}\n$$\n$$\n14 = \\frac{4}{9} \\times A\n$$\n$$\nA = \\frac{14 \\times 9}{4} = \\frac{126}{4} = 31.5\n$$\n\nSo, the area of the large square is **31.5**, or in **fractional form** as $ \\frac{63}{2} $.\n\n---\n\n### Final Answer\n\n$$\nAnswer: \\boxed{\\frac{63}{2}}\n$$',
     '63', 1.0, 0.0),
]

# (response head as printed by Miles, label, recorded reward of the full response)
NEMOTRON_HEADS = [
    ("The cubic is  \n", "5", 0.0),
    ("We are given the system  \n\n\\[\n\\begin{cases}\\", "18", 0.0),
    ("We are given  \n\n\\[\nx+", "10", 1.0),
    ("The cubic can be written as  \n\n\\[\n3x^{3}-3x^{2}+m=0\\qquad ( \\text{or } x", "0", 1.0),
    ("Let  \n\n\\[\nf_a(x)=x^{4}-5x+a ,\\qquad a\\in \\math", "0", 1.0),
]


@pytest.mark.parametrize("tail,label,old,new", REAL_TAILS)
def test_real_dapo_tails(tail, label, old, new):
    assert gsm8k_grade(tail, label) == old
    assert numeric_answer_grade(tail, label) == new


@pytest.mark.parametrize("head,label,_recorded", NEMOTRON_HEADS)
def test_nemotron_cut_off_heads_score_zero(head, label, _recorded):
    assert numeric_answer_grade(head, label) == 0.0


def test_old_fallback_credits_a_cut_off_response():
    # the "last number in the last 200 characters" fallback reads the "0" of "m=0"
    head, label, _ = NEMOTRON_HEADS[3]
    assert gsm8k_grade(head, label) == 1.0
    assert numeric_answer_grade(head, label) == 0.0


@pytest.mark.parametrize("answer,value", [
    ("34", 34.0), ("-7", -7.0), ("1\\,007", 1007.0), ("1{,}440", 1440.0), ("270,901", 270901.0),
    ("1\\!000\\!000", 1e6), ("1 000", 1000.0), ("\\$7", 7.0), ("\\text{\\textbf{2020}}", 2020.0),
    ("{\\displaystyle \\textbf{2020}}", 2020.0), ("\\answer{2016}", 2016.0), ("x = 5", 5.0),
    ("3 + 7 = 10", 10.0), ("45^\\circ", 45.0), ("45^{\\circ}", 45.0), ("25\\%", 25.0),
    ("12 \\text{ cm}", 12.0), ("**34**", 34.0), ("\\(34\\)", 34.0), ("\\frac{3625}{29}", 125.0),
    ("\\dfrac{1}{2}", 0.5), ("210^2", 44100.0), ("2^{10}", 1024.0), ("8.", 8.0),
    # not one number: no credit for the last number inside an expression
    ("2\\sqrt{3}", None), ("1, 2, 3", None), ("(2, 3)", None), ("3^{2188} + 1", None),
    ("512 \\text{ and } 125", None), ("", None), (None, None),
])
def test_answer_number(answer, value):
    assert answer_number(answer) == value


def test_final_answer_rules():
    assert last_boxed("a \\boxed{\\frac{1}{2}} b") == "\\frac{1}{2}"
    assert last_boxed("\\boxed{1} then \\boxed{2") == "1"  # unclosed last box: cut off
    assert last_boxed("no box") is None
    # only the text after the last </think> is graded
    assert final_answer("<think>\\boxed{3}</think>Answer: \\boxed{4}") == "4"
    assert final_answer("<think>\\boxed{3}</think>no answer") is None
    # "Answer:" line when there is no box
    assert final_answer("work\n**Answer:** 34") == "** 34"
    assert numeric_answer_grade("work\n**Answer:** 34", "34") == 1.0
    assert numeric_answer_grade("so the total is 34", "34") == 0.0  # no last-number fallback
    assert numeric_answer_grade("Answer: \\boxed{34}", "#### 34") == 1.0  # GSM8K label form
    assert numeric_answer_grade("Answer: \\boxed{34}", None) == 0.0


def test_entry_point_and_registry():
    from yeto.rl.numeric_answer_reward import score

    s = SimpleNamespace(response="Answer: \\boxed{1{,}440}", label="1440", metadata=None)
    assert asyncio.run(score(None, s)) == 1.0
    assert s.metadata == {"success": True}
    s = SimpleNamespace(response="Answer: \\boxed{\\frac{25}{7}}", label="25", metadata={})
    assert asyncio.run(score(None, s)) == 0.0
    assert s.metadata == {"success": False}
    assert "numeric_answer" in registry.registered_names()
    plugin = registry.load_reward("numeric_answer")
    assert plugin.fn(Trajectory(response="\\boxed{7}", label="7")).value == 1.0
