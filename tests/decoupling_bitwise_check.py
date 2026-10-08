"""CPU bitwise check: plugin functions before/after a decoupling refactor
(yeto-framework-decoupling task 4.4 / 4.6 / D6a item 3, hash-migration.md).

Takes the *old* function bodies from a git revision (default: the stage-2
branch tip ``s16-decouple-p2``), executes them in the namespace of the
current module (so helpers are shared and only the refactored body differs),
feeds both the same fixed inputs and compares outputs with ``torch.equal``
on float32 tensors.

    PYTHONPATH=/tmp/s15-noray python tests/decoupling_bitwise_check.py [REV]

Prints one line per case and exits non-zero on any difference. CPU only,
no Ray, no Miles.
"""

from __future__ import annotations

import ast
import contextlib
import io
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from yeto.rl.algos import reward_pipeline as rp  # noqa: E402
from yeto.rl.algos import seq_adv  # noqa: E402
from yeto.rl.engine.algorithm import load_extensions  # noqa: E402

load_extensions()


def _old_function(rev: str, path: str, name: str, module) -> object:
    source = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{path}"],
                            check=True, capture_output=True, text=True).stdout
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[node], type_ignores=[]), f"{rev}:{path}", "exec"), namespace)
    return namespace[name]


class _Sample(SimpleNamespace):
    pass


def _samples(rewards, *, components=None, segments_of=None):
    out = []
    for i, _ in enumerate(rewards):
        rid = segments_of[i] if segments_of else i
        meta = {}
        if components is not None:
            meta[seq_adv.REWARD_COMPONENTS_KEY] = components[rid]
        out.append(_Sample(index=i, rollout_id=rid, metadata=meta, response_length=10 + i))
    return out


CASES = [
    # (label, rewards, groups, segments_of)
    ("binary-4x2", [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0], [[0, 1, 2, 3], [4, 5, 6, 7]], None),
    ("float-3x3", [0.3, 0.71, 0.12, 0.5, 0.5, 0.5, 0.9, 0.05, 0.33], [[0, 1, 2], [3, 4, 5], [6, 7, 8]], None),
    ("single", [0.42], [[0]], None),
    ("multiseg", [1.0, 1.0, 0.0, 1.0, 0.0, 0.0], [[0, 1, 2, 3, 4, 5]], [0, 0, 1, 2, 3, 3]),
]


def _args(**kw):
    base = dict(advantage_estimator="grpo", rewards_normalization=True, grpo_std_normalization=True,
                yeto_rl_event_tape=None, yeto_rl_learner_id=None)
    base.update(kw)
    return SimpleNamespace(**base)


def main(argv: list[str]) -> int:
    rev = argv[0] if argv else "s16-decouple-p2"
    failures = 0

    def check(label, old, new):
        nonlocal failures
        same = torch.equal(torch.tensor(old, dtype=torch.float), torch.tensor(new, dtype=torch.float))
        failures += 0 if same else 1
        print(f"{'OK  ' if same else 'DIFF'} {label}")

    old_grpo = _old_function(rev, "yeto/rl/algos/reward_pipeline.py", "grpo_default", rp)
    for label, rewards, groups, seg in CASES:
        samples = _samples(rewards, segments_of=seg)
        for est, std in (("grpo", True), ("grpo", False), ("gspo", True),
                         ("reinforce_plus_plus_baseline", True)):
            args = _args(advantage_estimator=est, grpo_std_normalization=std)
            check(f"grpo_default {label} est={est} std={std}",
                  old_grpo(args, samples, rewards, groups, {}),
                  rp.grpo_default(args, samples, rewards, groups, {}))

    old_gdpo = _old_function(rev, "yeto/rl/algos/seq_adv.py", "gdpo", seq_adv)
    for whiten in (True, False):
        rewards = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
        groups = [[0, 1, 2], [3, 4, 5]]
        comps = {i: {"acc": float(r), "fmt": [0.2, 0.9, 0.4, 0.4, 0.7, 0.1][i]}
                 for i, r in enumerate(rewards)}
        samples = _samples(rewards, components=comps)
        cfg = {"schema": seq_adv.SEQ_ADV_SCHEMA, "algorithm_sha256": "check",
               "gdpo": {"components": [{"name": "acc", "weight": 1.0}, {"name": "fmt", "weight": 0.37}],
                        "whiten": whiten}}
        args = _args()
        setattr(args, seq_adv.SEQ_ADV_ATTR, {"config": cfg, "sha256": rp.canonical_sha256(cfg)})
        with contextlib.redirect_stdout(io.StringIO()):
            old = old_gdpo(args, samples, rewards, groups, {})
            new = seq_adv.gdpo(args, samples, rewards, groups, {})
        check(f"gdpo whiten={whiten}", old, new)

    for label, rewards, groups, seg in CASES:
        samples = _samples(rewards, segments_of=seg)
        if set(rewards) <= {0.0, 1.0}:
            with contextlib.redirect_stdout(io.StringIO()):
                old = _old_function(rev, "yeto/rl/algos/seq_adv.py", "maxrl", seq_adv)(
                    _args(), samples, rewards, groups, {})
                new = seq_adv.maxrl(_args(), samples, rewards, groups, {})
            check(f"maxrl {label} (body unchanged)", old, new)
    print(f"{failures} difference(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
