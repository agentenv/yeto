"""Dashboard field mapping of a buffer (async GRPO) trainer round.

The records follow a real Miles buffer round: ``rl_round_trained`` carries the
train-vs-rollout KL in ``mean_kl`` (the driver writes it there) and the PPO KL in
``train_metrics.ppo_kl``; the ``rl_local_round`` of the same round, written after
it, carries the PPO KL in ``mean_kl``. The page must show each under its own name,
whatever the order. ``resp_len_mean`` counts loss-mask tokens; the full response
length (with tool / env tokens) is ``resp_full_len_mean``.
"""

from __future__ import annotations

import pytest

from yeto.dashboard.reducer import Reducer, ppo_kl
from yeto.status_metrics import render_tape_summary

T0 = 1_795_144_397.0


def round_trained(rid: int, *, tr_kl: float, ppo: float, resp: float, full: float | None = None) -> dict:
    rec = {"event": "rl_round_trained", "island_id": 0, "rollout_id": rid, "time_unix": T0 + 600 * rid,
           "mean_kl": tr_kl, "grad_norm": 0.006, "reward_mean": 0.34, "resp_len_mean": resp,
           "resp_len_p95": 2 * resp, "trained_samples": 62,
           "mismatch": {"train_rollout_kl": tr_kl, "tis_clipfrac": 0.097},
           "train_metrics": {"ppo_kl": ppo, "train_rollout_kl": tr_kl,
                             "train_rollout_logprob_abs_diff": 0.21}}
    if full is not None:
        rec["resp_full_len_mean"] = full
    return rec


def local_round(rid: int, *, ppo: float) -> dict:
    return {"event": "rl_local_round", "island_id": 0, "local_round_id": rid + 1, "rollout_id": rid,
            "time_unix": T0 + 600 * rid + 3, "mean_kl": ppo, "grad_norm": 0.006, "reward_mean": 0.34}


def rounds(records: list[dict]) -> list[dict]:
    r = Reducer()
    for rec in records:
        r.feed(rec)
    return r.round_records(r.island("0"))


@pytest.mark.parametrize("local_first", [False, True])
def test_train_rollout_kl_and_ppo_kl_are_kept_apart(local_first):
    recs = []
    for rid, (tr, ppo) in enumerate([(0.10316, 0.000197), (0.09027, 0.0000895)]):
        pair = [round_trained(rid, tr_kl=tr, ppo=ppo, resp=12207.5), local_round(rid, ppo=ppo)]
        recs += pair[::-1] if local_first else pair
    got = rounds(recs)
    assert [x["kl_tr"] for x in got] == [0.10316, 0.09027]
    assert [x["kl"] for x in got] == [0.000197, 0.0000895]


def test_mean_kl_that_is_the_train_rollout_kl_is_not_the_ppo_kl():
    rec = round_trained(0, tr_kl=0.1, ppo=0.0002, resp=1.0)
    del rec["train_metrics"]["ppo_kl"]
    assert ppo_kl(rec) is None  # mean_kl == train_rollout_kl: not a PPO KL
    assert ppo_kl({"mean_kl": 0.003}) == 0.003  # a plain trainer round: mean_kl is its KL
    assert ppo_kl({"mean_kl": 0.1, "train_metrics": {"kl_loss": 0.02}}) == 0.02


def test_response_length_fields():
    got = rounds([round_trained(0, tr_kl=0.1, ppo=0.0002, resp=12207.5, full=56985.7),
                  round_trained(1, tr_kl=0.1, ppo=0.0002, resp=10380.5)])
    assert got[0]["resp_mean"] == 12207.5 and got[0]["resp_full_mean"] == 56985.7
    assert got[1]["resp_mean"] == 10380.5 and got[1]["resp_full_mean"] is None  # older tape: not inferred


def test_status_summary_shows_both_kls(tmp_path):
    import json
    tape = tmp_path / "t.jsonl"
    tape.write_text(json.dumps(round_trained(0, tr_kl=0.103, ppo=0.0002, resp=1.0)) + "\n")
    (line,) = [x for x in render_tape_summary(tape) if x.startswith("TRAIN")]
    assert "ppo_kl=0.0002" in line and "train_rollout_kl=0.103" in line


AXIS_JS = """
const fs=require('fs');global.localStorage={getItem(){return null},setItem(){}};global.location={hash:''};
global.document={getElementById(){return null},body:null};
eval(fs.readFileSync(process.argv[1],'utf8')+';global.YE=YetoExt;');
const a=JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify(a.map(function(r){var t=YE.nice(r[0],r[1],4),f=YE.numAxis(t);return t.map(f)})));
"""


def test_axis_labels_fit_the_margin():
    """Tick labels of small KL values used to be 0.000200 (cut by the left margin):
    scientific below 1e-3, one decimal count per axis, all labels distinct."""
    import json
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    js = Path(__file__).parent.parent / "yeto" / "dashboard" / "static" / "sft.js"
    ranges = [[0, 0.000197], [0, 0.2125], [0, 0.0064], [9e-6, 1.2e-5], [0, 58000]]
    res = subprocess.run([node, "-e", AXIS_JS, str(js), json.dumps(ranges)], capture_output=True, text=True,
                         timeout=60)
    assert res.returncode == 0, res.stderr
    kl, lp, gn, lr, resp = json.loads(res.stdout)
    assert kl[-1] == "2.0e-4" and kl[0] == "0"
    assert lp[-1] == "0.25" and gn[-1] == "0.008"
    for labels in (kl, lp, gn, lr, resp):
        assert len(set(labels)) == len(labels) and max(map(len, labels)) <= 7, labels


def test_rl_page_names_both_kls():
    from yeto.dashboard.export import render_page

    r = Reducer()
    for rec in (round_trained(0, tr_kl=0.103, ppo=0.0002, resp=1.0), local_round(0, ppo=0.0002)):
        r.feed(rec)
    html = render_page(r.full_view())
    assert '["rfKl","train-rollout KL"' in html and '["rfPkl","PPO KL"' in html
    assert "mean_kl / train_rollout_kl" not in html
