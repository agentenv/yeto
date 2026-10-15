"""Fake sampler / trainer processes for tests/test_rl_buffer_rollout.py (no Ray, no Miles).

``python -m tests.buffer_rollout_procs sampler STORE ID GROUPS VERSION``
    writes GROUPS groups (2 samples each) generated at engine VERSION;
``python -m tests.buffer_rollout_procs trainer STORE CUT LOG ROUNDS GROUPS [CRASH_ROUND]``
    resumes at the cut's watermark, trains ROUNDS rounds of GROUPS groups (the trained
    sample ids go to LOG), cuts after each round; exits hard after taking CRASH_ROUND.
"""

from __future__ import annotations

import json
import os
import sys

from yeto.rl.adapters.miles.buffer_rollout import BUFFER_ROOT_ENV, BatchReader, GroupWriter, open_buffer


def sample(index: int, group: int, version: int) -> dict:
    return {"index": index, "group_index": group, "response_length": 3, "tokens": [1, 2, 3, 4, 5],
            "rollout_log_probs": [-0.1, -0.2, -0.3], "status": "completed",
            "weight_versions": [[{"version": str(version), "abs_start": 2, "abs_end": 5}]]}


def sampler(store: str, sampler_id: str, groups: int, version: int) -> None:
    buffer = open_buffer({BUFFER_ROOT_ENV: store})
    writer = GroupWriter(buffer, sampler_id)
    start = len(buffer)
    for g in range(start, start + groups):
        writer.write([sample(2 * g, g, version), sample(2 * g + 1, g, version)], group_key=f"g{g}")


def trainer(store: str, cut_path: str, log_path: str, rounds: int, groups: int, crash: int | None) -> None:
    cut = json.loads(open(cut_path).read()) if os.path.exists(cut_path) else {"next": 0, "watermark": 0}
    reader = BatchReader(open_buffer({BUFFER_ROOT_ENV: store}, batch_size=groups), max_policy_age=2,
                         wait_s=0.0, poll_s=0.0)
    reader.reset(cut["watermark"])
    rid = cut["next"]
    for rid in range(cut["next"], cut["next"] + rounds):
        taken, report = reader.take(rid, groups, current_version=rid + 1)
        if rid == crash:
            os._exit(3)  # crash after consuming, before the train step and the cut
        with open(log_path, "a") as fh:  # the train step
            fh.write(json.dumps({"rollout_id": rid, "groups": [g[0]["group_index"] for _, g in taken],
                                 "stale": report.stale_groups}) + "\n")
        cut = {"next": rid + 1, "watermark": reader.watermark_before(rid + 1)}
        tmp = cut_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(cut, fh)
        os.replace(tmp, cut_path)


if __name__ == "__main__":
    mode, *rest = sys.argv[1:]
    if mode == "sampler":
        sampler(rest[0], rest[1], int(rest[2]), int(rest[3]))
    else:
        trainer(rest[0], rest[1], rest[2], int(rest[3]), int(rest[4]),
                int(rest[5]) if len(rest) > 5 else None)
