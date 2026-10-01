"""Gather every rank dump of the A8 arms into one file per (arm, tag) for offline G1-G6 (container).

    python pack_states.py <work>   ->  <work>/packed/<arm>_<tag>.pt + <work>/packed/index.json

A packed state is what ``compare.gathered`` builds: the adapter (DP-replicated, checked equal across
ranks), the named optimizer state merged over the DP ranks with fork-M5 ``merge_named_optimizer_states``
(FP32 main ``param``, ``exp_avg``, ``exp_avg_sq``, scalar ``step``, ``hyper``), the scheduler, Megatron
counters / weight version and the per-rank RNG digests. Size control: ``s8`` states (only G6 reads
them) keep the FP32 main copy only. ``index.json`` lists size and sha256 of every file and, per state,
field-level digests (for explaining digest differences without the tensors).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from harness import read_events  # noqa: E402

PARAM_ONLY_TAGS = ("s8",)


def field_digests(state: dict) -> dict:
    from yeto.rl.engine.miles_adapter.cut_plugin import state_digest

    opt = state["optimizer"]
    out = {"adapter": state_digest(state.get("adapter") or {}), "scheduler": state_digest(state["scheduler"]),
           "counters": state_digest(state.get("counters") or {})}
    for key in ("param", "exp_avg", "exp_avg_sq"):
        if all(key in e["tensors"] for e in opt.values()):
            out[key] = state_digest({n: e["tensors"][key] for n, e in opt.items()})
    out["step"] = state_digest({n: e["scalars"] for n, e in opt.items()})
    out["hyper"] = state_digest({n: e["hyper"] for n, e in opt.items()})
    out["shape"] = state_digest({n: (e["numel"], tuple(e["shape"])) for n, e in opt.items()})
    return out


def gather(dumps: list[dict], merge) -> dict:
    import torch

    first = dumps[0]
    for d in dumps[1:]:
        for n, v in first["adapter"].items():
            if not torch.equal(v, d["adapter"][n]):
                raise AssertionError(f"DP ranks hold different adapter {n}")
    return {"adapter": first["adapter"], "optimizer": merge([d["optimizer_named"] for d in dumps]),
            "scheduler": first["scheduler"],
            "counters": {"megatron": first.get("megatron_counters") or {},
                         "weight_version": first.get("weight_version")},
            "rank_rng": {json.dumps(d["coord"], sort_keys=True): d["rng_digest"] for d in dumps}}


def pack(work: Path, merge, load) -> dict:
    import torch

    from yeto.rl.engine.miles_adapter.cut_plugin import to_safe

    out_dir = Path(work) / "packed"
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict = {"files": {}, "fields": {}}
    for arm_dir in sorted((Path(work) / "arms").iterdir()):
        for e in read_events(arm_dir):
            if e["kind"] != "dump":
                continue
            name = f"{arm_dir.name}_{e['tag']}"
            try:
                state = gather([load(r["path"]) for r in e["ranks"]], merge)
            except Exception as exc:  # noqa: BLE001 - pack what exists (safety net after a failure)
                index.setdefault("errors", {})[name] = f"{type(exc).__name__}: {exc}"
                print(f"pack {name}: {exc}", flush=True)
                continue
            if e["tag"] in PARAM_ONLY_TAGS:
                state["adapter"] = None
                for entry in state["optimizer"].values():
                    entry["tensors"] = {"param": entry["tensors"]["param"]}
            index["fields"][name] = field_digests(state)
            path = out_dir / f"{name}.pt"
            torch.save(to_safe(state), path)
            index["files"][path.name] = {"bytes": path.stat().st_size,
                                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            print(f"packed {name} {path.stat().st_size / 2**20:.1f} MiB", flush=True)
    (out_dir / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    return index


def main(argv: list[str]) -> int:  # container
    from compare import _load
    from miles.backends.megatron_utils.lora.dp_invariant_state import merge_named_optimizer_states

    index = pack(Path(argv[0]), merge_named_optimizer_states, _load)
    total = sum(f["bytes"] for f in index["files"].values())
    print(f"packed {len(index['files'])} states, {total / 2**20:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
