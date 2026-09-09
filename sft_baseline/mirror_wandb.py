"""Numeric-only W&B mirror. Run locally; never copies credentials to the node.

Each event is durably assigned a W&B history step BEFORE log(). On restart,
W&B's resumed step is authoritative: replay pending steps, never repost older
ones. The local journal must be retained with the same W&B run ID.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

HOST = "ubuntu@100.66.155.50"
KNOWN_HOSTS = "/private/tmp/yeta-labeling-known-hosts"
FIELDS = {
    "train": {"step", "epoch", "loss", "grad_norm", "lr", "mem", "tps", "tps_per_gpu", "num_tokens_per_step", "num_label_tokens", "max_rank_peak_allocated_gib", "max_rank_peak_reserved_gib"},
    "validation": {"step", "epoch", "val_loss", "lr", "num_label_tokens", "mem"},
}


def clean_row(row, stream):
    if not isinstance(row, dict):
        raise ValueError("Metric row must be an object")
    required = {"step", "epoch", "timestamp", "loss" if stream == "train" else "val_loss"}
    if not required <= row.keys():
        raise ValueError("Metric row is missing required fields")
    result = {}
    for key in FIELDS[stream] & row.keys():
        value = row[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Metric must be finite numeric: " + key)
        if key in {"step", "epoch"} and (value < 0 or int(value) != value):
            raise ValueError("Invalid metric axis: " + key)
        result[key] = value
    timestamp = row["timestamp"]
    if not isinstance(timestamp, str) or len(timestamp) > 64:
        raise ValueError("Invalid metric timestamp")
    dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    result["timestamp"] = timestamp
    return result


def parse_bytes(data, stream):
    """Ignore only the unfinished final line; malformed complete rows fail."""
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("Metric file exceeds the bounded 2 MiB read")
    rows, offset = [], 0
    for raw in data.splitlines(keepends=True):
        if not raw.endswith(b"\n"):
            break
        end = offset + len(raw)
        if raw.strip():
            payload = clean_row(json.loads(raw), stream)
            rows.append({"offset": offset, "end": end, "sha256": hashlib.sha256(raw).hexdigest(), "payload": payload})
        offset = end
    return {"rows": rows, "complete_bytes": offset, "total_bytes": len(data)}


def parse_stdout(text):
    """Only the exact numeric trainer line is allowed to leave the node."""
    number = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| INFO \| root \| step (\d+) \| epoch (\d+) \| loss "
                         + number + r" \| grad_norm " + number + r" \| lr " + number + r" \| mem " + number
                         + r" GiB \| tps " + number + r"\(" + number + r"/gpu\) \| num_label_tokens (\d+)\s*$")
    progress_prefix = re.compile(r"^Training:\s+\d+%\|[^|\r\n]*\|\s+\d+/\d+\s+\[[^\]\r\n]{1,120}\]")
    rows = []
    for line in text.splitlines():
        line = progress_prefix.sub("", line, count=1)
        match = pattern.fullmatch(line)
        if not match:
            continue
        values = match.groups()
        row = {"timestamp": values[0].replace(" ", "T") + "+00:00", "step": int(values[1]), "epoch": int(values[2])}
        for key, value in zip(["loss", "grad_norm", "lr", "mem", "tps", "tps_per_gpu"], values[3:9]):
            row[key] = float(value)
        row["num_label_tokens"] = int(values[9])
        rows.append({"payload": clean_row(row, "train"), "sha256": hashlib.sha256(line.encode()).hexdigest()})
    for line in text.splitlines():
        if not line.startswith('{"event": "verified_optimizer_update",'):
            continue
        event = json.loads(line)
        if type(event.get("step")) is not int or not isinstance(event.get("measurements"), dict):
            raise ValueError("Invalid exact optimizer metric event")
        matches = [row for row in rows if row["payload"]["step"] == event["step"]]
        # The current source has one epoch. Require an unambiguous standard
        # trainer line for its timestamp and epoch instead of inventing them.
        if len(matches) != 1:
            continue
        target = matches[0]
        exact = {key: value for key, value in event["measurements"].items()
                 if key in FIELDS["train"] - {"epoch", "step"}}
        payload = clean_row({**target["payload"], **exact}, "train")
        if not same_stdout_values(target["payload"], payload):
            raise ValueError("Exact optimizer metrics disagree with trainer stdout")
        target["payload"] = payload
        target["sha256"] = hashlib.sha256((target["sha256"] + line).encode()).hexdigest()
    return rows


def same_stdout_values(fallback, precise):
    formats = {"loss": ".4f", "grad_norm": ".4f", "lr": ".2e", "mem": ".2f", "tps": ".2f", "tps_per_gpu": ".2f"}
    for key in fallback.keys() & precise.keys() - {"timestamp"}:
        if key in formats:
            if format(fallback[key], formats[key]) != format(precise[key], formats[key]):
                return False
        elif fallback[key] != precise[key]:
            return False
    return True


def source_identity(container, directory, source_run_id):
    if not re.fullmatch(r"[0-9a-f]{64}", container):
        raise ValueError("Source must be the full immutable container ID")
    if not re.fullmatch(r"/data/sft_baseline_20260908/runs/[A-Za-z0-9_-][A-Za-z0-9_./-]*/checkpoints", directory):
        raise ValueError("Source must be a checkpoint metric directory within the training runs")
    if any(part in {".", "..", ""} for part in directory.split("/")[1:]):
        raise ValueError("Source directory must be canonical")
    if source_run_id and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", source_run_id):
        raise ValueError("Invalid source run ID")
    return {"host": HOST, "container": container, "directory": directory, "run_id": source_run_id}


def verify_state_source(state, source):
    if state.get("schema") != 2 or state.get("source") != source:
        raise ValueError("Mirror state belongs to a different source; use a new state file")
    if state.get("finished"):
        raise ValueError("This mirror is already finished")


def remote_script(source):
    # Use the SAME tested parser remotely, so even unexpected extra fields never
    # leave the node. No source code/configs or raw metric rows are transported.
    import inspect
    prelude = "import json,math,hashlib,re,datetime as dt,subprocess\nfrom pathlib import Path\n"
    body = prelude + "FIELDS=" + repr(FIELDS) + "\n" + inspect.getsource(clean_row) + "\n" + inspect.getsource(parse_bytes) + "\n" + inspect.getsource(same_stdout_values) + "\n" + inspect.getsource(parse_stdout)
    body += "\nstate=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .State}}'," + repr(source["container"]) + "],text=True))\n"
    body += "out={'running':state['Running'],'exit_code':state['ExitCode'],'streams':{}}\n"
    body += "for stream,name in [('train','training.jsonl'),('validation','validation.jsonl')]:\n"
    body += " p=Path(" + repr(source["directory"]) + ")/name\n"
    body += " if not p.exists():\n  if not state['Running']: raise ValueError('Missing final metric file')\n  data=b''\n"
    body += " else:\n  with p.open('rb') as f: data=f.read(2*1024*1024+1)\n"
    body += " try: out['streams'][stream]=parse_bytes(data,stream)\n except (ValueError,TypeError,UnicodeError): out['streams'][stream]={'error':'invalid_complete_metric_row'}\n"
    body += "logs=subprocess.check_output(['docker','logs','--tail','5000'," + repr(source["container"]) + "],stderr=subprocess.STDOUT,timeout=20)\n"
    body += "out['stdout_rows']=parse_stdout(logs[-2*1024*1024:].decode('utf-8',errors='replace'))\nprint(json.dumps(out,allow_nan=False))\n"
    return body


def read_remote(source):
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes",
               "-o", "UserKnownHostsFile=" + KNOWN_HOSTS, HOST, "sudo", "-n", "python3", "-"]
    result = subprocess.run(command, input=remote_script(source), text=True, capture_output=True, timeout=45)
    if result.returncode:
        # Do not echo remote stderr or stdout: either could contain unexpected data.
        raise RuntimeError("Read-only metric retrieval failed (SSH/parse status %d)" % result.returncode)
    data = json.loads(result.stdout)
    if type(data.get("running")) is not bool or type(data.get("exit_code")) is not int:
        raise ValueError("Invalid container state")
    for stream in FIELDS:
        if "error" in data["streams"][stream]:
            continue
        for row in data["streams"][stream]["rows"]:
            row["payload"] = clean_row(row["payload"], stream)
    return data


def save_state(path, state):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(state, handle, allow_nan=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def ingest(state, snapshot):
    known = {(event["stream"], event["payload"]["epoch"], event["payload"]["step"]): event for event in state["events"]}
    source_rows = state.setdefault("source_rows", {})
    fresh = []
    for stream in FIELDS:
        current = snapshot["streams"][stream]
        if "error" in current:
            raise ValueError("Source metric parser rejected a complete row")
        if current["complete_bytes"] < state["cursor"].get(stream, 0):
            raise ValueError("Source metric file shrank")
        for row in current["rows"]:
            source_key = stream + ":" + str(row["offset"])
            if source_key in source_rows and source_rows[source_key] != row["sha256"]:
                raise ValueError("Previously mirrored source row changed")
            source_rows[source_key] = row["sha256"]
            key = (stream, row["payload"]["epoch"], row["payload"]["step"])
            previous = known.get(key)
            if previous:
                if previous.get("origin") == "stdout":
                    if not same_stdout_values(previous["payload"], row["payload"]):
                        raise ValueError("Buffered metrics disagree with the numeric stdout fallback")
                    previous["jsonl_confirmed"] = True
                elif previous["sha256"] != row["sha256"]:
                    raise ValueError("Previously mirrored source row changed")
            else:
                event = {"stream": stream, "origin": "jsonl", **row}
                fresh.append(event)
                known[key] = event
        state["cursor"][stream] = current["complete_bytes"]
    for row in snapshot.get("stdout_rows", []):
        payload = clean_row(row["payload"], "train")
        key = ("train", payload["epoch"], payload["step"])
        if key in known:
            if not same_stdout_values(payload, known[key]["payload"]):
                raise ValueError("Numeric stdout disagrees with the same source step")
            continue
        event = {"stream": "train", "origin": "stdout", "offset": -1, **row}
        fresh.append(event)
        known[key] = event
    fresh.sort(key=lambda row: (row["payload"]["timestamp"], row["stream"], row["offset"]))
    for event in fresh:
        event["history_step"] = len(state["events"])
        state["events"].append(event)
    return len(fresh)


def send_pending(run, state):
    resumed_step = run.step
    if resumed_step > len(state["events"]):
        raise ValueError("W&B history is ahead of the retained local journal")
    for event in state["events"]:
        if event["history_step"] < resumed_step:
            continue
        stream = event["stream"]
        payload = {stream + "/" + key: value for key, value in event["payload"].items()}
        payload["mirror/event_index"] = event["history_step"]
        payload["mirror/stdout_fallback"] = int(event.get("origin") == "stdout")
        run.log(payload, step=event["history_step"], commit=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--source-container", required=True)
    parser.add_argument("--source-directory", required=True)
    parser.add_argument("--source-run-id", default="")
    parser.add_argument("--expected-user", default="walden-lee")
    parser.add_argument("--dry-run", action="store_true", help="One read-only source poll; no W&B or local state writes")
    parser.add_argument("--poll-seconds", type=float, default=20)
    args = parser.parse_args(argv)
    source = source_identity(args.source_container, args.source_directory, args.source_run_id)
    if args.poll_seconds < 1:
        parser.error("poll-seconds must be >=1")
    if args.dry_run:
        snapshot = read_remote(source)
        if any("error" in value for value in snapshot["streams"].values()):
            raise ValueError("Source metric parser rejected a complete row")
        print(json.dumps({"running": snapshot["running"], "exit_code": snapshot["exit_code"],
                          "rows": {key: len(value["rows"]) for key, value in snapshot["streams"].items()},
                          "numeric_stdout_rows": len(snapshot.get("stdout_rows", []))}))
        return
    import fcntl
    import wandb
    if wandb.Api().viewer.username != args.expected_user:
        raise ValueError("Local W&B login is not the expected user; no run was created")
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix(args.state.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.state.exists():
            state = json.loads(args.state.read_text())
            verify_state_source(state, source)
        else:
            state = {"schema": 2, "run_id": uuid.uuid4().hex[:8], "source": source,
                     "cursor": {}, "events": [], "errors": []}
            save_state(args.state, state)
        run = wandb.init(entity="yeta", project="yeto-h200", id=state["run_id"], resume="allow",
                         name="qwen38-no-cot-metrics-mirror", settings=wandb.Settings(
                             console="off", disable_code=True, disable_git=True, x_disable_stats=True,
                             x_disable_meta=True, x_disable_machine_info=True, x_save_requirements=False),
                         config={"model": "Qwen3.8-27B", "learning_rate": 1e-5, "max_tokens": 262144,
                                 "source_run_id": args.source_run_id, "source_container": args.source_container,
                                 "is_metrics_mirror": True})
        for stream in FIELDS:
            run.define_metric(stream + "/step")
            run.define_metric(stream + "/*", step_metric=stream + "/step")
        state["url"] = run.url
        save_state(args.state, state)
        print(json.dumps({"run_id": state["run_id"], "url": run.url}), flush=True)
        try:
            while True:
                snapshot = None
                try:
                    snapshot = read_remote(source)
                    fresh = ingest(state, snapshot)
                    save_state(args.state, state)  # Durable outbox BEFORE log().
                    send_pending(run, state)
                    run.summary["source/container_running"] = int(snapshot["running"])
                    run.summary["source/logged_training_steps"] = sum(e["stream"] == "train" for e in state["events"])
                    run.summary["source/logged_validation_steps"] = sum(e["stream"] == "validation" for e in state["events"])
                    if not snapshot["running"]:
                        if any(v["complete_bytes"] != v["total_bytes"] for v in snapshot["streams"].values()):
                            raise ValueError("Exited source left an incomplete final metric row")
                        run.finish(exit_code=snapshot["exit_code"])
                        state["finished"] = True
                        state["source_exit_code"] = snapshot["exit_code"]
                        save_state(args.state, state)
                        return
                    if fresh:
                        print(json.dumps({"mirrored_events": len(state["events"]), "new_events": fresh}), flush=True)
                except (subprocess.SubprocessError, RuntimeError, ValueError, OSError, KeyError) as error:
                    record = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "category": type(error).__name__}
                    state["errors"].append(record)
                    save_state(args.state, state)
                    print(json.dumps({"mirror_error": record}), flush=True)
                    if snapshot is not None and not snapshot["running"]:
                        run.finish(exit_code=1)
                        state["finished"] = True
                        state["source_exit_code"] = snapshot["exit_code"]
                        state["final_drain_failed"] = True
                        save_state(args.state, state)
                        return
                time.sleep(args.poll_seconds)
        except BaseException:
            run.finish(exit_code=1)
            raise


if __name__ == "__main__":
    main()
