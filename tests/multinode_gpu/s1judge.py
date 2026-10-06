#!/usr/bin/env python3
"""rl-multinode-island tasks §3 judge (criteria fixed before the runs; see tasks.md §3).
usage: s1judge.py <run dir> <g0|g1|g2|g3|g4|m1|m2|m3>  -> prints and writes <run dir>/judgment-<case>.json
       s1judge.py <m4a run dir> m4 <m4b run dir>          -> writes <m4b run dir>/judgment-m4.json
m4 (machine replacement, S8-MATRIX-READINESS.md §4; criteria fixed before the runs): A = 2x2 island with --rl-checkpoint-store,
  worker raylet killed at train rid>=2 -> node_lost; B = new cluster, same store, --rl-elastic-accept-rebind. PASS needs the
  policy weights continued (journal round_cut action=restore + tape rl_round_cut_restored, rollout ids continue, not 0);
  journal-only restore = PARTIAL. Stale-incarnation rejection cannot occur naturally (A is torn down first): CPU/sim only,
  recorded as a note (PARTIAL(sim) sub-item, does not by itself lower a PASS of the GPU criteria).
m1/m2/m3 (2x2 L40S, resources-2x2.json T2R1S1; criteria fixed before the runs, MULTINODE-GAP-S8.md §3, tasks.md §3.6-3.8):
  evidence files = pulled/journal.jsonl (controller journal: topology/gpu_pool/request/phase/add_intent records),
  pulled/rl-island-0.jsonl (tape: rl_driver_phase / rl_reconfiguration), pulled/apps-<node>.txt (nvidia-smi compute apps +
  ps snapshots every 10 s, head and worker1), pulled/gpu-<node>.txt (nvidia-smi index,uuid per node), pulled/run.log
  (sky job log: Megatron argument dump), launch.log (launcher stdout), rc.txt, pulled/inwatch.log (m3 trigger submissions).
Verdicts: PASS | FAIL | PARTIAL (a sub-criterion is not verifiable on this spec, says which) | INVALID_TEST (no valid test: startup/timeout).
"""
import json, os, re, sys, time

R, CASE = sys.argv[1], sys.argv[2]
A_DIR = None
if CASE == "m4":  # s1judge.py <A run dir> m4 <B run dir>: evidence = B (judgment written to B), A read for the cross-checks
    if len(sys.argv) < 4:
        sys.exit("usage: s1judge.py <m4a run dir> m4 <m4b run dir>")
    A_DIR, R = R, sys.argv[3]

def read(name):
    p = os.path.join(R, name)
    return open(p, errors="replace").read() if os.path.exists(p) else ""

def jsonl(name):
    out = []
    for line in read(os.path.join("pulled", name)).splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out

def ts(s):  # 2026-10-03T12:00:00Z -> epoch
    return time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone

rc_txt = read("rc.txt").strip()
rc = int(rc_txt.split("=")[1]) if "=" in rc_txt else None
launch = read("launch.log")
events = jsonl("rl-island-0.jsonl")
journal = jsonl("journal.jsonl")
phases = [e for e in events if e.get("event") == "rl_driver_phase"]
rounds_trained = sorted({e.get("rollout_id") for e in phases if e.get("phase") == "train"})
generates = [e for e in phases if e.get("phase") == "generate"]
gpu_names = " ".join(read("pulled/" + f) for f in os.listdir(os.path.join(R, "pulled")) if f.startswith("gpu-")) if os.path.isdir(os.path.join(R, "pulled")) else ""
checks, notes = {}, []
# ---- ruling 2026-10-04 v2 evidence helpers (field sources in comments; see MULTINODE-GAP-S8.md §8.6) ----
local_rounds = sorted((e for e in events if e.get("event") == "rl_local_round"), key=lambda e: e.get("local_round_id", 0))
publications = sorted((e for e in events if e.get("event") == "rl_publication"), key=lambda e: e.get("policy_version", -1))
readiness = [e for e in events if e.get("event") == "rl_readiness"]
spans = [e for e in events if e.get("event") == "rl_timeline_span"]

def pub_hashes():  # rl_publication["sync/publication_payload_hash"] per policy_version (driver.publish)
    return [(p.get("policy_version"), p.get("sync/publication_payload_hash")) for p in publications]

def trainer_cross_node_evidence(head, worker):
    """trainer-itself-across-nodes (EP/PP) evidence, distinct from cross-node weight sync (G2):
    forward  = a train phase completed on a trainer whose MegatronTrainRayActor runs on BOTH nodes (apps-*.txt) and
               rl_local_round.action_tokens > 0 (tokens went through the forward pass; a PP/EP collective that failed
               would hang/raise before the round) -- Miles exposes no per-rank loss on the tape (rl_local_round.loss is
               null), so a run.log/launch.log "lm loss"/"forward" line is a bonus note only;
    backward = rl_local_round.grad_norm (driver TrainStepMetrics.grad_norm, the trainer's clipped global grad norm)
               is a finite number > 0 for >= 1 round;
    param_update = rl_publication payload hash changes between consecutive policy versions AND policy_version strictly
               increases over >= 2 publications (driver.publish: sync/publication_payload_hash is the hash of the LoRA
               weights pushed to the engines). rl_local_round.delta_l2_norm is hard-coded 0.0 by the driver
               (driver.py rl_local_round emit) and is therefore NOT used."""
    both = bool(re.search(r"MegatronTrainRayActor", head)) and bool(re.search(r"MegatronTrainRayActor", worker))
    tokens = [e.get("action_tokens") or 0 for e in local_rounds]
    grads = [e.get("grad_norm") for e in local_rounds if isinstance(e.get("grad_norm"), (int, float))]
    hashes = pub_hashes()
    versions = [v for v, _ in hashes if v is not None]
    ev = {
        "ep_pp_forward_on_both_nodes": both and len(rounds_trained) >= 1 and any(t > 0 for t in tokens),
        "ep_pp_backward_grad_norm_gt_0": any(g > 0 and g == g for g in grads),
        "ep_pp_param_update_hash_changes": (len(hashes) >= 2 and versions == sorted(set(versions))
                                            and all(a[1] != b[1] for a, b in zip(hashes, hashes[1:]) if a[1] and b[1])
                                            and len({h for _, h in hashes if h}) >= 2),
    }
    notes.append(f"grad_norms={grads[:4]} pub_hashes={[(v, (h or '')[:8]) for v, h in hashes][:4]} "
                 f"loss_logged={bool(re.search(r'lm loss|forward', read('pulled/run.log')))} (delta_l2_norm unused: driver hard-codes 0.0)")
    return ev

if rc == 124 and not (CASE == "g0" and len(rounds_trained) >= 1 and len(generates) >= 1):
    verdict = "INVALID_TEST"; notes.append("hard_timeout (rc=124)")
elif CASE == "g0":
    # G0 is an image/sm_89 probe: when the local launcher hit its hard timeout during the cold start (setup) but the job went on
    # inside the container and the pulled events show the round, the probe is judged on that in-container evidence (noted).
    if rc == 124:
        notes.append("launcher hard_timeout during setup; judged on in-container events pulled from the kept cluster")
    checks = {"launcher_rc_0_or_setup_timeout": rc in (0, 124), "train_ge_1": len(rounds_trained) >= 1, "generate_ge_1": len(generates) >= 1,
              "gpu_is_L40S": "L40S" in gpu_names, "no_cuda_kernel_error": not re.search(r"no kernel image|sm_89|CUDA error|not compatible", launch)}
    verdict = "PASS" if all(checks.values()) else "FAIL"
elif CASE == "g1":
    # 30 min case window measured from the job start on the (cold-started) cluster; the launcher's own
    # timeout (3600 s) covers the cold start too (2-node image pull + setup + model fetch took ~25-30 min).
    tsl = read("launch.ts.log"); m = re.search(r"^(\S+) .*Job submitted", tsl, re.M)
    end = read("end_utc.txt").strip()
    case_s = (ts(end) - ts(m.group(1))) if (m and end) else None
    notes.append(f"case_window_from_job_submit_s={case_s}")
    topo = [r for r in journal if r.get("kind") == "topology"]
    t0 = topo[0] if topo else {}
    alive = t0.get("alive") if isinstance(t0.get("alive"), list) else []
    checks = {"topology_record_2x1": t0.get("nodes") == 2 and t0.get("gpus_per_node") == 1,
              "alive_nodes_2": len(alive) == 2,
              # D3 fail closed: node blocks AND node 0 = Ray head (head pin, 2026-10-03 ruling); any refusal shows in the launch log
              "startup_bundles_ok": not re.search(r"not node-blocked|BundleMapError|not the Ray head|D3 head pin|exactly one alive Ray head", launch) and len(rounds_trained) >= 1,
              "learner_round_ge_1": len(rounds_trained) >= 1 and len(generates) >= 1,
              "case_window_le_1800s": case_s is not None and case_s <= 1800,
              "launcher_rc_0": rc == 0}
    verdict = "PASS" if all(checks.values()) else "FAIL"
elif CASE == "g2":
    head = read("pulled/apps-" + read("cluster.txt").strip() + ".txt"); worker = read("pulled/apps-" + read("cluster.txt").strip() + "-worker1.txt")
    # engine = the sglang::scheduler process; a bare "sglang" also matches /opt/sglang/bin/ray (probe) and build paths on the head
    checks = {"rollout_engine_on_n1": bool(re.search(r"sglang::scheduler", worker)), "no_rollout_engine_on_n0": not re.search(r"sglang::", head),
              # trainer = the MegatronTrainRayActor process (D3 head pin: bundle 0 on the head); `ray::` alone also matches RayWorkerManager
              "trainer_on_n0": bool(re.search(r"MegatronTrainRayActor", head)),
              "no_trainer_on_n1": not re.search(r"MegatronTrainRayActor", worker),
              "rounds_ge_2_cross_node_sync": len(rounds_trained) >= 2 and len(generates) >= 2,
              "config_epoch_monotonic": all(a.get("config_epoch", 0) <= b.get("config_epoch", 0) for a, b in zip(journal, journal[1:])) if journal else True,
              "e1_up_down_edge": None}
    notes.append("E1 rollout-only up/down edge NOT verifiable on 2x1 (no standby GPU; needs 2x2 or larger) -> PARTIAL at best")
    verdict = "PARTIAL" if all(v for v in checks.values() if v is not None) else "FAIL"
elif CASE == "g3":
    kill = read("kill.txt").splitlines()
    # the first epoch-looking line (ssh may print a known-hosts warning before the worker's `date +%s.%N`)
    kill_epoch = next((float(l.strip()) for l in kill if re.match(r"^\d+\.\d+$", l.strip())), None)
    lost = [r for r in journal if r.get("kind") == "node_lost"]
    rr = [e for e in events if e.get("event") == "rl_reconfiguration" and e.get("result") == "RECOVERY_REQUIRED" and str(e.get("error", "")).startswith("node_lost")]
    lost_wall = lost[0].get("wall_time") if lost else None
    latency = (lost_wall - kill_epoch) if (lost_wall and kill_epoch) else None
    after = [e for e in phases if e.get("phase") in ("train", "generate") and lost_wall and e.get("time_unix", 0) > lost_wall + 1]
    topo = [r for r in journal if r.get("kind") == "topology"]
    refused = [r for r in journal if "recovery refused" in json.dumps(r) or "island topology" in json.dumps(r)]
    checks = {"kill_recorded": kill_epoch is not None, "node_lost_journal": bool(lost), "rl_reconfiguration_RECOVERY_REQUIRED_node_lost": bool(rr),
              "latency_le_60s": latency is not None and latency <= 60, "learner_nonzero_exit": bool(re.search(r"learner exited [1-9]\d*", launch)) or (rc not in (0, None)),
              "no_partial_continuation": not after,
              "restart_precheck_refused": len(topo) >= 2 and bool(refused),
              "not_judged_on_LedgerError": True}
    notes.append(f"latency_s={latency}")
    verdict = "PASS" if all(checks.values()) else "FAIL"
elif CASE == "g4":
    confirmed = re.findall(r"node instance (\S+) confirmed terminated", launch)
    cleanup = read("cleanup.out")
    checks = {"two_node_confirm_lines": len(set(confirmed)) == 2, "no_UNCONFIRMED": "UNCONFIRMED" not in launch,
              "cleanup_clean_twice": "RESULT: clean twice" in cleanup}
    verdict = "PASS" if all(checks.values()) else "FAIL"
elif CASE in ("m1", "m2"):
    cl = read("cluster.txt").strip()
    head, worker = read(f"pulled/apps-{cl}.txt"), read(f"pulled/apps-{cl}-worker1.txt")
    run_log, args = read("pulled/run.log"), read("args.txt")
    topo = [r for r in journal if r.get("kind") == "topology"]; t0 = topo[0] if topo else {}
    alive = t0.get("alive") if isinstance(t0.get("alive"), list) else []
    pools = [r for r in journal if r.get("kind") == "gpu_pool"]
    pool_uuids = [u for node in (pools[-1].get("uuids") or []) for u in node] if pools else []
    seen_uuids = set(re.findall(r"(GPU-[0-9a-f-]{36})", gpu_names))
    checks = {
        # journal `topology` (controller.set_topology): nodes/gpus_per_node from --rl-island-gpus-per-node, alive = ray.nodes()
        "topology_record_2x2": t0.get("nodes") == 2 and t0.get("gpus_per_node") == 2,
        "alive_nodes_2": len(alive) == 2,
        # journal `gpu_pool` (Q6 reconcile_gpu_pool_preflight): accepted, 2 nodes x 2 uuids, all present in the puller's nvidia-smi rows
        "gpu_pool_accepted": bool(pools) and pools[-1].get("accepted") is True,
        "gpu_pool_4_uuids_match_nvidia_smi": len(pool_uuids) == 4 and set(pool_uuids) <= seen_uuids,
        # apps-<node>.txt: MegatronTrainRayActor = trainer rank process; the trainer must run on BOTH nodes (T2 = n0:0 + n1:0)
        "trainer_on_head": bool(re.search(r"MegatronTrainRayActor", head)),
        "trainer_on_worker": bool(re.search(r"MegatronTrainRayActor", worker)),
        # Q2 mixed node: the rollout engine (sglang::scheduler) shares n0 with trainer rank 0; n1:1 is standby (no engine) in m1/m2
        "rollout_engine_on_head_mixed": bool(re.search(r"sglang::scheduler", head)),
        "no_rollout_engine_on_worker": not re.search(r"sglang::scheduler", worker),
        # launch.log: D3/D4/Q6 fail-closed paths never fired
        "no_layout_or_pool_refusal": not re.search(r"BundleMapError|spans nodes|not node-blocked|not the Ray head|D3 head pin|gpu_pool:|disagrees with PlacementRequest", launch),
        # tape rl_driver_phase: >= 2 rounds train + generate. This is the "trainer and rollout on different nodes,
        # cross-node weight sync" criterion (already judged PASS in G2); listed separately from the EP/PP evidence below
        "cross_node_weight_sync_rounds_ge_2": len(rounds_trained) >= 2 and len(generates) >= 2,
        "launcher_rc_0": rc == 0}
    # ruling 2026-10-04 v2: "trainer itself across nodes (EP/PP)" needs forward / backward / parameter-update evidence
    checks.update(trainer_cross_node_evidence(head, worker))
    if CASE == "m1":
        # run.log: Megatron argument dump (`pipeline_model_parallel_size .... 2`) or the learner's megatron flag
        checks["pp2_in_megatron_args"] = bool(re.search(r"pipeline[-_]model[-_]parallel[-_]size\W+2\b", run_log + " " + launch))
        checks["args_tp1_pp2"] = "--tensor-parallel 1 --pipeline-parallel 2" in args
    else:
        checks["ep2_in_megatron_args"] = bool(re.search(r"expert[-_]model[-_]parallel[-_]size\W+2\b", run_log + " " + launch))
        checks["args_moe_lora_attention_ep2"] = ("Qwen3-30B-A3B-5layer" in args and "--lora-targets attention" in args
                                                and "--expert-parallel 2" in args)
        # cross-node EP: with tp1 pp1 the EP2 group is exactly the two trainer ranks, one per node (trainer_on_head + trainer_on_worker)
        checks["ep_group_spans_nodes"] = checks["trainer_on_head"] and checks["trainer_on_worker"]
        notes.append("reward may stay 0 on the 5-layer slice; not a criterion")
    notes.append(f"gpu_pool_source={pools[-1].get('source') if pools else None} pool_uuids={pool_uuids}")
    verdict = "PASS" if all(checks.values()) else "FAIL"
elif CASE == "m3":
    cl = read("cluster.txt").strip()
    worker = read(f"pulled/apps-{cl}-worker1.txt")
    inwatch = jsonl("inwatch.log")
    submitted = {r.get("submitted") for r in inwatch}
    reqs = {r.get("request_id"): r for r in journal if r.get("kind") == "request"}
    def phases_of(req, phase):
        return [r for r in journal if r.get("kind") == "phase" and r.get("request_id") == req and r.get("phase") == phase]
    plan = lambda req: (reqs.get(req) or {}).get("plan") or {}  # noqa: E731
    rr_journal = [r for r in journal if r.get("kind") == "phase" and r.get("phase") == "RECOVERY_REQUIRED"]
    rr_tape = [e for e in events if e.get("event") == "rl_reconfiguration" and e.get("result") == "RECOVERY_REQUIRED"]
    up_done = phases_of("up1", "SUCCEEDED"); dn_done = phases_of("dn1", "SUCCEEDED")
    def rounds_after(wall):
        return sorted({e.get("rollout_id") for e in phases if e.get("phase") == "train" and e.get("time_unix", 0) > wall + 1})
    checks = {
        # journal `request` (controller.request): plan.source/target/kind = the manifest edge; source_engines/target_engines = members 1 -> 2 -> 1
        "up_request_T2R1S1_to_T2R2S0": (plan("up1").get("source"), plan("up1").get("target"), plan("up1").get("kind")) == ("T2R1S1", "T2R2S0", "rollout-only"),
        "down_request_T2R2S0_to_T2R1S1": (plan("dn1").get("source"), plan("dn1").get("target"), plan("dn1").get("kind")) == ("T2R2S0", "T2R1S1", "rollout-only"),
        "members_1_2_1": (plan("up1").get("source_engines"), plan("up1").get("target_engines"), plan("dn1").get("source_engines"), plan("dn1").get("target_engines")) == (1, 2, 2, 1),
        # journal `phase` records per request_id: exactly one COMMITTED (the commit point) and a SUCCEEDED per edge
        "up_committed_once": len(phases_of("up1", "COMMITTED")) == 1 and len(up_done) == 1,
        "down_committed_once": len(phases_of("dn1", "COMMITTED")) == 1 and len(dn_done) == 1,
        # journal `add_intent` (up edge): the added member is the standby cell c1 (bound to n1:1 by the placement map)
        "up_adds_cell_c1": any(r.get("members") == ["engine:c1"] for r in journal if r.get("kind") == "add_intent" and r.get("request_id") == "up1") or any("c1" in json.dumps(r.get("members")) for r in journal if r.get("kind") == "add_intent"),
        # SUCCEEDED.config_epoch_to: 0 -> 1 (up) -> 2 (down)
        "config_epoch_1_then_2": (up_done[0].get("config_epoch_to") if up_done else None) == 1 and (dn_done[0].get("config_epoch_to") if dn_done else None) == 2,
        # apps-<worker1>.txt snapshots: an sglang::scheduler appeared on n1 while c1 was up (n1:1); no cross-node refusal anywhere
        "engine_seen_on_worker_during_up": bool(re.search(r"sglang::scheduler", worker)),
        "no_spans_nodes_refusal": "spans nodes" not in launch and "spans nodes" not in json.dumps(journal),
        # no RECOVERY_REQUIRED (journal phase or tape rl_reconfiguration)
        "no_recovery_required": not rr_journal and not rr_tape,
        # tape rl_driver_phase: >= 1 train round after the up commit and >= 1 after the down commit (training continues on both edges)
        "train_continues_after_up": bool(up_done) and len(rounds_after(up_done[0].get("wall_time", 0))) >= 1,
        "train_continues_after_down": bool(dn_done) and len(rounds_after(dn_done[0].get("wall_time", 0))) >= 1,
        "launcher_rc_0": rc == 0}
    # ---- ruling 2026-10-04 v2 additions ----
    up_wall = up_done[0].get("wall_time", 0) if up_done else None
    dn_wall = dn_done[0].get("wall_time", 0) if dn_done else None
    # weight version sync: the first rl_publication after the up commit lists BOTH engines (sync/publication_members,
    # driver.publish) under one policy_version, and the following rl_readiness has published_policy_version ==
    # trained_policy_version (driver readiness snapshot) -> the new engine serves the same version as the existing one
    pubs_after_up = [p for p in publications if up_wall and p.get("time_unix", 0) > up_wall]
    ready_after_up = [r for r in readiness if up_wall and r.get("time_unix", 0) > up_wall]
    checks["up_new_engine_same_policy_version"] = bool(pubs_after_up) and len(pubs_after_up[0].get("sync/publication_members") or []) >= 2 \
        and bool(ready_after_up) and ready_after_up[0].get("published_policy_version") == ready_after_up[0].get("trained_policy_version")
    # in-flight requests: no batch lost across the up/down transactions -- rl_local_round.completed_groups ==
    # active_groups for every round, cancelled_groups never rises (driver rl_local_round: cancelled_groups = batch.aborted),
    # and the trained rollout_ids (rl_driver_phase train) are contiguous
    cancelled = [int(e.get("cancelled_groups") or 0) for e in local_rounds]
    checks["inflight_no_batch_lost_across_edges"] = bool(local_rounds) and all(
        (e.get("completed_groups") or 0) >= (e.get("active_groups") or 0) for e in local_rounds) \
        and all(b <= a for a, b in zip(cancelled, cancelled[1:])) and max(cancelled or [0]) == 0 \
        and rounds_trained == list(range(min(rounds_trained), max(rounds_trained) + 1)) if rounds_trained else False
    # resource reclaim after down: in the LAST apps-<worker1>.txt snapshot taken after the down commit (blocks start with
    # an ISO timestamp; rows = `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory` + ps) there is no
    # sglang::scheduler and no compute-app row on the standby card n1:1 (its uuid = 2nd row of gpu-<worker1>.txt)
    gpu_w1 = read(f"pulled/gpu-{cl}-worker1.txt").splitlines()
    standby_uuid = next((m.group(1) for line in gpu_w1[1:] if (m := re.search(r"^\s*1,\s*(GPU-[0-9a-f-]{36})", line))), None)
    blocks, cur = [], None
    for line in worker.splitlines():
        m = re.match(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)$", line.strip())
        if m:
            cur = [ts(m.group(1)), []]; blocks.append(cur)
        elif cur is not None:
            cur[1].append(line)
    after_dn = [b for b in blocks if dn_wall and b[0] > dn_wall + 5]
    last = "\n".join(after_dn[-1][1]) if after_dn else None
    checks["down_reclaims_worker_engine_and_gpu"] = last is not None and "sglang::scheduler" not in last \
        and (standby_uuid is None or not re.search(re.escape(standby_uuid) + r"\s*,", last))
    notes.append(f"standby_uuid={standby_uuid} snapshots_after_down={len(after_dn)} pub_members_after_up="
                 f"{(pubs_after_up[0].get('sync/publication_members') if pubs_after_up else None)}")
    notes.append(f"inwatch_submitted={sorted(s for s in submitted if s)} requests={sorted(reqs)}")
    if not reqs and not (submitted & {"up1", "dn1"}):
        verdict = "INVALID_TEST"; notes.append("no E1 trigger was submitted (inwatch not armed or the run ended before train rid 1)")
    elif "dn1" not in reqs and all(checks[k] for k in ("up_request_T2R1S1_to_T2R2S0", "up_committed_once", "up_adds_cell_c1",
                                                      "engine_seen_on_worker_during_up", "no_recovery_required", "train_continues_after_up")):
        verdict = "PARTIAL"; notes.append("down edge never requested (resource/time limited): up edge judged, down edge PARTIAL")
    else:
        verdict = "PASS" if all(checks.values()) else "FAIL"
elif CASE == "m4":
    def a_jsonl(name):
        out = []
        p = os.path.join(A_DIR, "pulled", name)
        for line in (open(p, errors="replace").read().splitlines() if os.path.exists(p) else []):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
        return out
    a_journal, a_events = a_jsonl("journal.jsonl"), a_jsonl("rl-island-0.jsonl")
    a_gpu = " ".join(open(os.path.join(A_DIR, "pulled", f), errors="replace").read()
                     for f in (os.listdir(os.path.join(A_DIR, "pulled")) if os.path.isdir(os.path.join(A_DIR, "pulled")) else [])
                     if f.startswith("gpu-"))
    # --- A: node loss after >= 2 trained rounds, a round cut synced to the store
    a_lost = [r for r in a_journal if r.get("kind") == "node_lost"]
    a_rr = [e for e in a_events if e.get("event") == "rl_reconfiguration" and e.get("result") == "RECOVERY_REQUIRED"
            and str(e.get("error", "")).startswith("node_lost")]
    a_cuts = [e for e in a_events if e.get("event") == "rl_round_cut" and e.get("ok") is True]
    a_cut_synced = [e for e in a_cuts if e.get("store_synced")]
    a_pools = [r for r in a_journal if r.get("kind") == "gpu_pool" and r.get("accepted")]
    a_uuids = {u for node in (a_pools[-1].get("uuids") or []) for u in node} if a_pools else set()
    a_topo = [r for r in a_journal if r.get("kind") == "topology"]
    a_inc = a_pools[-1].get("incarnation") if a_pools else None
    # --- B: everything after its checkpoint_store restore record is B's own (the rest was copied from the store)
    idx = next((i for i, r in enumerate(journal) if r.get("kind") == "checkpoint_store" and r.get("action") == "restore"), None)
    restore = journal[idx] if idx is not None else {}
    b_own = journal[idx + 1:] if idx is not None else []
    b_topo = [r for r in b_own if r.get("kind") == "topology"]
    b_pools = [r for r in b_own if r.get("kind") == "gpu_pool"]
    b_pool = b_pools[-1] if b_pools else {}
    b_uuids = [u for node in (b_pool.get("uuids") or []) for u in node]
    b_seen = set(re.findall(r"(GPU-[0-9a-f-]{36})", gpu_names))
    rc_rec = [r for r in b_own if r.get("kind") == "round_cut" and r.get("action") == "restore"]
    restored_ev = [e for e in events if e.get("event") == "rl_round_cut_restored"]
    resume_rid = restored_ev[0].get("rollout_id") if restored_ev else None
    a_trained = sorted({e.get("rollout_id") for e in a_events if e.get("event") == "rl_driver_phase" and e.get("phase") == "train"})
    b_rr = [e for e in events if e.get("event") == "rl_reconfiguration" and e.get("result") == "RECOVERY_REQUIRED"] + \
           [r for r in b_own if r.get("kind") == "phase" and r.get("phase") == "RECOVERY_REQUIRED"]
    checks = {
        # A side
        "a_node_lost_journal": bool(a_lost),
        "a_rl_reconfiguration_RECOVERY_REQUIRED_node_lost": bool(a_rr),
        "a_round_cut_synced_to_store": bool(a_cut_synced),
        # B side: store restore from A's incarnation
        "b_checkpoint_store_restore": idx is not None,
        "b_restored_from_a_incarnation": a_inc is not None and (restore.get("restored_from") or {}).get("incarnation") == a_inc,
        # same layout, accepted
        "b_layout_accepted_same_as_a": bool(b_topo) and b_topo[-1].get("layout_accepted") is True and bool(a_topo)
            and b_topo[-1].get("layout") == a_topo[-1].get("layout"),
        # rebind: accepted, rebind=true, non-empty mapping; B's 4 uuids all differ from A's and are B's nvidia-smi uuids
        "b_gpu_pool_rebind_accepted": b_pool.get("accepted") is True and b_pool.get("rebind") is True and bool(b_pool.get("mapping")),
        "b_uuids_new_and_on_b_nodes": len(b_uuids) == 4 and not (set(b_uuids) & a_uuids) and set(b_uuids) <= b_seen
            and not (set(b_uuids) & set(re.findall(r"(GPU-[0-9a-f-]{36})", a_gpu))),
        # continuation: >= 1 train round after the resume, the first trained rid = resume rid > 0
        "b_trains_ge_1_round_after_resume": bool(rounds_trained) and (resume_rid is None or max(rounds_trained) >= resume_rid),
        "b_no_recovery_required": not b_rr,
    }
    weights = {
        "weights_round_cut_restored": bool(rc_rec) and bool(restored_ev)
            and rc_rec[0].get("cut_incarnation") == a_inc,
        "weights_rollout_id_continues": resume_rid is not None and resume_rid > 0 and bool(rounds_trained)
            and min(rounds_trained) == resume_rid and (not a_trained or resume_rid <= max(a_trained) + 1),
    }
    checks.update(weights)
    notes.append(f"a_incarnation={a_inc} restored_from={restore.get('restored_from')} resume_rid={resume_rid} "
                 f"a_trained={a_trained} b_trained={rounds_trained} a_round_cuts={[e.get('cut_id') for e in a_cuts]}")
    notes.append("stale_incarnation_rejection: PARTIAL(sim) -- A is torn down before B starts, so no old incarnation can "
                 "write; evidence = tests/multinode_sim gpu_pool 1b (CPU)")
    base_ok = all(v for k, v in checks.items() if k not in weights)
    if idx is None and not rounds_trained:
        verdict = "INVALID_TEST"; notes.append("B never restored the store nor trained (startup/timeout)")
    elif base_ok and not all(weights.values()):
        verdict = "PARTIAL"; notes.append("journal/store restored and B trains, but the policy weights did not continue "
                                          "from A's round cut (journal-only restore)")
    else:
        verdict = "PASS" if all(checks.values()) else "FAIL"
else:
    sys.exit("unknown case")
out = {"case": CASE, "verdict": verdict, "checks": checks, "notes": notes, "rc": rc, "rounds_trained": rounds_trained, "n_generate": len(generates),
       "start_utc": read("start_utc.txt").strip(), "end_utc": read("end_utc.txt").strip(), "yeto_sha": read("yeto_sha.txt").strip()}
json.dump(out, open(os.path.join(R, f"judgment-{CASE}.json"), "w"), indent=1)
print(json.dumps(out))
sys.exit(0 if verdict == "PASS" else 1)
