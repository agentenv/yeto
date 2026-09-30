"""GPU acceptance probes for rl-engine-ports (installed by monkeypatch; not yeto code)."""
import json, os, sys, time, math, dataclasses
OUT = os.environ.get("PROBE_OUT", "/work/out/probe.jsonl")
MODE = os.environ.get("PROBE_MODE", "normal")

def log(kind, **kw):
    rec = {"probe": kind, "t": time.time(), **kw}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print("[probe]", json.dumps(rec, default=str)[:2000], flush=True)

# ---- plugins run inside Megatron ranks via run_plugin -------------------
def moments(actor):
    import torch
    from yeto.rl.engine.miles_adapter.state_plugin import _optimizer_children
    tot = {"exp_avg": 0.0, "exp_avg_sq": 0.0, "n_state": 0, "steps": []}
    def leaves(opt):
        for attr in ("chained_optimizers", "sub_optimizers"):
            if hasattr(opt, attr):
                for c in getattr(opt, attr):
                    yield from leaves(c)
                return
        inner = getattr(opt, "optimizer", None)
        yield from (leaves(inner) if inner is not None else (opt,))
    for leaf in leaves(actor.optimizer):
        for st in leaf.state.values():
            tot["n_state"] += 1
            for k in ("exp_avg", "exp_avg_sq"):
                if k in st:
                    tot[k] += float(st[k].detach().float().abs().sum().item())
            if "step" in st:
                s = st["step"]; tot["steps"].append(float(s.item() if hasattr(s, "item") else s))
        for g in getattr(leaf, "param_groups", []):
            if "step" in g:
                tot["steps"].append(float(g["step"]))
    sch = getattr(actor, "opt_param_scheduler", None)
    tot["scheduler_num_steps"] = getattr(sch, "num_steps", None)
    tot["steps"] = sorted(set(tot["steps"]))
    return tot

def zero_grad_norm(actor):
    return 0.0

def raw_lora_names(actor):
    names = [n for chunk in actor.model for n, p in chunk.named_parameters() if p.requires_grad]
    return names[:6] + [len(names)]

# ---- driver-side patches -------------------------------------------------
def _scan_payload(obj, depth=0, acc=None):
    import torch
    acc = acc if acc is not None else {"tensors": 0, "tensor_bytes": 0, "types": set()}
    acc["types"].add(type(obj).__name__)
    if depth > 6: return acc
    if isinstance(obj, torch.Tensor):
        acc["tensors"] += 1; acc["tensor_bytes"] += obj.numel() * obj.element_size()
    elif isinstance(obj, dict):
        for v in obj.values(): _scan_payload(v, depth + 1, acc)
    elif isinstance(obj, (list, tuple)):
        for v in list(obj)[:10000]: _scan_payload(v, depth + 1, acc)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj): _scan_payload(getattr(obj, f.name), depth + 1, acc)
    return acc

def install():
    import hashlib
    from yeto.rl.engine import driver as D
    from yeto.rl.engine.miles_adapter import trainer as T, publish as P
    orig_gen, orig_round, orig_train, orig_pub = D.IslandDriver._generate, D.IslandDriver.run_round, T.MilesTrainerGroup.train_step, P.MilesPublisher.publish

    def _generate(self, rollout_id):
        batch = orig_gen(self, rollout_id)
        payload = batch.payload
        scan = _scan_payload(batch.groups)
        pscan = _scan_payload(payload)
        log("rollout_handle", rollout_id=rollout_id, payload_type=type(payload).__name__,
            payload_repr=repr(payload)[:300], payload_scan={**pscan, "types": sorted(pscan["types"])},
            groups=len(batch.groups), group_scan={**scan, "types": sorted(scan["types"])},
            policy_hash=batch.policy_hash, expected_token=self.expected_token,
            group_tokens=sorted({g.policy_token for g in batch.groups}),
            rewards=[(g.reward_mean, g.reward_std) for g in batch.groups])
        return batch

    def train_step(self, batch):
        r = orig_train(self, batch)
        log("receipt", rollout_id=batch.rollout_id, receipt=dataclasses.asdict(r),
            empty_fields=[k for k, v in dataclasses.asdict(r).items() if v in (None, "", (), [])],
            grad_norm=self.last_grad_norm)
        return r

    def publish(self, state, **kw):
        res = orig_pub(self, state, **kw)
        m = res.manifest
        ck = self.last_engine_checksums
        tensor_hash = state.policy_tensor_hash()
        pay_hash, _ = P.payload_digest(state)
        recomputed = hashlib.sha256(json.dumps({
            "token": P.policy_token(state.policy_version if kw.get("token_rollout_id") is None else kw["token_rollout_id"], tensor_hash),
            "policy_tensor_hash": tensor_hash, "payload_hash": pay_hash,
            "members": sorted(res.members), "engine_checksums": ck},
            sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        engine_vals = {e: hashlib.sha256(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest() for e, v in (ck or {}).items()}
        log("publication", policy_version=state.policy_version, manifest=dataclasses.asdict(m),
            members=sorted(res.members), engine_checksum_digest=engine_vals,
            engines_agree=len(set(engine_vals.values())) <= 1 and bool(engine_vals),
            manifest_hash_matches_engine_checksums=(recomputed == m.target_manifest_hash),
            checksum_sample={e: dict(list(v.items())[:3]) for e, v in (ck or {}).items()})
        return res

    def run_round(self, rollout_id):
        b = orig_round(self, rollout_id)
        if MODE == "normal" and rollout_id == 0:
            ps = self.policy_state
            run = ps._run
            log("raw_lora_names", names=run(self.trainer._actor.run_plugin("probe.raw_lora_names", {})))
            s1 = ps.export()
            before = run(self.trainer._actor.run_plugin("probe.moments", {}))
            ps.apply(s1, optimizer="preserve", local_step=rollout_id + 1)
            s_pres = ps.export()
            mid = run(self.trainer._actor.run_plugin("probe.moments", {}))
            ps.apply(s1, optimizer="reset", local_step=rollout_id + 1)
            s2 = ps.export()
            after = run(self.trainer._actor.run_plugin("probe.moments", {}))
            names = list(s1.tensor_names)
            log("export_apply_export", h1=s1.policy_tensor_hash(), h_preserve=s_pres.policy_tensor_hash(),
                h2=s2.policy_tensor_hash(), equal=s1.policy_tensor_hash() == s2.policy_tensor_hash() == s_pres.policy_tensor_hash(),
                boundary_hash=b.state.policy_tensor_hash(), n_tensors=len(names), names_sample=names[:4],
                canonical_prefix=all(n.startswith("base_model.model.") for n in names),
                moments_before=before, moments_after_preserve=mid, moments_after_reset=after)
        return b

    D.IslandDriver._generate = _generate
    D.IslandDriver.run_round = run_round
    T.MilesTrainerGroup.train_step = train_step
    P.MilesPublisher.publish = publish
    if MODE == "zerograd":
        T.GRAD_NORM = "probe.zero_grad_norm"
        log("inject", what="GRAD_NORM plugin -> probe.zero_grad_norm (returns 0.0) for every round")
    if MODE == "badtoken":
        from yeto.rl.engine.miles_adapter import rollout as R
        orig_rgen = R.MilesRolloutPool.generate
        def rgen(self, rollout_id):
            b = orig_rgen(self, rollout_id)
            if rollout_id == 1:
                g0 = b.groups[0]
                bad = dataclasses.replace(g0, policy_token="yeto:0:" + "0" * 64)
                b = dataclasses.replace(b, groups=(bad,) + tuple(b.groups[1:]))
                log("inject", what="rollout 1 group %s policy_token -> stale token" % g0.group_id)
            return b
        R.MilesRolloutPool.generate = rgen
    if MODE == "pubfail":
        orig_inner = P.MilesPublisher._publish
        class _BadEngine:
            def __init__(self, e): self._e = e
            def __getattr__(self, k): return getattr(self._e, k)
            async def update_weight_version(self, token):
                raise RuntimeError("injected: engine refused weight version " + token)
        class _Ctl:
            def __init__(self, c): self._c = c
            def __getattr__(self, k): return getattr(self._c, k)
            async def start_update_weights(self, *a, **kw):
                info = await self._c.start_update_weights(*a, **kw)
                if a or kw:
                    return info
                engines = list(info.rollout_engines)
                try:
                    info.rollout_engines = [_BadEngine(engines[0])] + engines[1:]
                except Exception:
                    info = dataclasses.replace(info, rollout_engines=[_BadEngine(engines[0])] + engines[1:])
                return info
        async def _publish(self, token):
            if token.startswith("yeto:1:"):
                log("inject", what="publication v1: engine 0 update_weight_version raises")
                real = self._controller; self._controller = _Ctl(real)
                try:
                    return await orig_inner(self, token)
                except BaseException as e:
                    log("publish_error", error=repr(e)[:500]); raise
                finally:
                    self._controller = real
            return await orig_inner(self, token)
        P.MilesPublisher._publish = _publish
    orig_emit = D.IslandDriver.emit
    def emit(self, event, **f):
        return orig_emit(self, event, **f)
    log("installed", mode=MODE)
