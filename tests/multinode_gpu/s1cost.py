#!/usr/bin/env python3
"""s1cost: read-only extraction of E1 (rollout-edge) transition cost segments.

Usage: s1cost.py RUN_DIR [RUN_DIR ...] [--json OUT]
RUN_DIR contains pulled/journal.jsonl and pulled/rl-island-0.jsonl.
Segments per request (seconds, journal monotonic clock):
  wait_safe  WAIT_SAFE -> QUIESCING      (safe-point wait)
  drain      QUIESCING -> INITIALIZING/TRANSFERRING
  init       fork_op issued -> done      (engine start / stop)
  verify     VERIFYING -> COMMITTED      (member engines, lora warmup+readback = publish/restore of adapter)
  resume     COMMITTED -> SUCCEEDED
  blocking   request -> SUCCEEDED
  first_step (generate+train of first round after SUCCEEDED) - steady median of other rounds
export/restore of trainer state: not applicable on rollout-only edges (reported as null).
"""
import json, os, statistics, sys


def _load(p):
    out = []
    for line in open(p):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def analyze(run):
    pul = os.path.join(run, 'pulled')
    j = _load(os.path.join(pul, 'journal.jsonl'))
    isl = _load(os.path.join(pul, 'rl-island-0.jsonl'))
    profile = next((r['plan'].get('profile_hash') for r in j if r.get('kind') == 'request'
                    and isinstance(r.get('plan'), dict)), None)
    spans = {}
    for r in isl:
        if r.get('event') == 'rl_timeline_span' and r.get('rollout_id') is not None:
            spans.setdefault(r['rollout_id'], 0.0)
            if r['task'] in ('generate', 'train'):
                spans[r['rollout_id']] += r['end'] - r['start']
    recon_rids = {r['rollout_id'] for r in isl if r.get('event') == 'rl_reconfiguration'}
    steady = [v for k, v in spans.items() if k not in recon_rids and k != 0]
    steady_med = statistics.median(steady) if steady else None
    reqs = {}
    for r in j:
        tx = r.get('tx_id')
        if not tx:
            continue
        d = reqs.setdefault(tx, {'ph': {}})
        if r['kind'] == 'request':
            d.update(rid=r['request_id'], target=r['body']['target'], t_req=r['monotonic'])
        elif r['kind'] == 'phase':
            d['ph'].setdefault(r['phase'], r['monotonic'])
        elif r['kind'] == 'fork_op':
            d['fork_' + r['status']] = r['monotonic']
        elif r['kind'] == 'pause_decision':
            d['safe_rid'] = r.get('rollout_id')
    rows = []
    prev_target = None
    for tx, d in sorted(reqs.items(), key=lambda kv: kv[1].get('t_req', 0)):
        ph = d['ph']
        g = lambda a, b: (round(b - a, 3) if a is not None and b is not None else None)
        q = ph.get('QUIESCING')
        nxt = ph.get('INITIALIZING', ph.get('TRANSFERRING'))
        rid = d.get('safe_rid')
        fs = spans.get(rid)
        source = prev_target or ('T2R1S1' if d.get('target') == 'T2R2S0' else 'T2R2S0')
        rows.append(dict(
            run=os.path.basename(run.rstrip('/')), request_id=d.get('rid'), source=source,
            target=d.get('target'), profile_hash=profile,
            wait_safe=g(ph.get('WAIT_SAFE'), q), drain=g(q, nxt),
            init=g(d.get('fork_issued'), d.get('fork_done')),
            verify=g(ph.get('VERIFYING'), ph.get('COMMITTED')),
            resume=g(ph.get('COMMITTED'), ph.get('SUCCEEDED')),
            export=None, restore=None,
            blocking=g(d.get('t_req'), ph.get('SUCCEEDED')),
            first_step_extra=(round(fs - steady_med, 3) if fs is not None and steady_med else None),
            steady_round_s=(round(steady_med, 3) if steady_med else None),
            succeeded='SUCCEEDED' in ph))
        prev_target = d.get('target')
    return rows


def sweep(run):
    """2.4 fixed-config run: validity (rc0, rounds == --total-steps, L40S x4 uuids, no RECOVERY_REQUIRED)
    and per-round wall = generate+train+outer_sync of rid + the publish that follows it; median over rid>=1."""
    pul = os.path.join(run, 'pulled')
    rd = lambda n: open(os.path.join(run, n)).read() if os.path.exists(os.path.join(run, n)) else ''
    isl = _load(os.path.join(pul, 'rl-island-0.jsonl')) if os.path.exists(os.path.join(pul, 'rl-island-0.jsonl')) else []
    j = _load(os.path.join(pul, 'journal.jsonl')) if os.path.exists(os.path.join(pul, 'journal.jsonl')) else []
    args = rd('args.txt')
    import re
    steps = int(re.search(r'--total-steps (\d+)', args).group(1)) if '--total-steps' in args else None
    cfg = (re.search(r'--rl-elastic-initial-config (\S+)', args) or [None, None])[1]
    seed = (re.search(r'--seed (\d+)', args) or [None, None])[1]
    rounds, last = {}, None
    for e in sorted((e for e in isl if e.get('event') == 'rl_timeline_span'), key=lambda e: e['start']):
        rid = e.get('rollout_id')
        if rid is None and e['task'] == 'publish':
            if last is not None:
                rounds[last] += e['end'] - e['start']
            continue
        rounds[rid] = rounds.get(rid, 0.0) + e['end'] - e['start']; last = rid
    trained = {e.get('rollout_id') for e in isl if e.get('event') == 'rl_round_trained'}
    gpu_txt = ''.join(open(os.path.join(pul, f)).read() for f in os.listdir(pul) if f.startswith('gpu-')) if os.path.isdir(pul) else ''
    uu = set(re.findall(r'GPU-[0-9a-f-]{36}', gpu_txt))
    pools = [r for r in j if r.get('kind') == 'gpu_pool']
    pool_uu = set(pools[-1].get('roles', {})) if pools else set()
    rr = any(r.get('phase') == 'RECOVERY_REQUIRED' for r in j) or any(e.get('result') == 'RECOVERY_REQUIRED' for e in isl)
    steady = [v for k, v in rounds.items() if k is not None and k >= 1]
    checks = {'rc0': 'rc=0' in rd('rc.txt'), 'rounds_eq_steps': steps is not None and len(trained) == steps,
              'l40s': 'L40S' in gpu_txt and 'H100' not in gpu_txt, 'uuids_4_match_pool': len(uu) == 4 and pool_uu == uu,
              'no_recovery_required': not rr}
    t0, t1 = rd('start_utc.txt').strip(), rd('end_utc.txt').strip()
    return dict(run=os.path.basename(run.rstrip('/')), config=cfg, seed=seed, valid=all(checks.values()), checks=checks,
                median_round_s=round(statistics.median(steady), 3) if steady else None,
                rounds_s={k: round(v, 3) for k, v in sorted(rounds.items())}, start_utc=t0, end_utc=t1)


def main(argv):
    if argv and argv[0] == 'sweep':
        res = [sweep(r) for r in argv[1:] if not r.startswith('--')]
        print(json.dumps(res, indent=1))
        return 0
    out = None
    if '--json' in argv:
        i = argv.index('--json'); out = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    rows = [r for run in argv for r in analyze(run)]
    cols = ['run', 'request_id', 'source', 'target', 'wait_safe', 'drain', 'init', 'verify',
            'resume', 'blocking', 'first_step_extra', 'steady_round_s']
    print('\t'.join(cols))
    for r in rows:
        print('\t'.join(str(r[c]) for c in cols))
    if out:
        json.dump(rows, open(out, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
