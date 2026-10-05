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


def main(argv):
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
