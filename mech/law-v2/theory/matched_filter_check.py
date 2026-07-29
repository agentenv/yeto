"""law-v2 theory: matched-filter (signal/energy) law test.

Model: over a fragment of T outer updates with per-step displacement
multipliers m_t (code-true, LeanMechanism terminalMultiplier), a tuned outer
rate in the linear-signal / quadratic-noise-energy regime satisfies
    eta* proportional to  C / E,   C = sum m_t,  E = sum m_t^2.
Hence eta*_mom/eta*_mu0 = (C/E)_conv / (T/T) = C/E, and the residual left
after the C-alignment law (eta ratio = T/C) is
    phi_pred = (C/E) / (T/C) = C^2 / (T E)  = 1/(1+CV^2(m)),
the Cauchy-Schwarz defect of the multiplier ramp. Zero free parameters.
"""
import csv, math, collections

def mults(conv, T, mu):
    # per-step displacement multiplier at applied update t = 1..T
    if mu == 0: return [1.0]*T
    if conv == 'nesterov_raw':
        return [(1-mu**(t+1))/(1-mu) for t in range(1, T+1)]
    if conv == 'heavy_ball':
        return [(1-mu**t)/(1-mu) for t in range(1, T+1)]
    if conv == 'nesterov_corrected':
        return [1.0/(1-mu)]*T   # correction removes age dependence exactly
    raise ValueError(conv)

def Mfac(conv, T, mu):
    if conv == 'nesterov_raw': return (1-mu**(T+1))/(1-mu)
    if conv == 'heavy_ball':  return (1-mu**T)/(1-mu)
    return 1.0

rows = list(csv.DictReader(open('/private/tmp/yeto-h200/mech/law-unification/paired_cancellation.csv')))
res = []
for r in rows:
    T = int(r['T']); mu = float(r['mu']); conv = r['convention']
    obs = float(r['observed_to_law_ratio'])
    m = mults(conv, T, mu)
    C = sum(m); E = sum(x*x for x in m)
    # matched-filter predicted eta ratio = C/E ; ledger normalizes by (1-mu)M
    pred_obs_to_law = (C/E) / ((1-mu)*Mfac(conv, T, mu))
    phi_obs = obs / ((T/C)/((1-mu)*Mfac(conv,T,mu)))   # residual after C-law
    phi_pred = C*C/(T*E)                                # matched-filter phi
    res.append((conv, mu, T, int(r['S']), r['campaign'], obs, pred_obs_to_law, phi_obs, phi_pred, phi_obs/phi_pred))

res.sort(key=lambda x:(x[0],x[1],x[2]))
print(f"{'convention':20} {'mu':>5} {'T':>4} {'S':>6} {'camp':10} {'obs':>8} {'MFpred':>8} {'phi_obs':>8} {'phi_MF':>8} {'obs/MF':>7}")
for conv,mu,T,S,c,obs,pred,po,pp,rr in res:
    print(f"{conv:20} {mu:>5} {T:>4} {S:>6} {c:10} {obs:8.4f} {pred:8.4f} {po:8.4f} {pp:8.4f} {rr:7.4f}")

logr = [math.log2(rr) for *_, rr in res]
import statistics
print(f"\nALL {len(res)} pairs: log2(obs/MF-pred) mean {statistics.mean(logr):+.3f}, sd {statistics.pstdev(logr):.3f}, range [{min(logr):+.3f},{max(logr):+.3f}]")
# split by convention
by = collections.defaultdict(list)
for conv,mu,T,S,c,obs,pred,po,pp,rr in res: by[conv].append(math.log2(rr))
for k,v in by.items():
    print(f"  {k:20} n={len(v):2d} mean {statistics.mean(v):+.3f} bits, sd {statistics.pstdev(v):.3f}")
# by mu
by2 = collections.defaultdict(list)
for conv,mu,T,S,c,obs,pred,po,pp,rr in res: by2[mu].append(math.log2(rr))
for k in sorted(by2):
    v=by2[k]; print(f"  mu={k:<5} n={len(v):2d} mean {statistics.mean(v):+.3f} bits, sd {statistics.pstdev(v):.3f}")
# by T
by3 = collections.defaultdict(list)
for conv,mu,T,S,c,obs,pred,po,pp,rr in res: by3[T].append(math.log2(rr))
for k in sorted(by3):
    v=by3[k]; print(f"  T={k:<4} n={len(v):2d} mean {statistics.mean(v):+.3f} bits, sd {statistics.pstdev(v):.3f}")
