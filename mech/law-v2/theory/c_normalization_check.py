"""law-v2 theory: test accumulated-displacement (C) normalization against the
banked paired-cancellation ledger, and characterize the shared residual phi.

C_conv(T,mu) = sum over the T applied updates of the code-true per-step
displacement coefficient (LeanMechanism.FiniteHorizonOuter.effectiveCoeff):
  nesterov_raw : C = T/(1-mu) - mu^2 (1-mu^T)/(1-mu)^2
  heavy_ball   : C = T/(1-mu) - mu   (1-mu^T)/(1-mu)^2
  corrected    : per-step coefficient is exactly 1/(1-mu) -> C = T/(1-mu)
  mu0          : C = T
Frozen-gradient alignment (finiteHorizon_any_minimizers_align) predicts
  eta*_mom / eta*_mu0 = C_mu0 / C_conv = T / C_conv(T,mu).
The banked ledger stores observed_to_law = (eta_mom/eta_mu0) / ((1-mu) M),
so the C-law prediction for that column is  [T/C] / [(1-mu) M].
Residual phi := observed / predicted should be shared across conventions.
"""
import csv, math, collections

def C(conv, T, mu):
    if mu == 0: return T
    if conv == 'nesterov_raw':
        return T/(1-mu) - mu**2*(1-mu**T)/(1-mu)**2
    if conv == 'heavy_ball':
        return T/(1-mu) - mu*(1-mu**T)/(1-mu)**2
    if conv == 'nesterov_corrected':
        return T/(1-mu)
    raise ValueError(conv)

def Mfac(conv, T, mu):
    if conv == 'nesterov_raw': return (1-mu**(T+1))/(1-mu)
    if conv == 'heavy_ball':  return (1-mu**T)/(1-mu)
    return 1.0

rows = list(csv.DictReader(open('/private/tmp/yeto-h200/mech/law-unification/paired_cancellation.csv')))
out = []
for r in rows:
    T = int(r['T']); mu = float(r['mu']); conv = r['convention']
    obs = float(r['observed_to_law_ratio'])
    pred = (T / C(conv, T, mu)) / ((1-mu) * Mfac(conv, T, mu))
    phi = obs / pred
    out.append((r['campaign'], conv, mu, T, int(r['S']), obs, pred, phi))

out.sort(key=lambda x: (x[1], x[2], x[3]))
print(f"{'campaign':10} {'convention':20} {'mu':>5} {'T':>3} {'S':>6} {'obs':>8} {'C-pred':>8} {'phi':>7}")
for c, conv, mu, T, S, obs, pred, phi in out:
    print(f"{c:10} {conv:20} {mu:>5} {T:>3} {S:>6} {obs:8.4f} {pred:8.4f} {phi:7.4f}")

# median phi per (convention, mu, T)
med = collections.defaultdict(list)
for c, conv, mu, T, S, obs, pred, phi in out:
    med[(conv, mu, T)].append(phi)
print('\nmedian phi by (convention, mu, T):')
for k in sorted(med, key=lambda k:(k[0],k[1],k[2])):
    v = sorted(med[k]); m = v[len(v)//2] if len(v)%2 else 0.5*(v[len(v)//2-1]+v[len(v)//2])
    print(f"  {k[0]:20} mu={k[1]:<5} T={k[2]:<3} n={len(v):<2} median_phi={m:.4f} log2={math.log2(m):+.3f}")

# fit phi ~ T^-gamma pooled across conventions (mu=0.9 cells)
import statistics
pts = [(math.log(T), math.log(phi)) for c,conv,mu,T,S,obs,pred,phi in out if mu==0.9]
n=len(pts); sx=sum(x for x,_ in pts); sy=sum(y for _,y in pts)
sxx=sum(x*x for x,_ in pts); sxy=sum(x*y for x,y in pts)
b=(n*sxy-sx*sy)/(n*sxx-sx*sx); a=(sy-b*sx)/n
resid=[y-(a+b*x) for x,y in pts]
print(f"\npooled mu=0.9 fit: phi = {math.exp(a):.4f} * T^({b:+.4f}),  rms log-resid = {statistics.pstdev(resid):.4f} nats ({statistics.pstdev(resid)/math.log(2):.3f} bits), n={n}")

# same fit but per convention, to test sharing
for cv in ['nesterov_raw','nesterov_corrected','heavy_ball']:
    pts=[(math.log(T),math.log(phi)) for c,conv,mu,T,S,obs,pred,phi in out if mu==0.9 and conv==cv]
    if len(pts)<3: continue
    n=len(pts); sx=sum(x for x,_ in pts); sy=sum(y for _,y in pts)
    sxx=sum(x*x for x,_ in pts); sxy=sum(x*y for x,y in pts)
    b=(n*sxy-sx*sy)/(n*sxx-sx*sx); a=(sy-b*sx)/n
    print(f"  {cv:20}: phi = {math.exp(a):.4f} * T^({b:+.4f}), n={n}")

# contrast: how bad is the pure-M law on the same cells (log2 spread of obs itself)?
obs09=[math.log2(o) for c,conv,mu,T,S,o,p,phi in out if mu==0.9]
phi09=[math.log2(phi) for c,conv,mu,T,S,o,p,phi in out if mu==0.9]
print(f"\nmu=0.9 cells: obs_to_law(M-law) log2 range [{min(obs09):+.2f},{max(obs09):+.2f}] span {max(obs09)-min(obs09):.2f} bits")
print(f"             phi (C-law resid) log2 range [{min(phi09):+.2f},{max(phi09):+.2f}] span {max(phi09)-min(phi09):.2f} bits")
