"""law-v2 theory: AR(1)-correlated pseudo-gradient fluctuation closure (L2).

p_t = gbar + f_t, Cov(f_j,f_k) = sigma^2 rho^{|j-k|}  (rho = outer-step
lag-1 alignment of pseudo-gradient fluctuations; telemetry band 0.55-0.71).
Endpoint displacement D = sum_j w_j p_j with forward-tail coefficients w_j.
Quadratic endpoint risk => eta* = <theta,gbar> C / (gbar^2 C^2 + sigma^2 Q),
Q = sum_{jk} w_j w_k rho^{|j-k|}.  Ratio to mu0 (w=1, Q0 over T):
  phi(T,mu;s,rho) = (1 + s Q0/T^2) / (1 + s Q/C^2),  s = sigma^2/gbar^2.
rho=0 reduces to the rejected i.i.d. floor; rho->1 reduces to full common-mode
(no penalty).  Fit (s, rho) jointly; report whether rho_hat falls in the
measured 0.55-0.71 telemetry band with NO tape input used in fitting.
"""
import csv, math, collections

def tails(conv, T, mu):
    if mu == 0: return [1.0]*T
    if conv == 'nesterov_raw':
        return [1 + mu*(1-mu**(T-j+1))/(1-mu) for j in range(1, T+1)]
    if conv == 'heavy_ball':
        return [(1-mu**(T-j+1))/(1-mu) for j in range(1, T+1)]
    if conv == 'nesterov_corrected':
        w=[]
        for j in range(1,T+1):
            tot=0.0
            for t in range(j,T+1):
                tot += ((1.0 if t==j else 0.0)+mu**(t-j+1))/(1-mu**(t+1))
            w.append(tot)
        return w
    raise ValueError(conv)

def quad(w, rho):
    T=len(w)
    # Q = sum_j w_j^2 + 2 sum_{j<k} w_j w_k rho^{k-j}, O(T) via recursion
    Q = sum(x*x for x in w)
    acc = 0.0   # acc_k = sum_{j<k} w_j rho^{k-j}
    for k in range(1,T):
        acc = (acc + w[k-1])*rho
        Q += 2*w[k]*acc
    return Q

CACHE={}
def phi_pred(conv,T,mu,s,rho):
    key=(conv,T,mu)
    if key not in CACHE: CACHE[key]=tails(conv,T,mu)
    w=CACHE[key]; C=sum(w)
    Q=quad(w,rho); Q0=quad([1.0]*T,rho)
    return (1+s*Q0/T**2)/(1+s*Q/(C*C))

rows=list(csv.DictReader(open('/private/tmp/yeto-h200/mech/law-unification/paired_cancellation.csv')))
data=[]
for r in rows:
    T=int(r['T']);mu=float(r['mu']);conv=r['convention']
    obs=float(r['observed_to_law_ratio'])
    w=tails(conv,T,mu);C=sum(w)
    Mfac={'nesterov_raw':(1-mu**(T+1))/(1-mu),'heavy_ball':(1-mu**T)/(1-mu),'nesterov_corrected':1.0}[conv]
    phi_obs=obs/((T/C)/((1-mu)*Mfac))
    data.append(dict(conv=conv,T=T,mu=mu,H=int(r['H']),S=int(r['S']),camp=r['campaign'],scale=r['scale'],phi=phi_obs))

import statistics
def sse(s,rho,ds=None):
    ds = ds or data
    return sum((math.log(d['phi'])-math.log(phi_pred(d['conv'],d['T'],d['mu'],s,rho)))**2 for d in ds)

# coarse-to-fine grid over (log s, rho)
best=(None,None,1e18)
grid_s=[math.exp(x/2) for x in range(-6,17)]
grid_r=[i/50 for i in range(0,50)]
for s in grid_s:
    for rho in grid_r:
        v=sse(s,rho)
        if v<best[2]: best=(s,rho,v)
s0,r0,_=best
# refine
for it in range(40):
    improved=False
    for ds_ in [1.15,1/1.15,1.03,1/1.03]:
        if sse(s0*ds_,r0)<sse(s0,r0): s0*=ds_; improved=True
    for dr in [0.02,-0.02,0.005,-0.005]:
        r1=min(0.995,max(0,r0+dr))
        if sse(s0,r1)<sse(s0,r0): r0=r1; improved=True
    if not improved: break
resid=[math.log2(d['phi']/phi_pred(d['conv'],d['T'],d['mu'],s0,r0)) for d in data]
print(f"GLOBAL (s,rho) fit: s_hat={s0:.3f}, rho_hat={r0:.3f}; rms {statistics.pstdev(resid):.3f} bits, mean {statistics.mean(resid):+.3f} (n={len(data)})")
print("  telemetry lag-1 band: 0.55-0.71; rho_hat in band:", 0.55<=r0<=0.71)

# rho fixed at telemetry band edges, fit s only
for rho in [0.55,0.63,0.69,0.71]:
    sb=s0
    for it in range(60):
        moved=False
        for ds_ in [1.2,1/1.2,1.04,1/1.04]:
            if sse(sb*ds_,rho)<sse(sb,rho): sb*=ds_; moved=True
        if not moved: break
    rr=[math.log2(d['phi']/phi_pred(d['conv'],d['T'],d['mu'],sb,rho)) for d in data]
    print(f"  rho fixed {rho}: s_hat={sb:.3f}, rms {statistics.pstdev(rr):.3f} bits, mean {statistics.mean(rr):+.3f}")

# per-H s (rho global)
print('\nper-H s at rho_hat:')
byH=collections.defaultdict(list)
for d in data: byH[d['H']].append(d)
for Hh in sorted(byH):
    ds=byH[Hh]; sb=s0
    for it in range(60):
        moved=False
        for f in [1.3,1/1.3,1.05,1/1.05]:
            if sse(sb*f,r0,ds)<sse(sb,r0,ds): sb*=f; moved=True
        if not moved: break
    rr=[math.log2(d['phi']/phi_pred(d['conv'],d['T'],d['mu'],sb,r0)) for d in ds]
    print(f"  H={Hh:<5} n={len(ds):2d} s={sb:9.3f} rms {statistics.pstdev(rr):.3f} bits")

# residual table at global fit
print(f"\n{'conv':20}{'mu':>5}{'T':>5}{'H':>6}{'camp':>10}{'phi_obs':>9}{'phi_L2':>8}{'resid':>8}")
for d in sorted(data,key=lambda d:(d['conv'],d['mu'],d['T'],d['H'])):
    pp=phi_pred(d['conv'],d['T'],d['mu'],s0,r0)
    print(f"{d['conv']:20}{d['mu']:>5}{d['T']:>5}{d['H']:>6}{d['camp']:>10}{d['phi']:9.4f}{pp:8.4f}{math.log2(d['phi']/pp):+8.3f}")

# discriminating predictions
print('\nDIP TABLE (raw Nesterov, global fit): T_min(mu), phi_min, and phi at T=160')
for mu in [0.5,0.8,0.9,0.95]:
    best=(None,10)
    for T in list(range(1,200)):
        p=phi_pred('nesterov_raw',T,mu,s0,r0)
        if p<best[1]: best=(T,p)
    print(f"  mu={mu}: T_dip={best[0]}, phi_dip={best[1]:.3f}, phi(T=160)={phi_pred('nesterov_raw',160,mu,s0,r0):.3f}")
print('\nSHARP TEST prediction, raw mu=0.9 T=160 at HIGH H (same s as H=512 stratum):')
print('  L2: phi ->', f"{phi_pred('nesterov_raw',160,0.9,s0,r0):.3f}", '(recovers)')
