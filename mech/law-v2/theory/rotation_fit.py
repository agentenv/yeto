"""law-v2 theory: buffer-staleness / gradient-rotation closure (L3).

Mechanism: the descent direction rotates as the iterate moves. Per outer step
the gradient direction rotates by angle theta; a pseudo-gradient banked at
outer step j and re-applied (via the buffer) at step t>j has rotated by
theta*(t-j). The mu0 arm always applies the current pseudo-gradient (zero
lag); momentum arms apply stale directions with geometric weights.
Formally: replace the real contraction mu by the complex contraction
mu*e^{i theta} in the forward-tail kernel:
  w~_j(raw)  = 1 + z(1-z^{T-j+1})/(1-z),  z = mu e^{i theta}
  w~_j(hb)   = (1-z^{T-j+1})/(1-z)
  w~_j(corr) = sum_t [d_{tj}+z^{t-j+1}]/(1-mu^{t+1})   (correction is real)
  C~ = sum_j w~_j.  Endpoint quadratic in the plane =>
  eta* = Re(C~)/(|C~|^2) * const  =>  phi_rot = C * Re(C~)/|C~|^2.
theta is a stratum property of local work: theta = theta0 * H^p
(p=1: ballistic inner drift; p=1/2: diffusive). Fit (theta0, p) globally.
"""
import csv, math, cmath, collections

def tails_c(conv, T, mu, th):
    z = mu*cmath.exp(1j*th)
    if conv=='nesterov_raw':
        return [1 + z*(1-z**(T-j+1))/(1-z) for j in range(1,T+1)]
    if conv=='heavy_ball':
        return [(1-z**(T-j+1))/(1-z) for j in range(1,T+1)]
    if conv=='nesterov_corrected':
        w=[]
        for j in range(1,T+1):
            tot=0
            for t in range(j,T+1):
                tot += ((1 if t==j else 0)+z**(t-j+1))/(1-mu**(t+1))
            w.append(tot)
        return w
    raise ValueError

def tails_r(conv,T,mu): return [x.real for x in tails_c(conv,T,mu,0.0)]

def phi_rot(conv,T,mu,th):
    C = sum(tails_r(conv,T,mu))
    Ct = sum(tails_c(conv,T,mu,th))
    return C*Ct.real/abs(Ct)**2

rows=list(csv.DictReader(open('/private/tmp/yeto-h200/mech/law-unification/paired_cancellation.csv')))
data=[]
for r in rows:
    T=int(r['T']);mu=float(r['mu']);conv=r['convention']
    obs=float(r['observed_to_law_ratio'])
    C=sum(tails_r(conv,T,mu))
    Mfac={'nesterov_raw':(1-mu**(T+1))/(1-mu),'heavy_ball':(1-mu**T)/(1-mu),'nesterov_corrected':1.0}[conv]
    data.append(dict(conv=conv,T=T,mu=mu,H=int(r['H']),camp=r['campaign'],
                     phi=obs/((T/C)/((1-mu)*Mfac))))

import statistics
def sse(th0,p):
    tot=0.0
    for d in data:
        th=min(2.5, th0*d['H']**p)
        pred=phi_rot(d['conv'],d['T'],d['mu'],th)
        if pred<=0: return 1e18
        tot+=(math.log(d['phi'])-math.log(pred))**2
    return tot

best=(None,None,1e18)
for p in [0.0,0.25,0.5,0.75,1.0]:
    for k in range(-40,10):
        th0=math.exp(k/4)/512**p
        v=sse(th0,p)
        if v<best[2]: best=(th0,p,v)
th0,p,_=best
for it in range(60):
    moved=False
    for f in [1.2,1/1.2,1.04,1/1.04]:
        if sse(th0*f,p)<sse(th0,p): th0*=f; moved=True
    for dp in [0.05,-0.05]:
        if 0<=p+dp<=1.2 and sse(th0,p+dp)<sse(th0,p): p+=dp; moved=True
    if not moved: break
resid=[math.log2(d['phi']/phi_rot(d['conv'],d['T'],d['mu'],min(2.5,th0*d['H']**p))) for d in data]
print(f"L3 rotation fit: theta0={th0:.5f}, p={p:.2f}; theta(H=512)={th0*512**p:.4f} rad")
print(f"  rms {statistics.pstdev(resid):.3f} bits, mean {statistics.mean(resid):+.3f} (n={len(data)})")
print(f"  [C-law only: rms 0.278 about -0.350; best variance closure: rms 0.224 about -0.170]")

print(f"\n{'conv':20}{'mu':>5}{'T':>5}{'H':>6}{'camp':>10}{'phi_obs':>9}{'phi_L3':>8}{'resid':>8}")
for d in sorted(data,key=lambda d:(d['conv'],d['mu'],d['T'],d['H'])):
    th=min(2.5,th0*d['H']**p)
    pp=phi_rot(d['conv'],d['T'],d['mu'],th)
    print(f"{d['conv']:20}{d['mu']:>5}{d['T']:>5}{d['H']:>6}{d['camp']:>10}{d['phi']:9.4f}{pp:8.4f}{math.log2(d['phi']/pp):+8.3f}")

# discriminating predictions
print('\npredicted phi (raw) with H=512 held fixed:')
th=th0*512**p
for mu in [0.5,0.8,0.9,0.95]:
    row=[f"T={T}:{phi_rot('nesterov_raw',T,mu,th):.3f}" for T in [2,5,10,20,40,80,160]]
    print(f"  mu={mu}: "+" ".join(row))
print('predicted phi (raw, mu=0.9) vs H at T=20:')
for Hh in [16,64,128,256,512,1024,2048]:
    print(f"  H={Hh}: {phi_rot('nesterov_raw',20,0.9,min(2.5,th0*Hh**p)):.3f}")
