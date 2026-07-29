"""law-v2 theory: noise-carried-forward (NCF) endpoint-variance closure.

Dynamics: pseudo-gradient p_t = g + n_t, Var(n)=sigma^2. Buffer b_t = mu b_{t-1} + p_t.
Updates (code-true):
  raw Nesterov  : u_t = p_t + mu b_t         -> noise coeff of n_j in u_t = d_{tj} + mu^{t-j+1}
  heavy-ball    : u_t = b_t                  -> coeff = mu^{t-j}
  corrected     : u_t = (p_t + mu b_t)/(1-mu^{t+1})  (det coeff exactly 1/(1-mu))
  mu0           : u_t = p_t                  -> coeff = d_{tj}
Endpoint after T outer steps: x_T = eta * [C g + sum_j w_j n_j],
  w_j = forward tail of noise-j coefficients, C = sum_j w_j (= effectiveCoeff).
Quadratic endpoint risk: L(eta) = (a/2)(theta - eta C g)^2 + (a/2) eta^2 sigma^2 W,  W = sum_j w_j^2.
  => eta* = theta g C/(g^2 C^2 + sigma^2 W)  =>  with s = sigma^2/g^2 (theta-free ratio form):
  eta*_conv/eta*_mu0 = (T/C) * (1+s/T)/(1+s W/C^2).
Residual after the C-alignment law: phi_pred = (1+s/T)/(1+s W/C^2).
One physical parameter s (noise-to-signal ratio of the pseudo-gradient),
possibly stratified by local work H = S/T.
"""
import csv, math, collections

def minimize_scalar(f, bounds=(-8,8), method=None):
    import math
    a,b = bounds
    gr = (math.sqrt(5)-1)/2
    c,d = b-gr*(b-a), a+gr*(b-a)
    for _ in range(200):
        if f(c) < f(d): b,d = d,c; c = b-gr*(b-a)
        else: a,c = c,d; d = a+gr*(b-a)
    class R: x = (a+b)/2
    return R


def tails(conv, T, mu):
    if mu == 0: return [1.0]*T
    if conv == 'nesterov_raw':
        return [1 + mu*(1-mu**(T-j+1))/(1-mu) for j in range(1, T+1)]
    if conv == 'heavy_ball':
        return [(1-mu**(T-j+1))/(1-mu) for j in range(1, T+1)]
    if conv == 'nesterov_corrected':
        w = []
        for j in range(1, T+1):
            tot = 0.0
            for t in range(j, T+1):
                coeff = (1.0 if t==j else 0.0) + mu**(t-j+1)
                tot += coeff/(1-mu**(t+1))
            w.append(tot)
        return w
    raise ValueError(conv)

def CW(conv, T, mu):
    w = tails(conv, T, mu)
    return sum(w), sum(x*x for x in w)

def phi_pred(conv, T, mu, s):
    Cc, W = CW(conv, T, mu)
    return (1+s/T)/(1+s*W/(Cc*Cc))

rows = list(csv.DictReader(open('/private/tmp/yeto-h200/mech/law-unification/paired_cancellation.csv')))
data = []
for r in rows:
    T=int(r['T']); mu=float(r['mu']); conv=r['convention']; S=int(r['S']); Hh=int(r['H'])
    obs=float(r['observed_to_law_ratio'])
    Cc,W = CW(conv,T,mu)
    Mfac = {'nesterov_raw':(1-mu**(T+1))/(1-mu),'heavy_ball':(1-mu**T)/(1-mu),'nesterov_corrected':1.0}[conv]
    phi_obs = obs/((T/Cc)/((1-mu)*Mfac))
    data.append(dict(conv=conv,T=T,mu=mu,S=S,H=Hh,camp=r['campaign'],scale=r['scale'],phi=phi_obs,C=Cc,W=W))

def loss_global(ls):
    s = math.exp(ls)
    return sum((math.log(d['phi']) - math.log(phi_pred(d['conv'],d['T'],d['mu'],s)))**2 for d in data)

res = minimize_scalar(loss_global, bounds=(-8,8), method='bounded')
s_hat = math.exp(res.x)
import statistics
resid = [math.log2(d['phi']/phi_pred(d['conv'],d['T'],d['mu'],s_hat)) for d in data]
print(f"GLOBAL single-s fit: s_hat = {s_hat:.4f}, rms residual {statistics.pstdev(resid):.3f} bits, mean {statistics.mean(resid):+.3f} (n={len(data)})")
print(f"  (compare: C-law-only residual phi has rms log2 spread {statistics.pstdev([math.log2(d['phi']) for d in data]):.3f} bits about mean {statistics.mean([math.log2(d['phi']) for d in data]):+.3f})")

# fit s per H stratum (H = inner steps per outer round)
print('\nper-H fits:')
byH = collections.defaultdict(list)
for d in data: byH[d['H']].append(d)
sH = {}
for Hh in sorted(byH):
    ds = byH[Hh]
    f = lambda ls: sum((math.log(d['phi'])-math.log(phi_pred(d['conv'],d['T'],d['mu'],math.exp(ls))))**2 for d in ds)
    r2 = minimize_scalar(f, bounds=(-8,8), method='bounded')
    sH[Hh] = math.exp(r2.x)
    rr = [math.log2(d['phi']/phi_pred(d['conv'],d['T'],d['mu'],sH[Hh])) for d in ds]
    print(f"  H={Hh:<5} n={len(ds):2d}  s={sH[Hh]:8.4f}  rms {statistics.pstdev(rr):.3f} bits")

# per-(scale) fit
print('\nper-scale fits:')
bySc = collections.defaultdict(list)
for d in data: bySc[d['scale']].append(d)
for sc in sorted(bySc):
    ds=bySc[sc]
    f = lambda ls: sum((math.log(d['phi'])-math.log(phi_pred(d['conv'],d['T'],d['mu'],math.exp(ls))))**2 for d in ds)
    r2 = minimize_scalar(f, bounds=(-8,8), method='bounded')
    sv = math.exp(r2.x)
    rr=[math.log2(d['phi']/phi_pred(d['conv'],d['T'],d['mu'],sv)) for d in ds]
    print(f"  scale={sc:5} n={len(ds):2d} s={sv:8.4f} rms {statistics.pstdev(rr):.3f} bits")

# show global-fit table
print(f"\n{'conv':20}{'mu':>5}{'T':>5}{'H':>6}{'camp':>10}{'phi_obs':>9}{'phi_NCF':>9}{'resid_bits':>11}")
for d in sorted(data,key=lambda d:(d['conv'],d['mu'],d['T'])):
    pp = phi_pred(d['conv'],d['T'],d['mu'],s_hat)
    print(f"{d['conv']:20}{d['mu']:>5}{d['T']:>5}{d['H']:>6}{d['camp']:>10}{d['phi']:9.4f}{pp:9.4f}{math.log2(d['phi']/pp):+11.3f}")

# where is the predicted minimum of phi(T) and how does it move with mu? (raw, s=s_hat)
print('\npredicted dip location/depth (raw Nesterov, global s):')
for mu in [0.5,0.8,0.9,0.95]:
    best=(None,1e9)
    for T in range(1,400):
        p=phi_pred('nesterov_raw',T,mu,s_hat)
        if p<best[1]: best=(T,p)
    print(f"  mu={mu}: T_min={best[0]}, phi_min={best[1]:.3f}")
