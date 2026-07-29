"""law-v2 theory: first-order inner-state interference closure (candidate C3).

Mechanism: each outer update of size eta*m_t displaces parameters under the
inner optimizer's adaptive state (Adam moments tuned to the pre-update basin).
The re-adaptation drag is first order in the displacement and diffusive in
inner steps (prop. to sqrt(H)); only the EXCESS displacement relative to the
mu0 control (m_t - 1 per step) survives in the matched ratio. Compounding:
  log phi(T,mu,conv,H,scale) = -beta * sqrt(H) * eta_ctrl(scale,T) * (C_conv - T)
with eta_ctrl(scale,T) = eta0(scale) q^T the control arm's tuned rate (known
from the primary fit), C_conv = effectiveCoeff. ONE new global parameter beta.
Key structural properties:
  - first order in eta  -> deficit self-quenches at large T (q^T -> 0)
  - (C-T): zero at mu=0, ~ T mu/(1-mu) for corrected, ramps for raw/hb
  - sqrt(H) drag        -> matches G6 T=20 column scaling
  - eta0(scale) factor  -> smaller deficit at 1.7B/7B for same cell
"""
import csv, math, collections, statistics

ETA0 = {'135M':0.031350405899186334, '1.7B':0.0051841, '7B':0.0062621}
Q = 0.9875756970034051

def C_of(conv,T,mu):
    if mu==0: return float(T)
    if conv=='nesterov_raw': return T/(1-mu)-mu**2*(1-mu**T)/(1-mu)**2
    if conv=='heavy_ball':  return T/(1-mu)-mu*(1-mu**T)/(1-mu)**2
    if conv=='nesterov_corrected': return T/(1-mu)
    raise ValueError

rows=list(csv.DictReader(open('/private/tmp/yeto-h200/mech/law-unification/paired_cancellation.csv')))
data=[]
for r in rows:
    T=int(r['T']);mu=float(r['mu']);conv=r['convention'];Hh=int(r['H']);sc=r['scale']
    obs=float(r['observed_to_law_ratio'])
    C=C_of(conv,T,mu)
    Mfac={'nesterov_raw':(1-mu**(T+1))/(1-mu),'heavy_ball':(1-mu**T)/(1-mu),'nesterov_corrected':1.0}[conv]
    phi=obs/((T/C)/((1-mu)*Mfac))
    x = math.sqrt(Hh)*ETA0[sc]*Q**T*(C-T)   # drag coordinate
    data.append(dict(conv=conv,T=T,mu=mu,H=Hh,sc=sc,camp=r['campaign'],phi=phi,x=x))

# OLS through origin: log phi = -beta x
num=sum(-math.log(d['phi'])*d['x'] for d in data); den=sum(d['x']**2 for d in data)
beta=num/den
resid=[math.log2(d['phi'])+beta*d['x']/math.log(2) for d in data]
print(f"C3 fit: beta={beta:.5f} (1 global param); rms {statistics.pstdev(resid):.3f} bits, mean {statistics.mean(resid):+.3f} (n={len(data)})")
print("  [C-law only rms 0.278 @ -0.350 | variance-family best 0.224 @ -0.170 | rotation 0.271 @ -0.342]")

print(f"\n{'conv':20}{'mu':>5}{'T':>5}{'H':>6}{'sc':>5}{'camp':>10}{'phi_obs':>9}{'phi_C3':>8}{'resid':>8}")
for d in sorted(data,key=lambda d:(d['conv'],d['mu'],d['T'],d['H'])):
    pp=math.exp(-beta*d['x'])
    print(f"{d['conv']:20}{d['mu']:>5}{d['T']:>5}{d['H']:>6}{d['sc']:>5}{d['camp']:>10}{d['phi']:9.4f}{pp:8.4f}{math.log2(d['phi']/pp):+8.3f}")

print('\nC3 discriminating predictions (raw, 135M):')
for (mu,T,Hh) in [(0.9,160,512),(0.9,40,512),(0.9,80,512),(0.95,40,512),(0.95,80,512),(0.5,20,512),(0.9,20,2048)]:
    C=C_of('nesterov_raw',T,mu)
    x=math.sqrt(Hh)*ETA0['135M']*Q**T*(C-T)
    print(f"  mu={mu:<5} T={T:<4} H={Hh:<5}: phi={math.exp(-beta*x):.3f} ({math.log2(math.exp(-beta*x)):+.2f} bits)")
print('\nfor contrast, variance-family floor C^2/(T*W) and C4 (curvature) prediction at (mu=0.9,T=160,H=512):')
# white floor
w=[1+0.9*(1-0.9**(160-j+1))/0.1 for j in range(1,161)]
Cc=sum(w);W=sum(v*v for v in w)
print(f"  variance floor: {Cc*Cc/(160*W):.3f} | C3: see above | C4: deficit >= T=20 level (~ -0.9 bits) since sharpening never anneals")
