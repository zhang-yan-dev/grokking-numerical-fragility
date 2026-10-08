"""
Audit-driven statistics for the 2nd-round (ARS) revision:
  (1) Mechanism: Pearson r + Fisher-z 95% CI + Spearman rho for logit_fourier vs heldout_acc,
      M=7 and M=10 (report uncertainty + rank correlation; r is anchored by a tiny class).
  (2) Knife-edge: Newcombe (1998) method-10 CI for the difference of two PAIRED proportions,
      replacing the mislabeled single-proportion Wilson "difference bound", for thread (t1 vs t4)
      and device (cpu vs cuda) at tau=0.70.
Usage: python 36_review2_stats.py
"""
import os, glob, json, math
RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = p + z*z/(2*n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-h)/d, (c+h)/d)

def fisher_ci(r, n, z=1.96):
    if n < 4: return (float('nan'), float('nan'))
    zr = math.atanh(max(min(r, 0.999999), -0.999999)); se = 1/math.sqrt(n-3)
    return (math.tanh(zr - z*se), math.tanh(zr + z*se))

def pearson(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    sxy = sum((a-mx)*(b-my) for a, b in zip(x, y))
    sxx = sum((a-mx)**2 for a in x); syy = sum((b-my)**2 for b in y)
    return sxy/math.sqrt(sxx*syy) if sxx*syy > 0 else 0.0

def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0]*len(v); i = 0
        while i < len(v):
            j = i
            while j+1 < len(v) and v[order[j+1]] == v[order[i]]: j += 1
            avg = (i+j)/2 + 1
            for k in range(i, j+1): r[order[k]] = avg
            i = j+1
        return r
    return pearson(ranks(x), ranks(y))

def load_mech(M):
    files = glob.glob(os.path.join(RES, f"20_fourier_m{M}.json")) + \
            glob.glob(os.path.join(RES, f"20_fourier_m{M}_s*.json"))
    seen = {}
    for f in files:
        try: data = json.load(open(f))
        except Exception: continue
        for x in data.get("results", []):
            if x["seed"] not in seen:
                seen[x["seed"]] = (x["logit_fourier"], x["heldout_acc"])
    xs = [v[0] for v in seen.values()]; ys = [v[1] for v in seen.values()]
    return xs, ys

print("="*70)
print("(1) MECHANISM: Pearson r (Fisher-z 95% CI) + Spearman rho")
print("="*70)
for M in (7, 10):
    xs, ys = load_mech(M)
    n = len(xs)
    if n < 3:
        print(f" M={M}: insufficient data ({n})"); continue
    r = pearson(xs, ys); lo, hi = fisher_ci(r, n); rho = spearman(xs, ys)
    ng = sum(1 for v in ys if v >= 0.70)
    print(f" M={M}: n={n} (grok={ng}, mem={n-ng})  Pearson r={r:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]"
          f"  Spearman rho={rho:+.2f}")

print("\n" + "="*70)
print("(2) KNIFE-EDGE: Newcombe method-10 CI for difference of PAIRED proportions (tau=0.70)")
print("="*70)

def newcombe_paired(a, b, c, d):
    n = a+b+c+d
    p1 = (a+b)/n; p2 = (a+c)/n; diff = p1 - p2
    l1, u1 = wilson(a+b, n); l2, u2 = wilson(a+c, n)
    denom = (a+b)*(c+d)*(a+c)*(b+d)
    phi = ((a*d - b*c)/math.sqrt(denom)) if denom > 0 else 0.0
    L = diff - math.sqrt(max(0.0, (p1-l1)**2 - 2*phi*(p1-l1)*(u2-p2) + (u2-p2)**2))
    U = diff + math.sqrt(max(0.0, (u1-p1)**2 - 2*phi*(u1-p1)*(p2-l2) + (p2-l2)**2))
    return diff, L, U, phi

def paired_table(pairs, tau=0.70):
    a=b=c=d=0
    for x, y in pairs:
        gx, gy = x>=tau, y>=tau
        if gx and gy: a+=1
        elif gx and not gy: b+=1
        elif not gx and gy: c+=1
        else: d+=1
    return a, b, c, d

def loadall(pat):
    out=[]
    for f in glob.glob(os.path.join(RES, pat)):
        try: out.append(json.load(open(f)))
        except Exception: pass
    return out

thr={}
for r in loadall("22_wd*_t*_s*.json"):
    if abs(r['wd']-0.01)<1e-9 and r['threads'] in (1,4):
        thr.setdefault(r['seed'],{})[r['threads']]=r['best']
tp=[(d[1],d[4]) for d in thr.values() if 1 in d and 4 in d]
dev={}
for r in loadall("26_dev*_s*.json"):
    dev.setdefault(r['seed'],{})[r['device']]=r['best']
dp=[(d['cpu'],d['cuda']) for d in dev.values() if 'cpu' in d and 'cuda' in d]

for label, pairs in [("Thread t1-vs-t4", tp), ("Device CPU-vs-GPU", dp)]:
    a,b,c,d = paired_table(pairs)
    diff,L,U,phi = newcombe_paired(a,b,c,d)
    n=a+b+c+d
    print(f" {label}: n={n}  table[both+={a}, 1only={b}, 2only={c}, both-={d}]  "
          f"rate1={(a+b)/n:.0%} rate2={(a+c)/n:.0%}")
    print(f"    paired diff = {diff:+.1%}  Newcombe 95% CI = [{L:+.1%}, {U:+.1%}]  (phi={phi:+.2f})")
print()
