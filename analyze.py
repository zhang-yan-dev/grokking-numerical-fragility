"""Paired analysis: author's published per-seed records vs our independent retrain.
n=300, wd=0.01, threads=4.  numpy only (no scipy)."""
import json, os, glob, math
import numpy as np

AUTH = r"D:\research\author-results-backup"
OURS = r"D:\research\run-ours\results"
WD, TH = "0.01", "4"

def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, f"22_wd{WD}_t{TH}_s*.json")):
        s = int(os.path.basename(f).rsplit("_s", 1)[1].split(".")[0])
        out[s] = json.load(open(f, encoding="utf-8"))
    return out

a, b = load(AUTH), load(OURS)
seeds = sorted(set(a) & set(b))
n = len(seeds)
print(f"paired seeds: {n}  (wd={WD}, threads={TH})\n")

ga = np.array([a[s]["grok"] for s in seeds])
gb = np.array([b[s]["grok"] for s in seeds])
ba = np.array([a[s]["best"] for s in seeds])
bb = np.array([b[s]["best"] for s in seeds])

# ---- 2x2 ----
A = int(np.sum(ga & gb))       # both grok
B = int(np.sum(ga & ~gb))      # author only
C = int(np.sum(~ga & gb))      # ours only
D = int(np.sum(~ga & ~gb))     # neither
print("2x2 contingency (paired):")
print(f"{'':>16}{'ours grok':>11}{'ours no':>10}")
print(f"{'author grok':>16}{A:>11}{B:>10}")
print(f"{'author no':>16}{C:>11}{D:>10}")
print(f"\ndiscordant pairs b={B}, c={C}, total={B+C}")

def binom_pmf(n_, k):
    return math.comb(n_, k) * 0.5**n_

# exact McNemar (two-sided)
m = B + C
k = min(B, C)
p_exact = min(1.0, 2.0 * sum(binom_pmf(m, i) for i in range(0, k + 1)))
print(f"exact McNemar p = {p_exact:.4f}   ({'NOT significant' if p_exact>0.05 else 'SIGNIFICANT'} at 0.05)")

# ---- rates + Wilson CI ----
def wilson(k_, n_, z=1.96):
    if n_ == 0: return (0, 0)
    p = k_ / n_
    den = 1 + z*z/n_
    ctr = (p + z*z/(2*n_)) / den
    half = z*math.sqrt(p*(1-p)/n_ + z*z/(4*n_*n_)) / den
    return (ctr - half, ctr + half)

ra, rb = ga.mean(), gb.mean()
la, ua = wilson(int(ga.sum()), n)
lb, ub = wilson(int(gb.sum()), n)
print(f"\ngrok rate  author = {ra:.1%}  ({int(ga.sum())}/{n})   95% CI [{la:.1%}, {ua:.1%}]")
print(f"grok rate  ours   = {rb:.1%}  ({int(gb.sum())}/{n})   95% CI [{lb:.1%}, {ub:.1%}]")
print(f"difference        = {rb-ra:+.1%}   ({(int(gb.sum())-int(ga.sum())):+d} seeds)")
print(f"CI overlap        = {'YES' if not (ub < la or ua < lb) else 'NO'}")

# ---- best accuracy ----
diff = np.abs(ba - bb)
print(f"\nbest accuracy comparison:")
print(f"  identical          : {int(np.sum(diff<1e-9))}/{n}  ({np.mean(diff<1e-9):.1%})")
print(f"  mean |delta|       : {diff.mean():.3f}")
print(f"  median |delta|     : {np.median(diff):.3f}")
print(f"  max |delta|        : {diff.max():.3f}")
print(f"  author mean best   : {ba.mean():.3f}")
print(f"  ours   mean best   : {bb.mean():.3f}")

# ---- H2: do flips concentrate near the threshold (tau=0.70)? ----
print(f"\nflip rate vs distance from tau=0.70  (grok threshold)")
print(f"{'author best bin':>16}{'n':>6}{'flips':>8}{'flip rate':>11}")
bins = [(0.0,0.3),(0.3,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,1.01)]
for lo, hi in bins:
    m_ = (ba >= lo) & (ba < hi)
    if m_.sum() == 0: continue
    fl = int(np.sum(ga[m_] != gb[m_]))
    print(f"{f'[{lo:.2f},{hi:.2f})':>16}{int(m_.sum()):>6}{fl:>8}{fl/m_.sum():>10.0%}")

near = np.abs(ba - 0.70) <= 0.10
far  = ~near
print(f"\n  within +/-0.10 of tau : n={int(near.sum()):>4}  flip rate {np.mean(ga[near]!=gb[near]):.1%}")
print(f"  farther than 0.10     : n={int(far.sum()):>4}  flip rate {np.mean(ga[far]!=gb[far]):.1%}")

# ---- counterfactual: if only the threshold moved, what rate would we get? ----
# take author's grok labels but rebuild them from OUR best with tau=0.70
rein = (bb >= 0.70)
print(f"\n'best' mean shift = {bb.mean()-ba.mean():+.3f}")
