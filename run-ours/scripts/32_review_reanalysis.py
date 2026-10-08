"""
ZERO-COST re-analyses addressing two headline reviewer concerns, computed entirely from the
`best` accuracy already stored in existing result JSONs (no retraining):

  (A) Threshold robustness: re-tabulate the weight-decay inverted-U, the thread/device
      knife-edge flip rates, and the decomposition grok-rates at grok thresholds
      {0.60, 0.70, 0.80}. If the qualitative story is invariant, the 0.70 choice is vindicated.

  (B) Knife-edge statistics: replace the asserted "unbiased / zero aggregate bias" with the
      correct paired-binary test (McNemar exact on discordant pairs), a Wilson CI on the
      per-condition rate difference, and an explicit power / detectable-effect statement.
      Applies to threads (t1 vs t4) and device (CPU vs GPU).

Usage: python 32_review_reanalysis.py
"""
import os, glob, json, math
import numpy as np

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

def loadall(pat):
    out = []
    for f in glob.glob(os.path.join(RES, pat)):
        try: out.append(json.load(open(f)))
        except Exception: pass
    return out

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = p + z*z/(2*n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-h)/d, (c+h)/d)

def mcnemar_exact(b, c):
    """two-sided exact McNemar on discordant counts b, c (binomial against 0.5)."""
    n = b + c
    if n == 0: return 1.0
    from math import comb
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k+1)) / (2**n)
    return min(1.0, 2*p)

print("="*70)
print("(A) THRESHOLD ROBUSTNESS  — grok-rate recomputed at tau in {0.60,0.70,0.80}")
print("="*70)

# weight-decay inverted-U (threads=4)
wd_rows = [r for r in loadall("22_wd*_t*_s*.json") if r.get("threads") == 4]
print("\nWeight-decay inverted-U (a+b)%10, threads=4:")
print(f"  {'wd':>5} {'n':>3}   " + "   ".join(f"tau={t:.2f}" for t in (.6,.7,.8)))
for w in sorted({r['wd'] for r in wd_rows}):
    g = [r for r in wd_rows if r['wd'] == w]
    rates = [sum(r['best'] >= t for r in g)/len(g) for t in (.6,.7,.8)]
    print(f"  {w:>5g} {len(g):>3}   " + "    ".join(f"{x:5.0%}" for x in rates))

# decomposition (pipeline grok-rate) at thresholds — from 25_decomp
dec = loadall("25_decomp_wd*_s*.json")
if dec:
    print("\nDecomposition pipeline-vs-monolith grok-rate (mul2add2, matched budget):")
    print(f"  {'wd':>5} {'n':>3}   " + "  ".join(f"pipe/mono@{t:.2f}" for t in (.6,.7,.8)))
    for w in sorted({r['wd'] for r in dec}):
        g = [r for r in dec if r['wd'] == w]
        cells = []
        for t in (.6,.7,.8):
            pr = sum(r['pipeline']['overall_acc'] >= t for r in g)/len(g)
            mo = sum(r['monolith']['best'] >= t for r in g)/len(g)
            cells.append(f"{pr:4.0%}/{mo:4.0%}")
        print(f"  {w:>5g} {len(g):>3}   " + "    ".join(cells))

def flip_table(pairs, label):
    print(f"\n{label}: grok-status flips & paired rates at each tau")
    print(f"  {'tau':>5} {'flips':>9} {'rate_A':>8} {'rate_B':>8}")
    for t in (.6,.7,.8):
        flips = sum((a >= t) != (b >= t) for a,b in pairs)
        ra = sum(a >= t for a,_ in pairs); rb = sum(b >= t for _,b in pairs)
        n = len(pairs)
        print(f"  {t:>5.2f}   {flips:>3}/{n:<3}   {ra:>3}/{n:<3}  {rb:>3}/{n:<3}")

# build paired arrays for threads (t1 vs t4 at wd=0.01) and device (cpu vs cuda)
thr = {}
for r in loadall("22_wd*_t*_s*.json"):
    if abs(r['wd']-0.01) < 1e-9 and r['threads'] in (1,4):
        thr.setdefault(r['seed'], {})[r['threads']] = r['best']
thr_pairs = [(d[1], d[4]) for d in thr.values() if 1 in d and 4 in d]

dev = {}
for r in loadall("26_dev*_s*.json"):
    dev.setdefault(r['seed'], {})[r['device']] = r['best']
dev_pairs = [(d['cpu'], d['cuda']) for d in dev.values() if 'cpu' in d and 'cuda' in d]

flip_table(thr_pairs, "Threads t1-vs-t4")
flip_table(dev_pairs, "Device CPU-vs-GPU")

print("\n" + "="*70)
print("(B) KNIFE-EDGE PAIRED STATISTICS (tau=0.70) — McNemar exact + Wilson CI + power")
print("="*70)

def paired_stats(pairs, label, tau=0.70):
    n = len(pairs)
    if n == 0:
        print(f"\n{label}: NO PAIRS FOUND"); return
    A = np.array([a >= tau for a,_ in pairs]); B = np.array([b >= tau for _,b in pairs])
    b_disc = int(np.sum(A & ~B))   # grok at A, not B
    c_disc = int(np.sum(~A & B))   # grok at B, not A
    p = mcnemar_exact(b_disc, c_disc)
    rA, rB = int(A.sum()), int(B.sum())
    loA, hiA = wilson(rA, n); loB, hiB = wilson(rB, n)
    drate = (rB - rA)/n
    ndisc = b_disc + c_disc
    md = float(np.mean([abs(a-b) for a,b in pairs]))
    mx = float(np.max([abs(a-b) for a,b in pairs]))
    print(f"\n{label}  (n={n}, tau={tau})")
    print(f"  flips (discordant): {b_disc} A-only / {c_disc} B-only  (total {ndisc})")
    print(f"  grok-rate A={rA}/{n} [{loA:.0%},{hiA:.0%}]  B={rB}/{n} [{loB:.0%},{hiB:.0%}]")
    print(f"  McNemar exact two-sided p = {p:.3f}  (n_discordant={ndisc} -> "
          f"{'NEGLIGIBLE' if ndisc<6 else 'LOW' if ndisc<12 else 'MODERATE'} power)")
    print(f"  aggregate rate diff = {drate:+.0%}; mean|dBest|={md:.3f}, max|dBest|={mx:.3f}")
    print(f"  -> cannot exclude a directional bias up to ~+/-{max(hiA-rA/n, rA/n-loA):.0%} "
          f"(Wilson half-width); report as 'no DETECTABLE bias', not 'unbiased'.")

paired_stats(thr_pairs, "Threads t1-vs-t4")
paired_stats(dev_pairs, "Device CPU-vs-GPU")
print()
