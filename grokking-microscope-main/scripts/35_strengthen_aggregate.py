"""
Aggregate the reviewer-strengthening grids into paper-ready tables.

  (1) Cardinality at domain 10^3 (script 33): grok-rate vs M at each coverage, with Wilson CIs.
      Tests whether the M-gradient replicates on a domain ten-fold smaller than the 10^4 composites.
  (2) Structure-vs-cardinality 2x2 (script 18 mono @ n=3000): grok-rate for
      {muladd,addmul} x {M8,M10}. Within-M structure gap vs across-M cardinality gap.
  (3) Context: existing 4-input thresholds (18 muladd_m7/m9, 16 muladd_m10) for the M-gradient.

Usage: python 35_strengthen_aggregate.py
"""
import os, glob, json, math, collections
RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

def loadall(pat):
    out = []
    for f in glob.glob(os.path.join(RES, pat)):
        try: out.append(json.load(open(f)))
        except Exception: pass
    return out

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = p + z*z/(2*n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-h)/d, (c+h)/d)

def rate_ci(rs, tau=0.70):
    n = len(rs); k = sum(r["best"] >= tau for r in rs)
    lo, hi = wilson(k, n)
    mb = sum(r["best"] for r in rs)/n if n else 0
    return n, k, lo, hi, mb

print("="*72)
print("(1) CARDINALITY at domain 10^3:  (a*b+c) mod M  -- grok-rate vs M per coverage")
print("="*72)
card = loadall("33_card_*.json")
if not card:
    print("  (no 33_card results yet)")
else:
    by = collections.defaultdict(list)
    for r in card: by[(r["mod"], r["n_train"])].append(r)
    covs = sorted({r["n_train"] for r in card})
    Ms = sorted({r["mod"] for r in card})
    for nt in covs:
        cov = nt/1000
        print(f"\n coverage {cov:.0%} (n_train={nt}):")
        print(f"   {'M':>3} {'grok':>8} {'95% CI':>14} {'meanbest':>9}")
        for M in Ms:
            rs = by.get((M, nt), [])
            if not rs: continue
            n,k,lo,hi,mb = rate_ci(rs)
            print(f"   {M:>3} {k:>3}/{n:<3}  [{lo:4.0%},{hi:4.0%}]   {mb:>7.2f}")

print("\n" + "="*72)
print("(2) STRUCTURE x CARDINALITY 2x2 (domain 10^4): BOTH budgets")
print("    n=2000 = M10 transition band (dissociation measurable);")
print("    n=3000 = saturation (both moduli grok -> both gaps vanish, as expected).")
print("="*72)
g18 = loadall("18_*_mono_*.json")
order = [("muladd_m8",8,"muladd"),("addmul_m8",8,"addmul"),
         ("muladd_m10",10,"muladd"),("addmul_m10",10,"addmul")]
for budget in (2000, 3000):
    cell = collections.defaultdict(list)
    for r in g18:
        if r.get("n_train") == budget and r["family"] in dict((o[0],1) for o in order):
            cell[r["family"]].append(r)
    print(f"\n n_train={budget}:")
    print(f"   {'family':>12} {'M':>3} {'struct':>8} {'grok':>8} {'95% CI':>14} {'meanbest':>9}")
    res2x2 = {}
    for fam, M, st in order:
        rs = cell.get(fam, [])
        if not rs:
            print(f"   {fam:>12} {M:>3} {st:>8}   (pending)"); continue
        n,k,lo,hi,mb = rate_ci(rs)
        res2x2[fam] = (k/n, mb, n)
        print(f"   {fam:>12} {M:>3} {st:>8} {k:>3}/{n:<3}  [{lo:4.0%},{hi:4.0%}]   {mb:>7.2f}")
    if len(res2x2) == 4:
        dm8 = abs(res2x2["muladd_m8"][0]-res2x2["addmul_m8"][0])
        dm10 = abs(res2x2["muladd_m10"][0]-res2x2["addmul_m10"][0])
        dmul = abs(res2x2["muladd_m8"][0]-res2x2["muladd_m10"][0])
        dadd = abs(res2x2["addmul_m8"][0]-res2x2["addmul_m10"][0])
        struct = (dm8+dm10)/2; card = (dmul+dadd)/2
        print(f"   within-M structure gap (mean) ={struct:.0%}; "
              f"across-M cardinality gap (mean) ={card:.0%}  "
              f"-> cardinality {'DOMINATES' if card > struct+0.2 else 'approx equals'} structure")

print("\n" + "="*72)
print("(3) CONTEXT: 4-input coverage thresholds (existing data), M-gradient at domain 10^4")
print("="*72)
ctx = collections.defaultdict(list)
for r in g18:
    if r["family"] in ("muladd_m7","muladd_m9"):
        ctx[(r["family"], r["n_train"])].append(r)
for r in loadall("16_mono_*.json"):  # muladd_m10
    ctx[("muladd_m10", r["n_train"])].append(r)
fams = ["muladd_m7","muladd_m9","muladd_m10"]
ns = sorted({k[1] for k in ctx})
print(f"   {'family':>12} " + " ".join(f"n={n}" for n in ns))
for fam in fams:
    row = []
    for n in ns:
        rs = ctx.get((fam,n), [])
        row.append(f"{sum(x['best']>=0.7 for x in rs)}/{len(rs)}" if rs else "-")
    print(f"   {fam:>12} " + "  ".join(f"{c:>5}" for c in row))
print()
