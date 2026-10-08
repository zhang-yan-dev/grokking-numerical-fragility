"""
Pool all script-20 outputs (results/20_fourier_m<M>*.json) and test the mechanistic
hypothesis with POWER: across all models per modulus, does held-out accuracy correlate
with effective rank (expect negative: generalizers are lower-rank) and with logit Fourier
concentration (expect positive: generalizers are more periodic)?

Correlation across the full continuum sidesteps the grokker/memorizer class imbalance.

Usage: python 21_aggregate_mechanism.py
"""
import os, glob, json
import numpy as np

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

def pearson(x, y):
    x = np.array(x, float); y = np.array(y, float)
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9: return None
    return float(np.corrcoef(x, y)[0, 1])

for M in (7, 10):
    files = glob.glob(os.path.join(RES, f"20_fourier_m{M}.json")) + \
            glob.glob(os.path.join(RES, f"20_fourier_m{M}_s*.json"))
    seen, rows = set(), []
    for f in files:
        for r in json.load(open(f))["results"]:
            if r["seed"] in seen: continue
            seen.add(r["seed"]); rows.append(r)
    if not rows:
        print(f"M={M}: no data"); continue
    held = [r["heldout_acc"] for r in rows]
    er = [r["eff_rank"] for r in rows]
    lf = [r["logit_fourier"] for r in rows]
    ef = [r["emb_fourier"] for r in rows]
    gk = [r for r in rows if r["grok"]]; mm = [r for r in rows if not r["grok"]]
    def avg(rows_, k): return round(float(np.mean([r[k] for r in rows_])), 3) if rows_ else None
    print(f"\n===== M={M}  (n={len(rows)} models, grokkers={len(gk)} memorizers={len(mm)}) =====")
    print(f"  correlations with held-out acc:")
    print(f"    eff_rank      r = {pearson(held, er)}   (expect NEGATIVE: generalizers lower-rank)")
    print(f"    logit_fourier r = {pearson(held, lf)}   (expect POSITIVE: generalizers more periodic)")
    print(f"    emb_fourier   r = {pearson(held, ef)}")
    print(f"  group means        grokkers / memorizers:")
    print(f"    eff_rank      {avg(gk,'eff_rank')} / {avg(mm,'eff_rank')}")
    print(f"    logit_fourier {avg(gk,'logit_fourier')} / {avg(mm,'logit_fourier')}")
    print(f"    emb_fourier   {avg(gk,'emb_fourier')} / {avg(mm,'emb_fourier')}")
