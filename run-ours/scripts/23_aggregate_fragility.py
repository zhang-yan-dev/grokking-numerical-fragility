"""
Aggregate script-22 numerical-fragility runs into grok-rate per knob value.

Pillar A of the paper: grokking of (a+b) mod 10 is a knife-edge NUMERICAL phenomenon.
Two slices:
  * THREAD slice  (wd fixed at 0.01): grok-rate vs CPU thread count {1,4,16}.
    A nonflat curve = changing only the float reduction order flips generalization.
  * WD slice (threads fixed at 4): grok-rate vs weight_decay {0,0.01,0.1,1.0}.
    Should rise with wd (Omnigrok) -> validates the setup.

Usage: python 23_aggregate_fragility.py
"""
import os, glob, json
import numpy as np

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

rows = []
for f in glob.glob(os.path.join(RES, "22_wd*_t*_s*.json")):
    try:
        rows.append(json.load(open(f)))
    except Exception as e:
        print(f"skip {os.path.basename(f)}: {e}")

if not rows:
    print("no 22_*.json results yet")
    raise SystemExit

def summarize(group):
    """group: list of run dicts -> (n, grok_rate, best_mean, best_std, grok_ep_median)"""
    n = len(group)
    gr = sum(1 for r in group if r["grok"]) / n
    best = [r["best"] for r in group]
    geps = [r["grok_ep"] for r in group if r["grok_ep"] is not None]
    return n, gr, float(np.mean(best)), float(np.std(best)), (int(np.median(geps)) if geps else None)

def fkey(x):
    # robust float compare for wd label
    return round(float(x), 4)

print(f"total runs: {len(rows)}")

# ---- THREAD slice: wd == 0.01 ----
print("\n===== THREAD slice (wd=0.01) : grok-rate vs CPU thread count =====")
print(f"{'threads':>8} {'n':>3} {'grok_rate':>10} {'best_mean':>10} {'best_std':>9} {'grok_ep_med':>12}")
thr = sorted({r["threads"] for r in rows if fkey(r["wd"]) == 0.01})
for t in thr:
    g = [r for r in rows if r["threads"] == t and fkey(r["wd"]) == 0.01]
    n, gr, bm, bs, ge = summarize(g)
    print(f"{t:>8} {n:>3} {gr:>10.0%} {bm:>10.2f} {bs:>9.2f} {str(ge):>12}")

# ---- WD slice: threads == 4 ----
print("\n===== WD slice (threads=4) : grok-rate vs weight_decay (Omnigrok control) =====")
print(f"{'wd':>8} {'n':>3} {'grok_rate':>10} {'best_mean':>10} {'best_std':>9} {'grok_ep_med':>12}")
wds = sorted({fkey(r["wd"]) for r in rows if r["threads"] == 4})
for w in wds:
    g = [r for r in rows if r["threads"] == 4 and fkey(r["wd"]) == w]
    n, gr, bm, bs, ge = summarize(g)
    print(f"{w:>8} {n:>3} {gr:>10.0%} {bm:>10.2f} {bs:>9.2f} {str(ge):>12}")

# ---- PAIRED t1-vs-t4 natural experiment (the pillar-A headline) ----
# Same seed, only the CPU thread count differs -> only the float reduction ORDER changes.
# If individual grok outcomes flip while the aggregate rate stays put, grokking sits on a
# numerical knife-edge perturbed by reduction-order noise alone.
print("\n===== PAIRED reduction-order experiment: t=1 vs t=4 (wd=0.01, same seeds) =====")
by = {}
for r in rows:
    if fkey(r["wd"]) == 0.01 and r["threads"] in (1, 4):
        by.setdefault(r["seed"], {})[r["threads"]] = r
paired = [(s, d[1], d[4]) for s, d in by.items() if 1 in d and 4 in d]
if paired:
    dbest = [abs(a["best"] - b["best"]) for _, a, b in paired]
    flips = sum(1 for _, a, b in paired if a["grok"] != b["grok"])
    g1 = sum(1 for _, a, _b in paired if a["grok"]); g4 = sum(1 for _, _a, b in paired if b["grok"])
    n = len(paired)
    print(f"  paired seeds              : {n}")
    print(f"  mean |delta best|         : {np.mean(dbest):.3f}")
    print(f"  max  |delta best|         : {np.max(dbest):.3f}")
    print(f"  grok-status FLIPS         : {flips}/{n}  ({flips/n:.0%} of seeds change grok yes/no)")
    print(f"  grok-rate  t1 / t4        : {g1}/{n} ({g1/n:.0%})  vs  {g4}/{n} ({g4/n:.0%})   (~equal = unbiased)")
    big = sorted(paired, key=lambda p: -abs(p[1]["best"] - p[2]["best"]))[:5]
    print(f"  largest flips (seed: t1_best -> t4_best):")
    for s, a, b in big:
        print(f"    seed {s:>2}: {a['best']:.2f} -> {b['best']:.2f}")
else:
    print("  (need both t=1 and t=4 runs at wd=0.01)")

# ---- full grid coverage table (for sanity) ----
print("\n===== coverage (runs present per (wd,threads) cell) =====")
cells = {}
for r in rows:
    cells.setdefault((fkey(r["wd"]), r["threads"]), 0)
    cells[(fkey(r["wd"]), r["threads"])] += 1
for (w, t), c in sorted(cells.items()):
    print(f"  wd={w:<5} threads={t:<3} -> {c} runs")
