"""
Aggregate the multi-seed grokking sweep (script 14 outputs results/14_seed*.json) into
the trustworthy, seed-robust view: per-task grok-rate, mean +/- std test acc, and the
grok-epoch distribution; plus the decomposition (pipeline vs monolith) win-rate.

Usage: python 15_aggregate_grok.py
"""
import os, glob, json, statistics as st

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
files = sorted(glob.glob(os.path.join(RES, "14_seed*.json")))
runs = [json.load(open(f)) for f in files]
n = len(runs)
print(f"seeds aggregated: {n}  ({', '.join(str(r['seed']) for r in runs)})")
if not n: raise SystemExit("no seed files yet")

# data coverage per task family (train cap / full space) for context
COV = {"add2":"80/100", "mul2":"80/100", "add3":"600/1000", "muladd":"600/1000",
       "add4":"800/10000", "mul2add2":"800/10000"}
TASKS = ["add2", "mul2", "add3", "muladd", "add4", "mul2add2"]

def pct(x): return f"{x*100:.0f}%"
print("\n== Single-agent capability frontier (multi-seed) ==")
print(f"{'task':9s} {'cov':11s} {'grok-rate':9s} {'mean':5s} {'std':5s} {'min':4s} {'max':4s}  grok-ep(median of grokked)")
for t in TASKS:
    bests = [r["tasks"][t]["best"] for r in runs]
    groks = [r["tasks"][t]["grok"] for r in runs]
    geps = [r["tasks"][t]["grok_ep"] for r in runs if r["tasks"][t]["grok_ep"] is not None]
    gr = sum(groks) / n
    mep = int(st.median(geps)) if geps else None
    sd = st.pstdev(bests)
    print(f"{t:9s} {COV[t]:11s} {sum(groks)}/{n}={pct(gr):5s} {pct(st.mean(bests)):5s} "
          f"{pct(sd):5s} {pct(min(bests)):4s} {pct(max(bests)):4s}  {mep}")

print("\n== Specialists (for pipelines) ==")
for k in ["mul", "add"]:
    v = [r["specialists"][k] for r in runs]
    print(f"  {k}: mean {pct(st.mean(v))}  std {pct(st.pstdev(v))}  range {pct(min(v))}-{pct(max(v))}")

print("\n== Decomposition: pipeline vs monolith ==")
for name, pk, mk in [("muladd  (mul->add)", "muladd_pipe", "muladd_mono"),
                     ("mul2add2(mul,mul->add)", "mul2add2_pipe", "mul2add2_mono")]:
    pipe = [r["pipelines"][pk] for r in runs]
    mono = [r["pipelines"][mk] for r in runs]
    wins = sum(p > m for p, m in zip(pipe, mono))
    deltas = [p - m for p, m in zip(pipe, mono)]
    print(f"  {name:24s} pipe {pct(st.mean(pipe))} vs mono {pct(st.mean(mono))}  "
          f"pipe-wins {wins}/{n}  mean Δ {pct(st.mean(deltas)):>5s}")

# headline correlations
print("\n== Notes ==")
add_only = ["add2", "add3", "add4"]
mul_inv = ["mul2", "muladd", "mul2add2"]
def fam_rate(fam): return st.mean([r["tasks"][t]["grok"] for r in runs for t in fam])
print(f"  grok-rate: addition-only {pct(fam_rate(add_only))}  vs  multiplication-involving {pct(fam_rate(mul_inv))}")
print(f"  mean wall-clock / seed: {st.mean([r['secs'] for r in runs]):.0f}s")
