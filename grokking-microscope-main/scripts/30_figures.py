"""
Generate the 5 publication figures for the grokking-governance paper from the result JSONs.
Colorblind-safe Okabe-Ito palette, 300 dpi, PDF + PNG into figures/.

F1: grok-rate vs coverage per modulus/structure       (16_mono + 18_*_mono)
F2: grok-rate vs weight decay (Omnigrok inverted-U)    (22_wd*_t4)
F3: paired per-seed flip scatter, threads & device      (22 t1/t4 + 26 cpu/cuda)
F4: logit-Fourier vs held-out accuracy                  (20_fourier_m*)
F5: pipeline vs monolith matched budget + SEEN/HELDOUT  (25_decomp + 24_wall)

Usage: python 30_figures.py
"""
import os, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito colorblind-safe palette
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000"}
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

def loadall(pat):
    out = []
    for f in glob.glob(os.path.join(RES, pat)):
        try: out.append(json.load(open(f)))
        except Exception: pass
    return out

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf + .png")

# ---------------------------------------------------------------- F1: coverage
def fig1():
    # gather grok-rate vs coverage per family. 16_mono = mul2add2_m10; 18_*_mono = others
    series = {}  # label -> {coverage: [grok bools]}
    for r in loadall("16_mono_s*_n*.json"):
        cov = r.get("coverage") or round(r["n_train"]/10000, 3)
        series.setdefault("(a*b+c*d) mod 10", {}).setdefault(cov, []).append(r["grok"])
    fam_label = {"muladd_m7": "(a*b+c*d) mod 7", "muladd_m9": "(a*b+c*d) mod 9",
                 "addmul_m10": "((a+b)*(c+d)) mod 10"}
    for r in loadall("18_*_mono_s*_n*.json"):
        fam = r["family"]; lab = fam_label.get(fam, fam)
        cov = r.get("coverage") or round(r["n_train"]/10000, 3)
        series.setdefault(lab, {}).setdefault(cov, []).append(r["grok"])
    order = ["(a*b+c*d) mod 7", "(a*b+c*d) mod 9", "((a+b)*(c+d)) mod 10", "(a*b+c*d) mod 10"]
    colors = {"(a*b+c*d) mod 7": OI["sky"], "(a*b+c*d) mod 9": OI["green"],
              "((a+b)*(c+d)) mod 10": OI["orange"], "(a*b+c*d) mod 10": OI["vermillion"]}
    marks = {"(a*b+c*d) mod 7": "o", "(a*b+c*d) mod 9": "s",
             "((a+b)*(c+d)) mod 10": "^", "(a*b+c*d) mod 10": "D"}
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for lab in order:
        if lab not in series: continue
        covs = sorted(series[lab])
        rate = [100*np.mean(series[lab][c]) for c in covs]
        ls = "--" if "mod 10" in lab else "-"
        ax.plot([100*c for c in covs], rate, ls, marker=marks[lab], color=colors[lab],
                label=lab, markersize=5, linewidth=1.6)
    ax.axhline(50, color="0.7", linewidth=0.8, linestyle=":", zorder=0)
    ax.set_xlabel("training coverage (% of input domain)")
    ax.set_ylabel("grok-rate (%)")
    ax.set_title("Coverage-gated transition; threshold set by modulus")
    ax.set_ylim(-5, 105); ax.legend(frameon=False, loc="lower right")
    save(fig, "fig1_coverage")

# ---------------------------------------------------------------- F2: weight decay
def fig2():
    rows = [r for r in loadall("22_wd*_t*_s*.json") if r["threads"] == 4]
    wds = sorted({r["wd"] for r in rows})
    rate, mean_best, err = [], [], []
    for w in wds:
        g = [r for r in rows if r["wd"] == w]
        rate.append(100*np.mean([x["grok"] for x in g]))
        mean_best.append(100*np.mean([x["best"] for x in g]))
        err.append(100*np.std([x["best"] for x in g]))
    x = np.arange(len(wds))
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(x, rate, "-o", color=OI["blue"], markersize=6, linewidth=1.8, label="grok-rate")
    ax.errorbar(x, mean_best, yerr=err, fmt="--s", color=OI["orange"], markersize=5,
                linewidth=1.2, capsize=3, label="mean best acc", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels([f"{w:g}" for w in wds])
    ax.set_xlabel("weight decay"); ax.set_ylabel("percent")
    ax.set_title("Omnigrok inverted-U at 12K params")
    ax.set_ylim(-5, 105); ax.legend(frameon=False, loc="upper left")
    # annotate peak
    pk = int(np.argmax(rate))
    ax.annotate(f"{rate[pk]:.0f}%", (x[pk], rate[pk]), textcoords="offset points",
                xytext=(-14, -2), ha="right", fontsize=8, color=OI["blue"])
    save(fig, "fig2_weightdecay")

# ---------------------------------------------------------------- F3: knife-edge flips
def fig3():
    # threads: pair t1 vs t4 at wd=0.01 ; device: pair cpu vs cuda
    def pair_thread():
        by = {}
        for r in loadall("22_wd*_t*_s*.json"):
            if abs(r["wd"]-0.01) < 1e-9 and r["threads"] in (1, 4):
                by.setdefault(r["seed"], {})[r["threads"]] = r["best"]
        return [(d[1], d[4]) for d in by.values() if 1 in d and 4 in d]
    def pair_dev():
        by = {}
        for r in loadall("26_dev*_s*.json"):
            by.setdefault(r["seed"], {})[r["device"]] = r["best"]
        return [(d["cpu"], d["cuda"]) for d in by.values() if "cpu" in d and "cuda" in d]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
    for ax, pairs, (xl, yl), title in [
        (axes[0], pair_thread(), ("1 thread", "4 threads"), "Reduction order (threads)"),
        (axes[1], pair_dev(), ("CPU", "GPU"), "Execution device")]:
        a = np.array([p[0] for p in pairs])*100; b = np.array([p[1] for p in pairs])*100
        thr = 70
        flip = ((a >= thr) != (b >= thr))
        ax.plot([-5, 105], [-5, 105], color="0.7", linewidth=0.8, linestyle=":", zorder=0)
        ax.axhline(thr, color="0.85", linewidth=0.7, zorder=0)
        ax.axvline(thr, color="0.85", linewidth=0.7, zorder=0)
        ax.scatter(a[~flip], b[~flip], color=OI["blue"], s=28, label="same grok status",
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter(a[flip], b[flip], color=OI["vermillion"], s=46, marker="D",
                   label="grok flipped", edgecolor="black", linewidth=0.6, zorder=4)
        ax.set_xlabel(f"best acc — {xl} (%)"); ax.set_ylabel(f"best acc — {yl} (%)")
        ax.set_title(f"{title}: {flip.sum()}/{len(pairs)} flip")
        ax.set_xlim(-5, 105); ax.set_ylim(-5, 105); ax.set_aspect("equal")
        ax.legend(frameon=False, loc="lower right", fontsize=7)
    fig.suptitle("Grokking on a numerical knife-edge: same seed, only the float environment differs",
                 fontsize=9.5, y=1.02)
    save(fig, "fig3_knifeedge")

# ---------------------------------------------------------------- F4: mechanism
def fig4():
    # dedup IDENTICALLY to scripts/21_aggregate_mechanism.py so figure r-values match the paper
    rows = []
    for M in (7, 10):
        files = glob.glob(os.path.join(RES, f"20_fourier_m{M}.json")) + \
                glob.glob(os.path.join(RES, f"20_fourier_m{M}_s*.json"))
        seen = set()
        for f in files:
            try: data = json.load(open(f))
            except Exception: continue
            for x in data["results"]:
                if x["seed"] in seen: continue
                seen.add(x["seed"]); rows.append((M, x))
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for M, col, mk in [(7, OI["sky"], "o"), (10, OI["vermillion"], "D")]:
        sub = [x for (m, x) in rows if m == M]
        if not sub: continue
        held = np.array([x["heldout_acc"] for x in sub])*100
        lf = np.array([x["logit_fourier"] for x in sub])
        ax.scatter(lf, held, color=col, s=30, marker=mk, alpha=0.85,
                   edgecolor="white", linewidth=0.4, label=f"M={M}")
        if held.std() > 1e-6 and lf.std() > 1e-6:
            r_ = np.corrcoef(lf, held)[0, 1]
            z = np.polyfit(lf, held, 1); xs = np.linspace(lf.min(), lf.max(), 50)
            ax.plot(xs, np.polyval(z, xs), color=col, linewidth=1.3, linestyle="--", alpha=0.8)
            ax.plot([], [], " ", label=f"  r={r_:+.2f}")
    ax.set_xlabel("logit Fourier concentration (output periodicity)")
    ax.set_ylabel("held-out accuracy (%)")
    ax.set_title("Generalization tracks output periodicity")
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    save(fig, "fig4_mechanism")

# ---------------------------------------------------------------- F5: decomposition
def fig5():
    dec = {}
    for r in loadall("25_decomp_wd*_s*.json"):
        dec.setdefault(r["wd"], []).append(r)
    wds = sorted(dec)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    # (a) grok-rate mono vs pipe per wd
    ax = axes[0]
    x = np.arange(len(wds)); w = 0.36
    mono = [100*np.mean([z["monolith"]["grok"] for z in dec[k]]) for k in wds]
    pipe = [100*np.mean([z["pipeline"]["grok"] for z in dec[k]]) for k in wds]
    ax.bar(x - w/2, mono, w, color=OI["orange"], label="monolith")
    ax.bar(x + w/2, pipe, w, color=OI["blue"], label="pipeline")
    for xi, v in zip(x - w/2, mono): ax.text(xi, v+2, f"{v:.0f}", ha="center", fontsize=7.5)
    for xi, v in zip(x + w/2, pipe): ax.text(xi, v+2, f"{v:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([f"wd={k:g}" for k in wds])
    ax.set_ylabel("grok-rate (%)"); ax.set_ylim(0, 112)
    ax.set_title("Matched 160-example budget")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    # (b) SEEN vs HELDOUT at wd=0.1 (memorization signature)
    ax = axes[1]
    sub = sorted(dec.get(0.1, []), key=lambda z: z["seed"])
    seen = [z["pipeline"]["acc_adder_pair_SEEN"] for z in sub]
    held = [z["pipeline"]["acc_adder_pair_HELDOUT"] for z in sub]
    seen = [100*s if s is not None else np.nan for s in seen]
    held = [100*h if h is not None else np.nan for h in held]
    xs = np.arange(len(sub))
    ax.plot(xs, seen, "-o", color=OI["green"], markersize=5, label="adder pair SEEN")
    ax.plot(xs, held, "--s", color=OI["purple"], markersize=5, label="adder pair HELD-OUT")
    ax.set_xticks(xs); ax.set_xticklabels([z["seed"] for z in sub], fontsize=7)
    ax.set_xlabel("seed"); ax.set_ylabel("pipeline accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_title("Memorization signature (wd=0.1)")
    ax.legend(frameon=False, loc="lower left", fontsize=7.5)
    fig.suptitle("Decomposition as data efficiency: pipeline groks where the monolith cannot",
                 fontsize=9.5, y=1.02)
    save(fig, "fig5_decomposition")

# ---------------------------------------------------------------- F6: cardinality
def fig6():
    import collections
    card = loadall("33_card_*.json")
    g18 = loadall("18_*_mono_s*_n*.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    # Panel A: grok-rate vs M at domain 10^3, per coverage
    ax = axes[0]
    by = collections.defaultdict(list)
    for r in card: by[(r["n_train"], r["mod"])].append(r["best"] >= 0.70)
    covmap = {300: ("30% coverage", OI["sky"], "o", "-"),
              600: ("60% coverage", OI["vermillion"], "D", "-")}
    for nt, (lab, col, mk, ls) in covmap.items():
        Ms = sorted({m for (n, m) in by if n == nt})
        if not Ms: continue
        rate = [100*np.mean(by[(nt, m)]) for m in Ms]
        ax.plot(Ms, rate, ls, marker=mk, color=col, markersize=5, linewidth=1.6, label=lab)
    ax.axhline(50, color="0.7", linewidth=0.8, linestyle=":", zorder=0)
    ax.set_xlabel("modulus M (output cardinality)"); ax.set_ylabel("grok-rate (%)")
    ax.set_title(r"Domain $10^3$: $(a\cdot b + c)\,\mathrm{mod}\,M$")
    ax.set_ylim(-5, 105)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, fontsize=8)
    # Panel B: 2x2 cardinality x structure at n=2000 (domain 10^4)
    ax = axes[1]
    cell = collections.defaultdict(list)
    for r in g18:
        if r.get("n_train") == 2000 and r["family"] in ("muladd_m8","addmul_m8","muladd_m10","addmul_m10"):
            cell[r["family"]].append(r["best"] >= 0.70)
    groups = [("M=8", "muladd_m8", "addmul_m8"), ("M=10", "muladd_m10", "addmul_m10")]
    x = np.arange(len(groups)); w = 0.36
    mul = [100*np.mean(cell[g[1]]) if cell[g[1]] else np.nan for g in groups]
    add = [100*np.mean(cell[g[2]]) if cell[g[2]] else np.nan for g in groups]
    ax.bar(x - w/2, mul, w, color=OI["blue"], label="sum-of-products")
    ax.bar(x + w/2, add, w, color=OI["orange"], label="product-of-sums")
    for xi, v in zip(x - w/2, mul):
        if not np.isnan(v): ax.text(xi, v+2, f"{v:.0f}", ha="center", fontsize=7.5)
    for xi, v in zip(x + w/2, add):
        if not np.isnan(v): ax.text(xi, v+2, f"{v:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("grok-rate (%)"); ax.set_ylim(0, 112)
    ax.set_title(r"Domain $10^4$, $n=2000$: cardinality vs structure")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle("Cardinality governs the threshold across a ten-fold change in domain size",
                 fontsize=9.5, y=1.02)
    save(fig, "fig6_cardinality")

# ---------------------------------------------------------------- F7: trajectory
def fig7():
    # grokking temporal dynamics (reviewer-2): representative grok seed (wd=0.1, s0) +
    # all-seed held-out curves for the transition-time lottery. (38_traj_wd0.1_s*.json)
    import math
    files = sorted(glob.glob(os.path.join(RES, "38_traj_wd0.1_s*.json")),
                   key=lambda p: int(p.split("_s")[-1].split(".")[0]))
    runs = [json.load(open(f)) for f in files]
    rep = next(r for r in runs if r["seed"] == 0)         # representative seed
    h = rep["hist"]; ep = np.array(h["epoch"])
    tra = np.array(h["train_acc"]) * 100; tea = np.array(h["test_acc"]) * 100
    trl = np.array(h["train_loss"]); tel = np.array(h["test_loss"])
    lnM = math.log(rep["M"])                               # ln 10 = 2.303 = chance CE
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    # (a) accuracy: representative train vs held-out + faint all-seed held-out (time lottery)
    ax = axes[0]
    for r in runs:
        if r["seed"] == 0: continue
        e = np.array(r["hist"]["epoch"]); t = np.array(r["hist"]["test_acc"]) * 100
        ax.plot(e, t, color="0.85", linewidth=0.6, zorder=1)
    ax.plot([], [], color="0.85", linewidth=0.6, label="held-out, other seeds")
    ax.fill_between(ep, tea, tra, where=(tra >= tea), color=OI["sky"], alpha=0.20,
                    zorder=1, label="train $-$ held-out gap")
    ax.plot(ep, tra, "-", color=OI["blue"], linewidth=1.8, label="train (seed 0)", zorder=3)
    ax.plot(ep, tea, "--", color=OI["vermillion"], linewidth=1.8, label="held-out (seed 0)", zorder=3)
    ax.axhline(10, color="0.6", linewidth=0.8, linestyle=":", zorder=0)
    ax.text(ep[-1], 12, "chance $1/M$", ha="right", va="bottom", fontsize=7, color="0.45")
    ax.set_xlabel("epoch"); ax.set_ylabel("answer-token accuracy (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("Chance plateau, then a seed-timed transition")
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    # (b) loss: representative train vs held-out with the ln(M) chance plateau line
    ax = axes[1]
    ax.axhline(lnM, color="0.6", linewidth=0.9, linestyle=":", zorder=0)
    ax.text(ep[-1], lnM + 0.06, r"$\ln M$ (chance)", ha="right", va="bottom", fontsize=7, color="0.45")
    ax.plot(ep, trl, "-", color=OI["blue"], linewidth=1.8, label="train loss")
    ax.plot(ep, tel, "--", color=OI["vermillion"], linewidth=1.8, label="held-out loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("answer-token cross-entropy")
    ax.set_ylim(0, 3.6)
    ax.set_title(r"Loss plateau at $\ln M$, then collapse")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    save(fig, "fig7_trajectory")

if __name__ == "__main__":
    print("generating figures ->", FIG)
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print("done.")
