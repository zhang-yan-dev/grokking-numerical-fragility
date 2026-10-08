"""Figure: where do cross-machine seed flips actually happen?

Left  : flip rate as a function of the author's held-out accuracy, vs tau = 0.70.
Right : paired scatter, author best vs our best, colored by whether grok status flipped.

Data: n=300 paired seeds, wd=0.01, threads=4 (author's published records vs our retrain).
"""
import json, os, glob, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUTH = r"D:\research\author-results-backup"
OURS = r"D:\research\run-ours\results"
WD, TH = "0.01", "4"
TAU = 0.70

def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, f"22_wd{WD}_t{TH}_s*.json")):
        s = int(os.path.basename(f).rsplit("_s", 1)[1].split(".")[0])
        out[s] = json.load(open(f, encoding="utf-8"))
    return out

a, b = load(AUTH), load(OURS)
seeds = sorted(set(a) & set(b))
ba = np.array([a[s]["best"] for s in seeds])
bb = np.array([b[s]["best"] for s in seeds])
ga = np.array([a[s]["grok"] for s in seeds])
gb = np.array([b[s]["grok"] for s in seeds])
flip = ga != gb
n = len(seeds)

def wilson(k, m, z=1.96):
    if m == 0:
        return 0.0, 0.0, 0.0
    p = k / m
    den = 1 + z * z / m
    ctr = (p + z * z / (2 * m)) / den
    half = z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / den
    return p, max(0.0, ctr - half), min(1.0, ctr + half)

# equal-width bins, keep only those with enough seeds
edges = np.arange(0.0, 1.01, 0.1)
centers, rates, los, his, ns = [], [], [], [], []
print(f"{'bin':>14}{'n':>6}{'flips':>7}{'rate':>8}   95% CI")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (ba >= lo) & (ba < hi) if hi < 1.0 else (ba >= lo) & (ba <= hi)
    k = int(flip[m].sum())
    mm = int(m.sum())
    if mm < 10:
        continue
    p, l, u = wilson(k, mm)
    centers.append((lo + hi) / 2); rates.append(p); los.append(p - l); his.append(u - p); ns.append(mm)
    print(f"{f'[{lo:.1f},{hi:.1f})':>14}{mm:>6}{k:>7}{p:>7.0%}   [{l:.0%}, {u:.0%}]")

centers = np.array(centers); rates = np.array(rates)
los = np.array(los); his = np.array(his); ns = np.array(ns)
width = 0.1 * 0.85

# aggregate contrast: near the threshold vs far from it
near = np.abs(ba - TAU) <= 0.10
p_near, l_near, u_near = wilson(int(flip[near].sum()), int(near.sum()))
p_far, l_far, u_far = wilson(int(flip[~near].sum()), int((~near).sum()))
print(f"  within +/-0.10 of tau  n={int(near.sum()):>4}  flip rate {p_near:.1%}  [{l_near:.0%}, {u_near:.0%}]")
print(f"  farther than 0.10      n={int((~near).sum()):>4}  flip rate {p_far:.1%}  [{l_far:.0%}, {u_far:.0%}]")
print(f"  ratio = {p_near/p_far:.2f}x")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

# ---- left: flip rate vs accuracy ----
bars = ax1.bar(centers, rates, width=width, color="#c0392b", alpha=0.85,
               edgecolor="black", linewidth=0.8,
               yerr=np.vstack([los, his]), capsize=3,
               error_kw=dict(ecolor="black", lw=1.0))
ax1.axvline(TAU, color="black", ls="--", lw=1.6)
ax1.text(TAU + 0.012, 0.60, f"$\\tau$ = {TAU}", fontsize=10, va="bottom")
for c, m in zip(centers, ns):
    ax1.text(c, 0.012, f"n={m}", ha="center", va="bottom", fontsize=7.5, color="#555555")
ax1.set_xlim(0, 1.0)
ax1.set_ylim(0, 0.78)
ax1.set_xticks(np.arange(0, 1.01, 0.2))
ax1.set_xlabel("author's held-out accuracy  (best)")
ax1.set_ylabel("fraction of seeds whose grok status flipped")
ax1.set_title("Flips concentrate at the decision threshold")
ax1.grid(axis="y", alpha=0.25, lw=0.6)
ax1.set_axisbelow(True)

ax1.text(0.02, 0.97,
         f"within $\\pm$0.10 of $\\tau$ : {p_near:.1%}   (n={int(near.sum())})\n"
         f"farther than 0.10        : {p_far:.1%}   (n={int((~near).sum())})\n"
         f"ratio                    : {p_near/p_far:.1f}$\\times$",
         transform=ax1.transAxes, fontsize=8.5, va="top", family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=0.8, alpha=0.92))

print(f"\ntotal: {n} seeds, {int(flip.sum())} flipped ({flip.mean():.1%})")

# ---- right: paired scatter ----
ax2.scatter(ba[~flip], bb[~flip], s=14, c="#2c7fb8", alpha=0.55, label=f"same grok status (n={int((~flip).sum())})")
ax2.scatter(ba[flip], bb[flip], s=22, c="#c0392b", alpha=0.85, marker="x", label=f"FLIPPED (n={int(flip.sum())})")
ax2.plot([0, 1], [0, 1], color="black", ls=":", lw=1.0)
ax2.axhline(TAU, color="gray", ls="--", lw=0.9)
ax2.axvline(TAU, color="gray", ls="--", lw=0.9)
ax2.set_xlim(-0.02, 1.02); ax2.set_ylim(-0.02, 1.02)
ax2.set_xlabel("author's best accuracy")
ax2.set_ylabel("our best accuracy")
ax2.set_title(f"Same seed, different machine  (n={n})")
ax2.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax2.grid(alpha=0.2, lw=0.6)
ax2.set_axisbelow(True)

plt.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(r"D:\research", f"flip_vs_threshold.{ext}")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    print("wrote", p)
