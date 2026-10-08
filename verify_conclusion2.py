"""Independent verification of Conclusion 2 (the inverted-U over weight decay).

The handover doc's section 4.2 verified this only at the ANALYSIS layer
(author's JSON re-aggregated with author's code). This script compares our
INDEPENDENTLY RETRAINED wd grid against the author's published records for the
same (wd, threads, seed) cells.

Our runs are tagged by their `secs` field (<200s = this laptop; author ran 30
processes concurrently, ~2000s). Anything >=200s in run-ours/results is still
the author's file and is reported as NOT retrained.

Outputs a JSON report + console table.
"""
import json, glob, os, math

AUTH = r"D:\research\author-results-backup"
OURS = r"D:\research\run-ours\results"
TH = "4"
WDS = ["0.0", "0.1", "1.0"]
OURS_SECS_CUTOFF = 200.0   # see handover doc 7.4: our runs ~60-75s, author ~2000s


def load(d, wd, probe_secs=False):
    out = {}
    for f in glob.glob(os.path.join(d, f"22_wd{wd}_t{TH}_s*.json")):
        s = int(os.path.basename(f).rsplit("_s", 1)[1].split(".")[0])
        j = json.load(open(f, encoding="utf-8"))
        if probe_secs:
            j["_ours"] = j.get("secs", 1e9) < OURS_SECS_CUTOFF
        out[s] = j
    return out


def wilson(k, m, z=1.96):
    if m == 0:
        return 0.0, 0.0, 0.0
    p = k / m
    den = 1 + z * z / m
    ctr = (p + z * z / (2 * m)) / den
    half = z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / den
    return p, max(0.0, ctr - half), min(1.0, ctr + half)


report = {"grid": {}, "independence_note": ""}

print("=" * 74)
print("Conclusion 2 (inverted-U over weight decay): independent retrain check")
print("=" * 74)

for wd in WDS:
    a = load(AUTH, wd)
    o = load(OURS, wd, probe_secs=True)
    ours = {s: j for s, j in o.items() if j["_ours"]}
    stale = {s: j for s, j in o.items() if not j["_ours"]}

    ka = int(sum(j["grok"] for j in a.values()))
    na = len(a)
    ko = int(sum(j["grok"] for j in ours.values()))
    no = len(ours)

    pa, la, ua = wilson(ka, na)
    entry = {
        "author": {"n": na, "grok": ka, "rate": round(pa, 4),
                   "ci": [round(la, 4), round(ua, 4)]},
        "ours": {"n": no, "grok": ko,
                 "rate": round(ko / no, 4) if no else None,
                 "ci": [round(x, 4) for x in wilson(ko, no)[1:]] if no else None},
        "seeds_not_retrained": sorted(s for s in stale),
    }

    print(f"\n--- wd = {wd}  (threads={TH}) ---")
    print(f"  author : grok {ka}/{na} = {pa:.0%}   95% CI [{la:.0%}, {ua:.0%}]")
    if no:
        po = ko / no
        _, lo, uo = wilson(ko, no)
        print(f"  ours   : grok {ko}/{no} = {po:.0%}   95% CI [{lo:.0%}, {uo:.0%}]")
        if stale:
            print(f"  NOTE   : {len(stale)} seed(s) in run-ours still hold AUTHOR data, "
                  f"not retrained: {sorted(stale)}")
    else:
        print(f"  ours   : NO independent runs found")

    # paired detail when we have both
    if ours:
        common = sorted(set(a) & set(ours))
        if common:
            flips = [s for s in common if a[s]["grok"] != ours[s]["grok"]]
            deltas = [ours[s]["best"] - a[s]["best"] for s in common]
            print(f"  paired : {len(common)} seeds, flips={len(flips)} {flips}")
            print(f"  best |delta|: mean {sum(abs(d) for d in deltas)/len(deltas):.3f}, "
                  f"max {max(abs(d) for d in deltas):.3f}")
            entry["paired"] = {
                "n": len(common),
                "flip_seeds": flips,
                "mean_abs_delta": round(sum(abs(d) for d in deltas) / len(deltas), 4),
                "max_abs_delta": round(max(abs(d) for d in deltas), 4),
            }
    report["grid"][wd] = entry

# inverted-U check on the author's series (should reproduce paper: 20% -> 90% -> 0%)
print("\n" + "=" * 74)
seq = []
for wd in WDS:
    e = report["grid"][wd]
    seq.append((wd, e["author"]["rate"], e["ours"]["rate"]))
print("inverted-U shape check")
print(f"{'wd':>6}{'author':>10}{'ours':>10}")
for wd, ra, ro in seq:
    rs = f"{ro:.0%}" if ro is not None else "n/a"
    print(f"{wd:>6}{ra:>10.0%}{rs:>10}")
report["author_sequence"] = [(wd, ra) for wd, ra, _ in seq]
report["ours_sequence"] = [(wd, ro) for wd, _, ro in seq]

with open(r"D:\research\_conclusion2_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("\nwrote D:\\research\\_conclusion2_report.json")
