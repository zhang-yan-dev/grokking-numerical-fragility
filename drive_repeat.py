"""Repeat a single (wd, threads, seed) N times to measure same-machine rerun spread.

Motivation: wd=0.1 seed 3 gave best=0.85 in one run and best=0.80 in another on
the SAME machine. The project's cross-machine noise scale is 0.164; if the
same-machine spread is comparable, then "trajectory divergence" is not specific
to cross-machine comparison at all.

Each repetition writes to results/ as usual, which would clobber the record. So
this script snapshots the file before, runs, records the value, and RESTORES the
original file at the end unless --keep is passed.

Usage: drive_repeat.py <wd> <threads> <seed> <n> [--keep]
"""
import json, os, shutil, subprocess, sys, time

ROOT = r"D:\research\run-ours"
PY = r"D:\venvs\grokking\Scripts\python.exe"
RESULTS = os.path.join(ROOT, "results")

env = dict(os.environ)
env["PYTHONUTF8"] = "1"
env["PYTHONIOENCODING"] = "utf-8"


def target(wd, threads, seed):
    return os.path.join(RESULTS, f"22_wd{wd}_t{threads}_s{seed}.json")


def main():
    wd, threads, seed, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    keep = "--keep" in sys.argv[5:]

    f = target(wd, threads, seed)
    backup = f + ".pre_repeat"
    existed = os.path.exists(f)
    if existed:
        shutil.copy2(f, backup)
        original = json.load(open(f, encoding="utf-8"))
        print(f"original: best={original['best']} grok={original['grok']} "
              f"secs={original['secs']}", flush=True)
    else:
        original = None
        print("original: (file did not exist)", flush=True)

    runs = []
    for i in range(n):
        t0 = time.time()
        p = subprocess.run(
            [PY, os.path.join("scripts", "22_numerical_fragility.py"),
             wd, threads, seed],
            cwd=ROOT, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        el = round(time.time() - t0, 1)
        if p.returncode != 0:
            print(f"  run {i+1}: FAILED rc={p.returncode}", flush=True)
            print((p.stderr or "")[-800:], flush=True)
            runs.append({"run": i + 1, "error": True})
            continue
        j = json.load(open(f, encoding="utf-8"))
        runs.append({"run": i + 1, "best": j["best"], "grok": j["grok"],
                     "grok_ep": j.get("grok_ep"), "secs": j["secs"],
                     "elapsed": el})
        print(f"  run {i+1}: best={j['best']} grok={j['grok']} "
              f"grok_ep={j.get('grok_ep')}", flush=True)

    ok = [r for r in runs if "error" not in r]
    summary = {"config": {"wd": wd, "threads": threads, "seed": seed},
               "original": original, "runs": runs}
    if ok:
        bests = [r["best"] for r in ok]
        groks = [r["grok"] for r in ok]
        summary["stats"] = {
            "n": len(ok),
            "best_values": bests,
            "best_min": min(bests), "best_max": max(bests),
            "best_range": round(max(bests) - min(bests), 3),
            "best_mean": round(sum(bests) / len(bests), 4),
            "grok_true": sum(1 for g in groks if g),
            "grok_false": sum(1 for g in groks if not g),
            "grok_flipped": len(set(groks)) > 1,
        }
        s = summary["stats"]
        print(f"\n  best values : {bests}")
        print(f"  range       : {s['best_range']}   (cross-machine scale = 0.164)")
        print(f"  grok        : {s['grok_true']} True / {s['grok_false']} False"
              f"   flipped={s['grok_flipped']}")

    # restore original state
    if existed and not keep:
        shutil.copy2(backup, f)
        os.remove(backup)
        print(f"\n  restored original best={original['best']}", flush=True)
    elif keep:
        print("\n  --keep: leaving the last run in place", flush=True)

    outp = os.path.join(r"D:\research",
                        f"_repeat_wd{wd}_t{threads}_s{seed}.json")
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"  wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
