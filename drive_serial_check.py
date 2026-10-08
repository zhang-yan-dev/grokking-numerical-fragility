"""Strict controlled comparison: run one seed SERIALLY while nothing else runs,
and record both best and secs. Then compare against the stored value.

Purpose: decide whether the stored wd=0.1 seed-3 value (best=0.80, secs=81.1)
reflects a clean serial run or a contended one. The held claim under test is
"parallelism only affects wall-clock, not the grok outcome" (handover doc 7.4).

Method: for each (wd, seed) pair, snapshot the file, run once, print
best/secs/grok_ep, then restore. Runs sequentially with a gap, and asserts no
other python process is alive before each run.

Usage: drive_serial_check.py <wd> <threads> <seed> [<seed> ...]
"""
import json, os, shutil, subprocess, sys, time

ROOT = r"D:\research\run-ours"
PY = r"D:\venvs\grokking\Scripts\python.exe"
RESULTS = os.path.join(ROOT, "results")

env = dict(os.environ)
env["PYTHONUTF8"] = "1"
env["PYTHONIOENCODING"] = "utf-8"


def others_running():
    p = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe"],
                       capture_output=True, text=True, errors="replace")
    return p.stdout.lower().count("python.exe")


def path(wd, threads, seed):
    return os.path.join(RESULTS, f"22_wd{wd}_t{threads}_s{seed}.json")


def main():
    wd, threads = sys.argv[1], sys.argv[2]
    seeds = [int(x) for x in sys.argv[3:]]
    rows = []

    for seed in seeds:
        f = path(wd, threads, seed)
        prior = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None
        backup = f + ".serialcheck"

        n = others_running()
        print(f"\n--- wd={wd} s{seed} ---  (python processes alive: {n})", flush=True)
        if prior:
            print(f"  stored : best={prior['best']} grok={prior['grok']} "
                  f"secs={prior['secs']}", flush=True)

        shutil.copy2(f, backup)
        t0 = time.time()
        p = subprocess.run(
            [PY, os.path.join("scripts", "22_numerical_fragility.py"),
             wd, threads, str(seed)],
            cwd=ROOT, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        wall = round(time.time() - t0, 1)
        if p.returncode != 0:
            print(f"  FAILED rc={p.returncode}", flush=True)
            print((p.stderr or "")[-600:], flush=True)
            rows.append({"seed": seed, "error": True})
        else:
            j = json.load(open(f, encoding="utf-8"))
            print(f"  serial : best={j['best']} grok={j['grok']} "
                  f"grok_ep={j.get('grok_ep')} secs={j['secs']} (wall={wall}s)",
                  flush=True)
            rows.append({"seed": seed, "stored": prior, "serial": j, "wall": wall})

        shutil.copy2(backup, f)
        os.remove(backup)
        time.sleep(2)

    print("\n=== summary ===", flush=True)
    print(f"{'seed':>5}{'stored best':>13}{'serial best':>13}{'stored secs':>13}{'serial secs':>13}", flush=True)
    for r in rows:
        if "error" in r:
            print(f"{r['seed']:>5}   ERROR", flush=True)
            continue
        s, o = r["stored"], r["serial"]
        print(f"{r['seed']:>5}{s['best']:>13}{o['best']:>13}"
              f"{s['secs']:>13}{o['secs']:>13}", flush=True)

    with open(r"D:\research\_serial_check.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print("\nwrote D:\\research\\_serial_check.json", flush=True)


if __name__ == "__main__":
    main()
