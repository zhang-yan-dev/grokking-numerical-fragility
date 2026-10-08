"""Test whether extra CPU load changes a seed's `best`, with `best` held fixed otherwise.

Findings so far:
  - wd=0.1 s3: 5/5 serial runs give best=0.80, grok_ep=1000, range=0.0
  - wd=0.1 s3: the very first run (4-way parallel, heavy concurrent load from
    other work on the same box) gave best=0.85
  - serial re-runs of s0/s1/s3/s5/s9 all match their stored `best`

So `best` looks deterministic given identical load, but may shift when the CPU
contention profile differs. This script runs the SAME seed twice — once idle,
once with N background CPU burners — and compares.

Usage: drive_load_test.py <wd> <threads> <seed> <burners>
"""
import json, os, shutil, subprocess, sys, time

ROOT = r"D:\research\run-ours"
PY = r"D:\venvs\grokking\Scripts\python.exe"
RESULTS = os.path.join(ROOT, "results")
BURN = r"D:\research\_burn.py"

env = dict(os.environ)
env["PYTHONUTF8"] = "1"
env["PYTHONIOENCODING"] = "utf-8"

BURN_SRC = """import time
t0 = time.time()
x = 0
while time.time() - t0 < 200:
    x = (x * 1103515245 + 12345) % (2 ** 31)
"""


def path(wd, threads, seed):
    return os.path.join(RESULTS, f"22_wd{wd}_t{threads}_s{seed}.json")


def one_run(wd, threads, seed, label):
    f = path(wd, threads, seed)
    backup = f + f".loadtest_{label}"
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
        shutil.copy2(backup, f); os.remove(backup)
        return {"label": label, "error": p.returncode}
    j = json.load(open(f, encoding="utf-8"))
    shutil.copy2(backup, f); os.remove(backup)
    return {"label": label, "best": j["best"], "grok": j["grok"],
            "grok_ep": j.get("grok_ep"), "secs": j["secs"], "wall": wall}


def main():
    wd, threads, seed, burners = (sys.argv[1], sys.argv[2],
                                  int(sys.argv[3]), int(sys.argv[4]))
    with open(BURN, "w", encoding="utf-8") as f:
        f.write(BURN_SRC)

    print(f"wd={wd} threads={threads} seed={seed} burners={burners}", flush=True)

    print("\n[idle run] nothing else on the box:", flush=True)
    idle = one_run(wd, threads, seed, "idle")
    print(f"  {idle}", flush=True)

    print(f"\n[loaded run] launching {burners} CPU burners:", flush=True)
    procs = [subprocess.Popen([PY, BURN]) for _ in range(burners)]
    time.sleep(3)
    loaded = one_run(wd, threads, seed, "loaded")
    print(f"  {loaded}", flush=True)
    for pr in procs:
        pr.terminate()
    for pr in procs:
        try:
            pr.wait(timeout=10)
        except Exception:
            pr.kill()

    print("\n=== comparison ===", flush=True)
    if "error" in idle or "error" in loaded:
        print("  one of the runs failed", flush=True)
    else:
        same = idle["best"] == loaded["best"]
        print(f"  idle   : best={idle['best']} grok_ep={idle['grok_ep']} wall={idle['wall']}s", flush=True)
        print(f"  loaded : best={loaded['best']} grok_ep={loaded['grok_ep']} wall={loaded['wall']}s", flush=True)
        print(f"  best identical      : {same}", flush=True)
        print(f"  best delta          : {abs(idle['best'] - loaded['best'])}", flush=True)
        print(f"  grok status identical: {idle['grok'] == loaded['grok']}", flush=True)

    with open(r"D:\research\_load_test.json", "w", encoding="utf-8") as fh:
        json.dump({"idle": idle, "loaded": loaded}, fh, indent=2, ensure_ascii=False)
    print("\nwrote D:\\research\\_load_test.json", flush=True)


if __name__ == "__main__":
    main()
