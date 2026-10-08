"""Parallel driver for script 22, avoiding PowerShell Start-Process env issues.

Runs the remaining wd grid: wd=0.0 (all seeds except 7, already OURS) and wd=1.0 (all 10).
Writes a JSON report of what happened to D:\\research\\_wdgrid_report.json
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = r"D:\research\run-ours"
PY = r"D:\venvs\grokking\Scripts\python.exe"
LOGDIR = os.path.join(ROOT, "logs")
os.makedirs(LOGDIR, exist_ok=True)

jobs = []
for s in range(10):
    if s != 7:
        jobs.append(("0.0", s))
for s in range(10):
    jobs.append(("1.0", s))

# skip wd=1.0 s0 (already done in dry run)
jobs = [j for j in jobs if j != ("1.0", 0)]

env = dict(os.environ)
env["PYTHONUTF8"] = "1"
env["PYTHONIOENCODING"] = "utf-8"

def run(job):
    wd, s = job
    tag = wd.replace(".", "p")
    t0 = time.time()
    p = subprocess.run(
        [PY, os.path.join("scripts", "22_numerical_fragility.py"), str(wd), "4", str(s)],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    el = time.time() - t0
    with open(os.path.join(LOGDIR, f"wd{tag}_s{s}.log"), "w", encoding="utf-8") as f:
        f.write((p.stdout or "") + "\n--- STDERR ---\n" + (p.stderr or ""))
    return {"wd": wd, "seed": s, "rc": p.returncode, "elapsed": round(el, 1),
            "stdout_tail": (p.stdout or "").strip().splitlines()[-1] if p.stdout else ""}

results = []
t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    for r in ex.map(run, jobs):
        results.append(r)
        ok = "OK " if r["rc"] == 0 else "ERR"
        print(f"{ok} wd={r['wd']} s{r['seed']} {r['elapsed']}s  {r['stdout_tail']}", flush=True)

report = {"total": len(jobs), "elapsed": round(time.time() - t0, 1),
          "failed": [r for r in results if r["rc"] != 0], "results": results}
with open(r"D:\research\_wdgrid_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\nDONE {len(jobs) - len(report['failed'])}/{len(jobs)} in {report['elapsed']}s", flush=True)
