"""对比作者公开记录与本机重训结果。

用法:  python compare.py 0.1 4        # wd, threads
"""
import json, os, sys, glob

AUTH = r"D:\research\author-results-backup"
OURS = r"D:\research\run-ours\results"

wd, th = sys.argv[1], sys.argv[2]

def load(d, tag):
    out = {}
    for f in glob.glob(os.path.join(d, f"22_wd{wd}_t{th}_s*.json")):
        s = int(os.path.basename(f).rsplit("_s", 1)[1].split(".")[0])
        try:
            out[s] = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            print(f"  skip {os.path.basename(f)}: {e}")
    return out

a, b = load(AUTH, "author"), load(OURS, "ours")
seeds = sorted(set(a) & set(b))
if not seeds:
    print(f"没有可对比的数据 (wd={wd}, threads={th})")
    print(f"  作者: {len(a)} 个, 本机: {len(b)} 个")
    raise SystemExit

print(f"\n=== wd={wd}  threads={th}  对比 {len(seeds)} 个种子 ===\n")
print(f"{'seed':>5} {'作者best':>9} {'本机best':>9} {'作者grok':>9} {'本机grok':>9}  {'一致?'}")
print("-" * 60)
same_grok = same_best = 0
for s in seeds:
    ga, gb = a[s]["grok"], b[s]["grok"]
    ba, bb = a[s]["best"], b[s]["best"]
    if ga == gb: same_grok += 1
    if abs(ba - bb) < 1e-9: same_best += 1
    mark = "yes" if (ga == gb and abs(ba - bb) < 1e-9) else ("grok同" if ga == gb else "**翻转**")
    print(f"{s:>5} {ba:>9.2f} {bb:>9.2f} {str(ga):>9} {str(gb):>9}  {mark}")

n = len(seeds)
ra = sum(1 for s in seeds if a[s]["grok"]) / n
rb = sum(1 for s in seeds if b[s]["grok"]) / n
print("-" * 60)
print(f"\ngrok 率    作者 {ra:.0%} ({sum(1 for s in seeds if a[s]['grok'])}/{n})"
      f"   本机 {rb:.0%} ({sum(1 for s in seeds if b[s]['grok'])}/{n})")
print(f"grok 判定一致     : {same_grok}/{n}")
print(f"best 完全相同     : {same_best}/{n}")
print(f"best 平均绝对差   : {sum(abs(a[s]['best']-b[s]['best']) for s in seeds)/n:.3f}")
print()
print("结论看这两个数: grok 率是否接近 (论文的达标线), best 是否相等 (数值环境的敏感度)")
