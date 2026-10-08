"""
Pillar A — close the last loose end: is the CPU-vs-GPU grokking difference (the old
"CPU 95% vs GPU 45%" anecdote) REAL, or another seed confound like the thread-count claim?

Paired control: same seed -> identical data split (Python random), identical batch order
(torch generator), identical pretrained initial weights (Glimmer-1 loads deterministically
from disk). The ONLY thing that differs is WHERE the float math runs (CPU vs CUDA kernels /
accumulation order). Task (a+b) mod 10, 80/20, wd=0.1 (CPU baseline groks 9/10 -> high
dynamic range to detect a device-induced drop).

One job = one (device, seed). Compare grok-rate and per-seed flips, exactly like script 22's
thread experiment.

Usage: python 26_device_control.py <device:cpu|cuda> <seed>
       -> results/26_dev<device>_s<seed>.json
"""
import os, sys, json, random, time, torch
DEV = sys.argv[1]; SEED = int(sys.argv[2])
if DEV == "cpu":
    torch.set_num_threads(4)   # match script 22 wd=0.1 arm (threads=4)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

M, WD, MAX_EPOCHS, GROK = 10, 0.1, 1500, 0.70
LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", f"26_dev{DEV}_s{SEED}.json"))
tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
if tok.pad_token is None: tok.pad_token = tok.eos_token
BOS, EOS, PAD = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
def enc(s): return tok(s, add_special_tokens=False)["input_ids"]
def make(): return AutoModelForCausalLM.from_pretrained(LOCAL_DIR, dtype=torch.float32).to(DEV)

@torch.no_grad()
def acc(model, ex):
    model.eval()
    seqs = [[BOS] + enc(p) for p, _ in ex]; ml = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), ml), PAD, dtype=torch.long)
    att = torch.zeros((len(seqs), ml), dtype=torch.long)
    for j, s in enumerate(seqs): ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
    logits = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).logits
    idx = torch.tensor([len(s) - 1 for s in seqs], device=DEV)
    pred = logits[torch.arange(len(seqs), device=DEV), idx].argmax(-1).tolist()
    return sum(tok.decode([p]).strip() == a for p, (_, a) in zip(pred, ex)) / len(ex)

def main():
    t0 = time.time()
    random.seed(SEED); torch.manual_seed(SEED)
    P = [(a, b) for a in range(10) for b in range(10)]
    random.shuffle(P); n = int(len(P) * 0.2); te, tr = P[:n], P[n:]
    train_ex = [(f"{a}+{b}=", str((a+b)%M)) for a, b in tr]
    test_ex = [(f"{a}+{b}=", str((a+b)%M)) for a, b in te]
    model = make(); rows = []
    for p, a in train_ex:
        f = [BOS] + enc(p + a) + [EOS]; pl = len([BOS] + enc(p))
        lab = [-100]*len(f); lab[pl] = f[pl]; rows.append((f, lab))
    ml = max(len(f) for f, _ in rows)
    ids = torch.tensor([f + [PAD]*(ml-len(f)) for f, _ in rows])
    lab = torch.tensor([l + [-100]*(ml-len(l)) for _, l in rows])
    msk = torch.tensor([[1]*len(f) + [0]*(ml-len(f)) for f, _ in rows])
    g = torch.Generator().manual_seed(SEED)   # device-independent batch order
    dl = DataLoader(TensorDataset(ids, lab, msk), batch_size=16, shuffle=True, generator=g)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=WD)
    best, grok_ep = 0.0, None
    for ep in range(1, MAX_EPOCHS+1):
        model.train()
        for bi, bl, bm in dl:
            opt.zero_grad()
            model(input_ids=bi.to(DEV), attention_mask=bm.to(DEV), labels=bl.to(DEV)).loss.backward()
            opt.step()
        if ep % 100 == 0 or ep == MAX_EPOCHS:
            t = acc(model, test_ex)
            if t > best: best = t
            if grok_ep is None and t >= GROK: grok_ep = ep
            if t >= 0.985: break
    out = {"device": DEV, "wd": WD, "seed": SEED, "best": round(best,3),
           "grok": best >= GROK, "grok_ep": grok_ep, "secs": round(time.time()-t0,1)}
    json.dump(out, open(OUT,"w"), indent=2)
    print(f"dev={DEV} s{SEED} best={best:.0%} grok={best>=GROK} ({out['secs']}s)", flush=True)

if __name__ == "__main__":
    main()
