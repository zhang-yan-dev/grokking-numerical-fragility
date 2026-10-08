"""
Step 3 (pillar A) — grokking as a knife-edge NUMERICAL phenomenon. We measure grok-rate
of the canonical task (a+b) mod 10 (80/20 split) while varying ONE knob at a time:

  * num_threads in {1,2,4,8}: pure floating-point reduction-ORDER change. The grok
    OUTCOME for a given (threads, seed) is deterministic and independent of process
    contention (set_num_threads fixes the reduction tree), so we can oversubscribe.
    Headline claim: changing CPU thread count alone flips grok-rate.
  * weight_decay in {0, 0.01, 0.1, 1.0}: sanity control — should reproduce the known
    "wd promotes grokking" (Omnigrok) trend, validating our setup.

One job = one (wd, threads, seed). Aggregate to grok-rate per knob value.

Usage: python 22_numerical_fragility.py <wd> <threads> <seed>
       -> results/22_wd<wd>_t<threads>_s<seed>.json
"""
import os, sys
WD = float(sys.argv[1]); THREADS = int(sys.argv[2]); SEED = int(sys.argv[3])
# pin BLAS/OMP threads BEFORE importing torch so the reduction order is truly THREADS-wide
os.environ["OMP_NUM_THREADS"] = str(THREADS)
os.environ["MKL_NUM_THREADS"] = str(THREADS)
import json, random, time, torch
torch.set_num_threads(THREADS)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

M, DEV, MAX_EPOCHS, GROK = 10, "cpu", 1500, 0.70
LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", f"22_wd{WD}_t{THREADS}_s{SEED}.json"))
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
    logits = model(input_ids=ids, attention_mask=att).logits
    idx = torch.tensor([len(s) - 1 for s in seqs])
    pred = logits[torch.arange(len(seqs)), idx].argmax(-1).tolist()
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
    dl = DataLoader(TensorDataset(ids, lab, msk), batch_size=16, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=WD)
    best, grok_ep = 0.0, None
    for ep in range(1, MAX_EPOCHS+1):
        model.train()
        for bi, bl, bm in dl:
            opt.zero_grad()
            model(input_ids=bi, attention_mask=bm, labels=bl).loss.backward()
            opt.step()
        if ep % 100 == 0 or ep == MAX_EPOCHS:
            t = acc(model, test_ex)
            if t > best: best = t
            if grok_ep is None and t >= GROK: grok_ep = ep
            if t >= 0.985: break
    out = {"wd": WD, "threads": THREADS, "seed": SEED, "best": round(best,3),
           "grok": best >= GROK, "grok_ep": grok_ep, "secs": round(time.time()-t0,1)}
    json.dump(out, open(OUT,"w"), indent=2)
    print(f"wd{WD} t{THREADS} s{SEED} best={best:.0%} grok={best>=GROK} ({out['secs']}s)", flush=True)

if __name__ == "__main__":
    main()
