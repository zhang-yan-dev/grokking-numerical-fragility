"""
go/no-go part 1 — is the mul2add2 monolith wall a DATA-COVERAGE effect (grokking) or a
hard capacity ceiling?

We train the monolith on (a*b+c*d)%10 with a varying number of training tuples drawn
from the 10,000-tuple space, and evaluate on a FIXED held-out test set (identical across
all runs). One job = one (seed, train_size). Launch the grid in parallel.

  - If monolith grok-rate RISES with coverage -> the wall is a sparse-data / grokking
    effect; decomposition helps by making each sub-task densely coverable (GREEN).
  - If it stays ~chance even at high coverage -> it's a capacity ceiling, and the
    "decomposition converts generalization into memorization" framing is weaker (RED).

Usage: python 16_coverage_sweep.py <seed> <train_size>  -> results/16_mono_s<seed>_n<size>.json
"""
import os, sys, json, random, time, torch
torch.set_num_threads(1)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NTRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
GROK, DEV = 0.70, "cpu"
TEST_N, TEST_SEED, MAX_EPOCHS = 1000, 12345, 1200

LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", f"16_mono_s{SEED}_n{NTRAIN}.json"))
tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
if tok.pad_token is None: tok.pad_token = tok.eos_token
BOS, EOS, PAD = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
def enc(s): return tok(s, add_special_tokens=False)["input_ids"]
def make(): return AutoModelForCausalLM.from_pretrained(LOCAL_DIR, dtype=torch.float32).to(DEV)

@torch.no_grad()
def next_tokens(model, prompts, bs=2048):
    model.eval()
    seqs = [[BOS] + enc(p) for p in prompts]; lens = [len(s) for s in seqs]
    ml = max(lens); out = []
    for i in range(0, len(seqs), bs):
        ch, cl = seqs[i:i+bs], lens[i:i+bs]
        ids = torch.full((len(ch), ml), PAD, dtype=torch.long)
        att = torch.zeros((len(ch), ml), dtype=torch.long)
        for j, s in enumerate(ch):
            ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
        logits = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).logits
        idx = torch.tensor([l - 1 for l in cl], device=DEV)
        out += logits[torch.arange(len(ch), device=DEV), idx].argmax(-1).tolist()
    return out

def acc(model, ex):
    preds = next_tokens(model, [p for p, _ in ex])
    return sum(tok.decode([t]).strip() == a for t, (_, a) in zip(preds, ex)) / len(ex)

def train(train_ex, test_ex, bs=16, lr=3e-3, wd=0.01, eval_every=100):
    model = make(); rows = []
    for p, a in train_ex:
        f = [BOS] + enc(p + a) + [EOS]; pl = len([BOS] + enc(p))
        lab = [-100] * len(f); lab[pl] = f[pl]; rows.append((f, lab))
    ml = max(len(f) for f, _ in rows)
    ids = torch.tensor([f + [PAD] * (ml - len(f)) for f, _ in rows])
    lab = torch.tensor([l + [-100] * (ml - len(l)) for _, l in rows])
    msk = torch.tensor([[1] * len(f) + [0] * (ml - len(f)) for f, _ in rows])
    dl = DataLoader(TensorDataset(ids, lab, msk), batch_size=bs, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, grok_ep = 0.0, None
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for bi, bl, bm in dl:
            opt.zero_grad()
            model(input_ids=bi.to(DEV), attention_mask=bm.to(DEV), labels=bl.to(DEV)).loss.backward()
            opt.step()
        if ep % eval_every == 0 or ep == MAX_EPOCHS:
            te = acc(model, test_ex)
            if te > best: best = te
            if grok_ep is None and te >= GROK: grok_ep = ep
            if te >= 0.985: break
    return best, grok_ep

def fmt(a, b, c, d): return f"{a}*{b}+{c}*{d}="
def fn(a, b, c, d): return (a * b + c * d) % 10

def main():
    t0 = time.time()
    Q4 = [(a, b, c, d) for a in range(10) for b in range(10) for c in range(10) for d in range(10)]
    # fixed held-out test (identical across all seeds/sizes), train pool = the rest
    rng = random.Random(TEST_SEED); pool = Q4[:]; rng.shuffle(pool)
    test_tuples, train_pool = pool[:TEST_N], pool[TEST_N:]
    # this job's train sample
    rj = random.Random(SEED); rj.shuffle(train_pool); train_tuples = train_pool[:NTRAIN]
    train_ex = [(fmt(*t), str(fn(*t))) for t in train_tuples]
    test_ex = [(fmt(*t), str(fn(*t))) for t in test_tuples]
    torch.manual_seed(SEED)
    best, gep = train(train_ex, test_ex)
    out = {"seed": SEED, "n_train": NTRAIN, "coverage": round(NTRAIN/10000, 3),
           "best": round(best, 3), "grok": best >= GROK, "grok_ep": gep,
           "secs": round(time.time()-t0, 1)}
    with open(OUT, "w") as f: json.dump(out, f, indent=2)
    print(f"s{SEED} n{NTRAIN} cov{out['coverage']:.0%} best={best:.0%} grok={best>=GROK} ({out['secs']}s)", flush=True)

if __name__ == "__main__":
    main()
