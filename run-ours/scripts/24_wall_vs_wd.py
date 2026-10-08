"""
DECISIVE check: is the "(a*b+c*d) mod 10 never groks" single-agent WALL (which motivates
decomposition / Pillar B) robust to weight decay, or was it an artifact of wd=0.01?

Pillar A just showed (a+b)%10 groks 9/10 at wd=0.1 vs 2-3/10 at wd=0.01. So every prior
frontier/coverage result (all at wd=0.01) sat in the low-grok regime. If mul2add2 grokks
at wd=0.1 too, the wall collapses and the decomposition story needs restructuring. If it
does NOT, the wall is robust to the strongest grok-promoting regularizer -> Pillar B holds.

Fixed held-out 1000 (TEST_SEED=12345), disjoint train pool, mul2add2 monolith.
One job = one (wd, n_train, seed). Compare grok-rate vs the wd=0.01 baseline (0/10).

Usage: python 24_wall_vs_wd.py <wd> <n_train> <seed>
       -> results/24_wall_wd<wd>_n<n_train>_s<seed>.json
"""
import os, sys, json, random, time, torch
torch.set_num_threads(1)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

WD = float(sys.argv[1]); NTRAIN = int(sys.argv[2]); SEED = int(sys.argv[3])
M, DEV, GROK = 10, "cpu", 0.70
TEST_N, TEST_SEED, MAX_EPOCHS = 1000, 12345, 1200
COMP = lambda a, b, c, d: (a*b + c*d) % M
FMT  = lambda a, b, c, d: f"{a}*{b}+{c}*{d}="
LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", f"24_wall_wd{WD}_n{NTRAIN}_s{SEED}.json"))
tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
if tok.pad_token is None: tok.pad_token = tok.eos_token
BOS, EOS, PAD = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
def enc(s): return tok(s, add_special_tokens=False)["input_ids"]
def make(): return AutoModelForCausalLM.from_pretrained(LOCAL_DIR, dtype=torch.float32).to(DEV)

@torch.no_grad()
def acc(model, ex, bs=2048):
    model.eval()
    seqs = [[BOS] + enc(p) for p, _ in ex]; lens = [len(s) for s in seqs]; ml = max(lens)
    preds = []
    for i in range(0, len(seqs), bs):
        ch, cl = seqs[i:i+bs], lens[i:i+bs]
        ids = torch.full((len(ch), ml), PAD, dtype=torch.long); att = torch.zeros((len(ch), ml), dtype=torch.long)
        for j, s in enumerate(ch): ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
        logits = model(input_ids=ids, attention_mask=att).logits
        idx = torch.tensor([l-1 for l in cl])
        preds += logits[torch.arange(len(ch)), idx].argmax(-1).tolist()
    return sum(tok.decode([p]).strip() == a for p, (_, a) in zip(preds, ex)) / len(ex)

def main():
    t0 = time.time()
    Q4 = [(a, b, c, d) for a in range(10) for b in range(10) for c in range(10) for d in range(10)]
    rng = random.Random(TEST_SEED); pool = Q4[:]; rng.shuffle(pool)
    test_t, train_pool = pool[:TEST_N], pool[TEST_N:]
    rj = random.Random(SEED); rj.shuffle(train_pool); train_t = train_pool[:NTRAIN]
    torch.manual_seed(SEED)
    train_ex = [(FMT(*t), str(COMP(*t))) for t in train_t]
    test_ex  = [(FMT(*t), str(COMP(*t))) for t in test_t]
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
            opt.zero_grad(); model(input_ids=bi, attention_mask=bm, labels=bl).loss.backward(); opt.step()
        if ep % 100 == 0 or ep == MAX_EPOCHS:
            t = acc(model, test_ex)
            if t > best: best = t
            if grok_ep is None and t >= GROK: grok_ep = ep
            if t >= 0.985: break
    out = {"task": "mul2add2", "wd": WD, "n_train": NTRAIN, "coverage": round(NTRAIN/10000, 3),
           "seed": SEED, "best": round(best, 3), "grok": best >= GROK, "grok_ep": grok_ep,
           "secs": round(time.time()-t0, 1)}
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"mul2add2 wd{WD} cov{out['coverage']:.0%} s{SEED} best={best:.0%} grok={best>=GROK} ({out['secs']}s)", flush=True)

if __name__ == "__main__":
    main()
