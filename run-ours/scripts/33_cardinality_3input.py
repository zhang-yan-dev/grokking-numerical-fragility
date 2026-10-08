"""
Reviewer-strengthening (cardinality regularity): does the grok coverage-threshold track output
CARDINALITY (modulus M) rather than DOMAIN SIZE?

The 4-input composites in scripts 16/18 all live on a domain of 10^4 tuples; there, the
threshold coverage rises with M (M7 ~10%, M9 ~15%, M10 ~30%). But domain and cardinality are
confounded only by structure there, not by domain size. Here we move to a 3-input composite

    (a*b + c) mod M ,   a,b,c in 0..9   ->   domain = 10^3 (ten-fold smaller)

and sweep M in {5..10} at fixed coverage. If grok-rate still falls with M on this SMALLER
domain (and the threshold coverage matches the 10^4 numbers at equal M), the governing variable
is cardinality M, not the absolute domain size. wd=0.01 to match the 10^4 sweeps exactly.

One job = one (M, n_train, seed). Launch the grid in parallel.

Usage: python 33_cardinality_3input.py <M> <n_train> <seed>
       -> results/33_card_m<M>_n<n_train>_s<seed>.json
"""
import os, sys, json, random, time, torch
torch.set_num_threads(1)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

M = int(sys.argv[1]); NTRAIN = int(sys.argv[2]); SEED = int(sys.argv[3])
GROK, DEV, WD = 0.70, "cpu", 0.01
TEST_N, TEST_SEED, MAX_EPOCHS = 300, 12345, 1200

LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", f"33_card_m{M}_n{NTRAIN}_s{SEED}.json"))
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

def fmt(a, b, c): return f"{a}*{b}+{c}="
def fn(a, b, c): return (a * b + c) % M

def train(train_ex, test_ex, bs=16, lr=3e-3):
    model = make(); rows = []
    for p, a in train_ex:
        f = [BOS] + enc(p + a) + [EOS]; pl = len([BOS] + enc(p))
        lab = [-100] * len(f); lab[pl] = f[pl]; rows.append((f, lab))
    ml = max(len(f) for f, _ in rows)
    ids = torch.tensor([f + [PAD] * (ml - len(f)) for f, _ in rows])
    lab = torch.tensor([l + [-100] * (ml - len(l)) for _, l in rows])
    msk = torch.tensor([[1] * len(f) + [0] * (ml - len(f)) for f, _ in rows])
    dl = DataLoader(TensorDataset(ids, lab, msk), batch_size=bs, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    best, grok_ep = 0.0, None
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for bi, bl, bm in dl:
            opt.zero_grad()
            model(input_ids=bi.to(DEV), attention_mask=bm.to(DEV), labels=bl.to(DEV)).loss.backward()
            opt.step()
        if ep % 100 == 0 or ep == MAX_EPOCHS:
            te = acc(model, test_ex)
            if te > best: best = te
            if grok_ep is None and te >= GROK: grok_ep = ep
            if te >= 0.985: break
    return best, grok_ep

def main():
    t0 = time.time()
    Q3 = [(a, b, c) for a in range(10) for b in range(10) for c in range(10)]  # 1000
    rng = random.Random(TEST_SEED); pool = Q3[:]; rng.shuffle(pool)
    test_t, train_pool = pool[:TEST_N], pool[TEST_N:]
    rj = random.Random(SEED); rj.shuffle(train_pool); train_t = train_pool[:NTRAIN]
    train_ex = [(fmt(*t), str(fn(*t))) for t in train_t]
    test_ex = [(fmt(*t), str(fn(*t))) for t in test_t]
    torch.manual_seed(SEED)
    best, gep = train(train_ex, test_ex)
    out = {"task": "3input_mulpadd", "mod": M, "domain": 1000, "seed": SEED,
           "n_train": NTRAIN, "coverage": round(NTRAIN / 1000, 3),
           "best": round(best, 3), "grok": best >= GROK, "grok_ep": gep,
           "secs": round(time.time() - t0, 1)}
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"M{M} n{NTRAIN} cov{out['coverage']:.0%} s{SEED} best={best:.0%} grok={best>=GROK} ({out['secs']}s)", flush=True)

if __name__ == "__main__":
    main()
