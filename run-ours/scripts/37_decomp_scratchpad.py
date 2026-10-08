"""
DISENTANGLING the decomposition confound (reviewer control).

The pipeline beats the monolith at a matched 160-example budget (script 25), but its examples
carry BOTH (i) denser sub-domain coverage and (ii) more decomposed SUPERVISION (the specialists
see the intermediate products and sum as direct targets). To separate them we add a third arm:

  DIRECT monolith    : 160 composite 4-tuples, target = final digit only.            (baseline)
  SCRATCHPAD monolith: the SAME 160 composite 4-tuples, the SAME single model, but the target
                       is the 3-token chain P,Q,R = (a*b)%M,(c*d)%M,(P+Q)%M -- intermediate
                       supervision WITHOUT a pipeline and at the SAME sparse composite coverage.
  PIPELINE           : mul-spec (80 of 100 products) + add-spec (80 of 100 sums), composed.

DIRECT vs SCRATCHPAD isolates intermediate supervision (architecture + coverage held fixed).
SCRATCHPAD vs PIPELINE isolates the remaining pipeline/coverage effect. All three share the
frozen 1,000-tuple held-out test set, so the comparison is paired per seed.

Usage: python 37_decomp_scratchpad.py <wd> <seed>  ->  results/37_scratch_wd<wd>_s<seed>.json
"""
import os, sys, json, random, time, torch
torch.set_num_threads(1)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

WD = float(sys.argv[1]); SEED = int(sys.argv[2])
M, DEV, GROK = 10, "cpu", 0.70
TEST_N, TEST_SEED, MONO_N = 1000, 12345, 160
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", 1200))  # override for smoke tests
COMP = lambda a, b, c, d: (a * b + c * d) % M
LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", f"37_scratch_wd{WD}_s{SEED}.json"))
tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
if tok.pad_token is None: tok.pad_token = tok.eos_token
BOS, EOS, PAD = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
def enc(s): return tok(s, add_special_tokens=False)["input_ids"]
def make(): return AutoModelForCausalLM.from_pretrained(LOCAL_DIR, dtype=torch.float32).to(DEV)
# single-token id per digit (digits 0-9 are single tokens); assembled directly to avoid
# any BPE merge of adjacent scratchpad digits.
DIG = {}
for d in range(10):
    e = enc(str(d)); assert len(e) == 1, f"digit {d} is not a single token: {e}"
    DIG[d] = e[0]

@torch.no_grad()
def next_tokens(model, prompts, bs=2048):
    model.eval()
    seqs = [[BOS] + enc(p) for p in prompts]; lens = [len(s) for s in seqs]; ml = max(lens); out = []
    for i in range(0, len(seqs), bs):
        ch, cl = seqs[i:i+bs], lens[i:i+bs]
        ids = torch.full((len(ch), ml), PAD, dtype=torch.long); att = torch.zeros((len(ch), ml), dtype=torch.long)
        for j, s in enumerate(ch): ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
        logits = model(input_ids=ids, attention_mask=att).logits
        idx = torch.tensor([l - 1 for l in cl])
        out += logits[torch.arange(len(ch)), idx].argmax(-1).tolist()
    return out

@torch.no_grad()
def gen_k(model, prompts, k=3, bs=1024):
    """greedy autoregressive generation of k tokens; returns [[t1..tk], ...]."""
    model.eval(); res = []
    base = [[BOS] + enc(p) for p in prompts]
    for i in range(0, len(base), bs):
        chunk = [s[:] for s in base[i:i+bs]]; gen = [[] for _ in chunk]
        for _ in range(k):
            lens = [len(s) for s in chunk]; ml = max(lens)
            ids = torch.full((len(chunk), ml), PAD, dtype=torch.long); att = torch.zeros((len(chunk), ml), dtype=torch.long)
            for j, s in enumerate(chunk): ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
            logits = model(input_ids=ids, attention_mask=att).logits
            idx = torch.tensor([l - 1 for l in lens])
            nxt = logits[torch.arange(len(chunk)), idx].argmax(-1).tolist()
            for j, t in enumerate(nxt): chunk[j].append(t); gen[j].append(t)
        res += gen
    return res

def acc_ex(model, ex):
    if not ex: return None
    preds = next_tokens(model, [p for p, _ in ex])
    return sum(tok.decode([t]).strip() == a for t, (_, a) in zip(preds, ex)) / len(ex)

def train_single(train_ex, test_ex=None, eval_grok=False):
    """one supervised answer token (direct monolith and the two specialists)."""
    model = make(); rows = []
    for p, a in train_ex:
        f = [BOS] + enc(p + a) + [EOS]; pl = len([BOS] + enc(p))
        lab = [-100] * len(f); lab[pl] = f[pl]; rows.append((f, lab))
    ml = max(len(f) for f, _ in rows)
    ids = torch.tensor([f + [PAD] * (ml - len(f)) for f, _ in rows])
    lab = torch.tensor([l + [-100] * (ml - len(l)) for _, l in rows])
    msk = torch.tensor([[1] * len(f) + [0] * (ml - len(f)) for f, _ in rows])
    dl = DataLoader(TensorDataset(ids, lab, msk), batch_size=16, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=WD)
    best, gep = 0.0, None
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for bi, bl, bm in dl:
            opt.zero_grad(); model(input_ids=bi, attention_mask=bm, labels=bl).loss.backward(); opt.step()
        if eval_grok and (ep % 100 == 0 or ep == MAX_EPOCHS):
            t = acc_ex(model, test_ex)
            if t > best: best = t
            if gep is None and t >= GROK: gep = ep
            if t >= 0.985: break
    return model, best, gep

def train_scratch(tuples, test_tuples, eval_grok=True):
    """three supervised answer tokens P,Q,R (intermediate-supervised monolith)."""
    model = make(); rows = []
    for a, b, c, d in tuples:
        P, Q = (a * b) % M, (c * d) % M; R = (P + Q) % M
        pre = [BOS] + enc(f"{a}*{b}+{c}*{d}=")
        f = pre + [DIG[P], DIG[Q], DIG[R], EOS]
        lab = [-100] * len(f)
        lab[len(pre)] = DIG[P]; lab[len(pre) + 1] = DIG[Q]; lab[len(pre) + 2] = DIG[R]
        rows.append((f, lab))
    ml = max(len(f) for f, _ in rows)
    ids = torch.tensor([f + [PAD] * (ml - len(f)) for f, _ in rows])
    lab = torch.tensor([l + [-100] * (ml - len(l)) for _, l in rows])
    msk = torch.tensor([[1] * len(f) + [0] * (ml - len(f)) for f, _ in rows])
    dl = DataLoader(TensorDataset(ids, lab, msk), batch_size=16, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=WD)
    prompts = [f"{a}*{b}+{c}*{d}=" for a, b, c, d in test_tuples]
    finals = [str(COMP(*t)) for t in test_tuples]
    best, gep = 0.0, None
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for bi, bl, bm in dl:
            opt.zero_grad(); model(input_ids=bi, attention_mask=bm, labels=bl).loss.backward(); opt.step()
        if eval_grok and (ep % 100 == 0 or ep == MAX_EPOCHS):
            gens = gen_k(model, prompts, k=3)
            fin = sum(tok.decode([g[2]]).strip() == fv for g, fv in zip(gens, finals)) / len(finals)
            if fin > best: best = fin
            if gep is None and fin >= GROK: gep = ep
            if fin >= 0.985: break
    gens = gen_k(model, prompts, k=3)
    pq = sum(tok.decode([g[0]]).strip() == str((a * b) % M) and tok.decode([g[1]]).strip() == str((c * d) % M)
             for g, (a, b, c, d) in zip(gens, test_tuples)) / len(test_tuples)
    return model, best, gep, round(pq, 3)

def main():
    t0 = time.time()
    Q4 = [(a, b, c, d) for a in range(10) for b in range(10) for c in range(10) for d in range(10)]
    rng = random.Random(TEST_SEED); pool = Q4[:]; rng.shuffle(pool)
    test_t, train_pool = pool[:TEST_N], pool[TEST_N:]
    rj = random.Random(SEED); rj.shuffle(train_pool); mono_train = train_pool[:MONO_N]

    # arm 1: DIRECT monolith (reproduces script 25 monolith)
    torch.manual_seed(SEED)
    _, direct_best, direct_gep = train_single(
        [(f"{a}*{b}+{c}*{d}=", str(COMP(a, b, c, d))) for a, b, c, d in mono_train],
        [(f"{a}*{b}+{c}*{d}=", str(COMP(a, b, c, d))) for a, b, c, d in test_t], eval_grok=True)

    # arm 2: SCRATCHPAD monolith (same 160 tuples + intermediate supervision)
    torch.manual_seed(SEED)
    _, scr_best, scr_gep, scr_inter = train_scratch(mono_train, test_t, eval_grok=True)

    # arm 3: PIPELINE (mul-spec + add-spec) -- identical recipe to script 25
    random.seed(SEED); torch.manual_seed(SEED)
    pairs = [(x, y) for x in range(10) for y in range(10)]
    pm = pairs[:]; random.shuffle(pm); nm = int(len(pm) * 0.2); mul_tr = pm[nm:]
    mul_model, _, _ = train_single([(f"{x}*{y}=", str((x * y) % M)) for x, y in mul_tr])
    pa = pairs[:]; random.shuffle(pa); na = int(len(pa) * 0.2); add_tr = pa[na:]
    add_model, _, _ = train_single([(f"{x}+{y}=", str((x + y) % M)) for x, y in add_tr])
    run = lambda model, prompts: [tok.decode([t]).strip() for t in next_tokens(model, prompts)]
    p = run(mul_model, [f"{a}*{b}=" for a, b, c, d in test_t])
    q = run(mul_model, [f"{c}*{d}=" for a, b, c, d in test_t])
    keep = [(pi, qi, t) for pi, qi, t in zip(p, q, test_t) if pi.isdigit() and qi.isdigit()]
    r = run(add_model, [f"{pi}+{qi}=" for pi, qi, _ in keep])
    pipe_ok = sum(tkn == str(COMP(*t)) for tkn, (pi, qi, t) in zip(r, keep))
    pipe_acc = pipe_ok / len(test_t)

    out = {"task": "mul2add2", "wd": WD, "seed": SEED, "mono_n": MONO_N,
           "direct":     {"best": round(direct_best, 3), "grok": direct_best >= GROK, "grok_ep": direct_gep},
           "scratchpad": {"best": round(scr_best, 3), "grok": scr_best >= GROK, "grok_ep": scr_gep,
                          "inter_PQ_acc": scr_inter},
           "pipeline":   {"best": round(pipe_acc, 3), "grok": pipe_acc >= GROK},
           "secs": round(time.time() - t0, 1)}
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"scratch wd{WD} s{SEED}: direct={direct_best:.0%}(g{direct_best>=GROK}) "
          f"scratch={scr_best:.0%}(g{scr_best>=GROK},PQ{scr_inter:.0%}) "
          f"pipe={pipe_acc:.0%}(g{pipe_acc>=GROK}) ({out['secs']}s)", flush=True)

if __name__ == "__main__":
    main()
