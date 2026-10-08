"""
Does the DECOMPOSITION advantage survive in the grok-friendly regime (wd=0.1)?

The "wall" is dead (script 24: mul2add2 monolith groks 2-4/10 at wd=0.1, not 0/10). So
decomposition can no longer be sold as "enabling the impossible." It survives as a paper
section ONLY if, at MATCHED weight decay and MATCHED data budget, the pipeline still beats
the monolith. We run both arms in one job per seed for a clean head-to-head.

  MONOLITH: train mul2add2 (a*b+c*d)%10 on N composite 4-tuples, test on held-out 1000.
  PIPELINE: train mul-spec on (a*b)%10 (80/100) + add-spec on (a+b)%10 (80/100) = 160
            examples; compose mul,mul->add at inference on the same held-out 1000.
            (matched budget: monolith N=160 too; we also know monolith@800 from script 24.)

Report monolith best/grok and pipeline overall + SEEN/HELDOUT split (memorization signature).

Usage: python 25_decomp_vs_wd.py <wd> <seed>
       -> results/25_decomp_wd<wd>_s<seed>.json
"""
import os, sys, json, random, time, torch
torch.set_num_threads(1)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

WD = float(sys.argv[1]); SEED = int(sys.argv[2])
M, DEV, GROK = 10, "cpu", 0.70
TEST_N, TEST_SEED, MAX_EPOCHS, MONO_N = 1000, 12345, 1200, 160
COMP = lambda a, b, c, d: (a*b + c*d) % M
LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", f"25_decomp_wd{WD}_s{SEED}.json"))
tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
if tok.pad_token is None: tok.pad_token = tok.eos_token
BOS, EOS, PAD = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
def enc(s): return tok(s, add_special_tokens=False)["input_ids"]
def make(): return AutoModelForCausalLM.from_pretrained(LOCAL_DIR, dtype=torch.float32).to(DEV)

@torch.no_grad()
def next_tokens(model, prompts, bs=2048):
    model.eval()
    seqs = [[BOS] + enc(p) for p in prompts]; lens = [len(s) for s in seqs]; ml = max(lens); out = []
    for i in range(0, len(seqs), bs):
        ch, cl = seqs[i:i+bs], lens[i:i+bs]
        ids = torch.full((len(ch), ml), PAD, dtype=torch.long); att = torch.zeros((len(ch), ml), dtype=torch.long)
        for j, s in enumerate(ch): ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
        logits = model(input_ids=ids, attention_mask=att).logits
        idx = torch.tensor([l-1 for l in cl])
        out += logits[torch.arange(len(ch)), idx].argmax(-1).tolist()
    return out

def acc_ex(model, ex):
    if not ex: return None
    preds = next_tokens(model, [p for p, _ in ex])
    return sum(tok.decode([t]).strip() == a for t, (_, a) in zip(preds, ex)) / len(ex)

def train(train_ex, test_ex=None, eval_grok=False):
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
        if eval_grok and (ep % 100 == 0 or ep == MAX_EPOCHS):
            t = acc_ex(model, test_ex)
            if t > best: best = t
            if grok_ep is None and t >= GROK: grok_ep = ep
            if t >= 0.985: break
    return model, best, grok_ep

def main():
    t0 = time.time()
    Q4 = [(a, b, c, d) for a in range(10) for b in range(10) for c in range(10) for d in range(10)]
    rng = random.Random(TEST_SEED); pool = Q4[:]; rng.shuffle(pool)
    test_t, train_pool = pool[:TEST_N], pool[TEST_N:]

    # ---- MONOLITH arm (matched budget MONO_N) ----
    rj = random.Random(SEED); rj.shuffle(train_pool); mono_train = train_pool[:MONO_N]
    torch.manual_seed(SEED)
    _, mono_best, mono_gep = train([(f"{a}*{b}+{c}*{d}=", str(COMP(a,b,c,d))) for a,b,c,d in mono_train],
                                   [(f"{a}*{b}+{c}*{d}=", str(COMP(a,b,c,d))) for a,b,c,d in test_t], eval_grok=True)

    # ---- PIPELINE arm: mul-spec + add-spec ----
    random.seed(SEED); torch.manual_seed(SEED)
    pairs = [(x, y) for x in range(10) for y in range(10)]
    pm = pairs[:]; random.shuffle(pm); nm = int(len(pm)*0.2); mul_tr = pm[nm:]
    mul_model, _, _ = train([(f"{x}*{y}=", str((x*y)%M)) for x, y in mul_tr])
    pa = pairs[:]; random.shuffle(pa); na = int(len(pa)*0.2); add_tr = pa[na:]; add_train_set = set(add_tr)
    add_model, _, _ = train([(f"{x}+{y}=", str((x+y)%M)) for x, y in add_tr])
    mul_tr_acc = acc_ex(mul_model, [(f"{x}*{y}=", str((x*y)%M)) for x, y in mul_tr])
    add_tr_acc = acc_ex(add_model, [(f"{x}+{y}=", str((x+y)%M)) for x, y in add_tr])

    def run(model, prompts): return [tok.decode([t]).strip() for t in next_tokens(model, prompts)]
    p = run(mul_model, [f"{a}*{b}=" for a,b,c,d in test_t])
    q = run(mul_model, [f"{c}*{d}=" for a,b,c,d in test_t])
    keep = [(pi, qi, t) for pi, qi, t in zip(p, q, test_t) if pi.isdigit() and qi.isdigit()]
    r = run(add_model, [f"{pi}+{qi}=" for pi, qi, _ in keep])
    seen_ok=seen_n=held_ok=held_n=tot=0
    for tkn, (pi, qi, t) in zip(r, keep):
        ok = tkn == str(COMP(*t)); tot += ok
        if (int(pi), int(qi)) in add_train_set: seen_n += 1; seen_ok += ok
        else: held_n += 1; held_ok += ok
    pipe_acc = tot/len(test_t)

    out = {"task": "mul2add2", "wd": WD, "seed": SEED,
           "monolith": {"n_train": MONO_N, "best": round(mono_best,3), "grok": mono_best>=GROK, "grok_ep": mono_gep},
           "pipeline": {"overall_acc": round(pipe_acc,3), "grok": pipe_acc>=GROK,
                        "mul_train_acc": round(mul_tr_acc,3), "add_train_acc": round(add_tr_acc,3),
                        "acc_adder_pair_SEEN": round(seen_ok/seen_n,3) if seen_n else None,
                        "acc_adder_pair_HELDOUT": round(held_ok/held_n,3) if held_n else None,
                        "frac_seen": round(seen_n/(seen_n+held_n),3) if (seen_n+held_n) else None},
           "pipe_minus_mono": round(pipe_acc - mono_best, 3), "secs": round(time.time()-t0,1)}
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"mul2add2 wd{WD} s{SEED}: mono={mono_best:.0%}(grok={mono_best>=GROK}) "
          f"pipe={pipe_acc:.0%}(grok={pipe_acc>=GROK}) delta={out['pipe_minus_mono']:+.0%} ({out['secs']}s)", flush=True)

if __name__ == "__main__":
    main()
