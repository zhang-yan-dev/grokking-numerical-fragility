"""
M1 — grokking PROBABILITY vs compositional complexity (multi-seed, glass-box).

Motivation: script 13 showed grokking here is severely seed-sensitive -- the SAME task
(a+b)%10 grokked to 95% on one split and only 40% on another; (a+b+c)%10 gave 100% vs a
prior 6%. So no single-seed capability number is trustworthy. This script runs the full
frontier + decomposition for ONE seed (passed on argv) under a fixed budget and emits a
compact JSON summary. Launch many seeds in parallel (32 cores; torch pinned to 1 thread
each) and aggregate to get, per task: grok-rate = P(test>=0.70 within budget), mean best
test, and median time-to-grok -- turning the noise into the primary signal.

Usage: python 14_grok_probability.py <seed>   ->  results/14_seed<seed>.json
"""
import os, sys, json, random, time, torch
torch.set_num_threads(1)  # pin: many seeds run as parallel processes, 1 core each
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
GROK = 0.70  # test-acc threshold that counts as "generalized / grokked"
DEV = "cpu"  # CPU is faster AND cleaner than GPU at 11,856 params (see script 13)

LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", f"14_seed{SEED}.json"))
random.seed(SEED); torch.manual_seed(SEED)
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

def train(train_ex, test_ex, max_epochs=1500, bs=16, lr=3e-3, wd=0.01, eval_every=100):
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
    for ep in range(1, max_epochs + 1):
        model.train()
        for bi, bl, bm in dl:
            opt.zero_grad()
            model(input_ids=bi.to(DEV), attention_mask=bm.to(DEV), labels=bl.to(DEV)).loss.backward()
            opt.step()
        if ep % eval_every == 0 or ep == max_epochs:
            te = acc(model, test_ex)
            if te > best: best = te
            if grok_ep is None and te >= GROK: grok_ep = ep
            if te >= 0.985: break
    return model, best, grok_ep

def split(space, frac=0.2, cap_train=None, cap_test=None):
    sp = list(space); random.shuffle(sp); n = int(len(sp) * frac)
    test, tr = sp[:n], sp[n:]
    if cap_test: test = test[:cap_test]
    if cap_train: tr = tr[:cap_train]
    return tr, test
def mk(space, fmt, fn): return [(fmt(*t), str(fn(*t))) for t in space]

def main():
    t0 = time.time()
    P = [(a, b) for a in range(10) for b in range(10)]
    T3 = [(a, b, c) for a in range(10) for b in range(10) for c in range(10)]
    Q4 = [(a, b, c, d) for a in range(10) for b in range(10) for c in range(10) for d in range(10)]

    tasks = [
        ("add2",     P,  lambda a,b: f"{a}+{b}=",             lambda a,b: (a+b)%10,         None, None),
        ("mul2",     P,  lambda a,b: f"{a}*{b}=",             lambda a,b: (a*b)%10,         None, None),
        ("add3",     T3, lambda a,b,c: f"{a}+{b}+{c}=",       lambda a,b,c: (a+b+c)%10,     600, 150),
        ("muladd",   T3, lambda a,b,c: f"{a}*{b}+{c}=",       lambda a,b,c: (a*b+c)%10,     600, 150),
        ("add4",     Q4, lambda a,b,c,d: f"{a}+{b}+{c}+{d}=", lambda a,b,c,d: (a+b+c+d)%10, 800, 200),
        ("mul2add2", Q4, lambda a,b,c,d: f"{a}*{b}+{c}*{d}=", lambda a,b,c,d: (a*b+c*d)%10, 800, 200),
    ]
    out = {"seed": SEED, "grok_thresh": GROK, "tasks": {}}
    for name, space, fmt, fn, ctr, cte in tasks:
        tr, te = split(space, cap_train=ctr, cap_test=cte)
        _, best, gep = train(mk(tr, fmt, fn), mk(te, fmt, fn))
        out["tasks"][name] = {"best": round(best, 3), "grok": best >= GROK, "grok_ep": gep}
        print(f"seed{SEED} [{name}] best={best:.0%} grok={best>=GROK} @ep{gep}", flush=True)

    # specialists + pipelines (decomposition vs monolith)
    trm, tem = split(P); mul, mb, _ = train(mk(trm, lambda a,b:f"{a}*{b}=", lambda a,b:(a*b)%10),
                                            mk(tem, lambda a,b:f"{a}*{b}=", lambda a,b:(a*b)%10))
    tra, tea = split(P); add, ab, _ = train(mk(tra, lambda a,b:f"{a}+{b}=", lambda a,b:(a+b)%10),
                                            mk(tea, lambda a,b:f"{a}+{b}=", lambda a,b:(a+b)%10))
    _, te3 = split(T3, cap_test=300)
    s1 = [tok.decode([t]).strip() for t in next_tokens(mul, [f"{a}*{b}=" for a,b,c in te3])]
    valid = [(s,(a,b,c)) for s,(a,b,c) in zip(s1, te3) if s.isdigit()]
    p2 = next_tokens(add, [f"{s}+{c}=" for s,(a,b,c) in valid])
    pipe_muladd = sum(tok.decode([t]).strip()==str((a*b+c)%10) for t,(s,(a,b,c)) in zip(p2,valid))/len(te3)
    _, te4 = split(Q4, cap_test=300)
    pp = [tok.decode([t]).strip() for t in next_tokens(mul, [f"{a}*{b}=" for a,b,c,d in te4])]
    qq = [tok.decode([t]).strip() for t in next_tokens(mul, [f"{c}*{d}=" for a,b,c,d in te4])]
    keep = [(p,q,(a,b,c,d)) for p,q,(a,b,c,d) in zip(pp,qq,te4) if p.isdigit() and q.isdigit()]
    r3 = next_tokens(add, [f"{p}+{q}=" for p,q,_ in keep])
    pipe_m2a2 = sum(tok.decode([t]).strip()==str((a*b+c*d)%10) for t,(p,q,(a,b,c,d)) in zip(r3,keep))/len(te4)

    out["specialists"] = {"mul": round(mb,3), "add": round(ab,3)}
    out["pipelines"] = {
        "muladd_pipe": round(pipe_muladd,3), "muladd_mono": out["tasks"]["muladd"]["best"],
        "mul2add2_pipe": round(pipe_m2a2,3), "mul2add2_mono": out["tasks"]["mul2add2"]["best"],
    }
    out["secs"] = round(time.time()-t0, 1)
    with open(OUT, "w") as f: json.dump(out, f, indent=2)
    print(f"seed{SEED} DONE -> {OUT}  ({out['secs']}s)", flush=True)

if __name__ == "__main__":
    main()
