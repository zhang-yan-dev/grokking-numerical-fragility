"""
M0/M1 — single-agent capability frontier vs multi-agent pipelines, on GPU.

Speed fixes vs script 11:
  * GPU (RTX 3090 Ti).
  * Evaluation = ONE batched forward over all prompts (teacher-forced argmax at the
    answer position == greedy 1-token gen for single-token answers). No generate loop.

We sweep increasing compositional depth and ask where a single ~11,856-param agent
caps, and whether a pipeline of simple specialists stays accurate past that point.
"""
import os, random, time, torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))
random.seed(0); torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
if tok.pad_token is None: tok.pad_token = tok.eos_token
BOS, EOS, PAD = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
def enc(s): return tok(s, add_special_tokens=False)["input_ids"]
def make(): return AutoModelForCausalLM.from_pretrained(LOCAL_DIR, dtype=torch.float32).to(DEV)

@torch.no_grad()
def next_tokens(model, prompts, bs=1024):
    """Greedy next token id after each prompt, batched."""
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

def train(train_ex, test_ex, name, max_epochs=1500, bs=32, lr=3e-3, wd=0.01,
          eval_every=50, target=0.985, patience=5, log=True):
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
    best, bep, since, t0 = 0.0, 0, 0, time.time()
    for ep in range(1, max_epochs + 1):
        model.train()
        for bi, bl, bm in dl:
            opt.zero_grad()
            model(input_ids=bi.to(DEV), attention_mask=bm.to(DEV), labels=bl.to(DEV)).loss.backward()
            opt.step()
        if ep % eval_every == 0 or ep == max_epochs:
            tr, te = acc(model, train_ex), acc(model, test_ex)
            if te > best: best, bep, since = te, ep, 0
            else: since += 1
            if log and (ep % (eval_every*4) == 0 or te >= target):
                print(f"  [{name}] ep{ep:4d} train={tr:.0%} test={te:.0%}")
            if te >= target or (tr >= 0.999 and since >= patience): break
    print(f"  [{name}] -> best test {best:.0%} @ ep{bep}  ({time.time()-t0:.0f}s)")
    return model, best

def split(space, frac=0.2):
    sp = list(space); random.shuffle(sp); n = int(len(sp) * frac)
    return sp[n:], sp[:n]
def mk(space, fmt, fn): return [(fmt(*t), str(fn(*t))) for t in space]

def main():
    print(f"device={DEV}  {torch.cuda.get_device_name(0) if DEV=='cuda' else ''}")
    P = [(a, b) for a in range(10) for b in range(10)]
    T3 = [(a, b, c) for a in range(10) for b in range(10) for c in range(10)]
    Q4 = [(a, b, c, d) for a in range(10) for b in range(10) for c in range(10) for d in range(10)]
    res = {}

    print("\n== Single-agent capability frontier ==")
    tasks = [
        ("add2",     P,  lambda a,b: f"{a}+{b}=",            lambda a,b: (a+b)%10),
        ("mul2",     P,  lambda a,b: f"{a}*{b}=",            lambda a,b: (a*b)%10),
        ("add3",     T3, lambda a,b,c: f"{a}+{b}+{c}=",      lambda a,b,c: (a+b+c)%10),
        ("muladd",   T3, lambda a,b,c: f"{a}*{b}+{c}=",      lambda a,b,c: (a*b+c)%10),
        ("add4",     Q4, lambda a,b,c,d: f"{a}+{b}+{c}+{d}=", lambda a,b,c,d: (a+b+c+d)%10),
        ("mul2add2", Q4, lambda a,b,c,d: f"{a}*{b}+{c}*{d}=", lambda a,b,c,d: (a*b+c*d)%10),
    ]
    for name, space, fmt, fn in tasks:
        tr, te = split(space)
        _, b = train(mk(tr, fmt, fn), mk(te, fmt, fn), name)
        res[name] = b

    print("\n== Validated specialists (for pipelines) ==")
    trm, tem = split(P)
    mul, mb = train(mk(trm, lambda a,b: f"{a}*{b}=", lambda a,b: (a*b)%10),
                    mk(tem, lambda a,b: f"{a}*{b}=", lambda a,b: (a*b)%10), "mulS", max_epochs=2500)
    tra, tea = split(P)
    add, ab = train(mk(tra, lambda a,b: f"{a}+{b}=", lambda a,b: (a+b)%10),
                    mk(tea, lambda a,b: f"{a}+{b}=", lambda a,b: (a+b)%10), "addS")

    print("\n== Pipelines (composition emergent at inference) ==")
    # muladd: mul -> add
    _, te3 = split(T3)
    p1 = next_tokens(mul, [f"{a}*{b}=" for a,b,c in te3])
    s1 = [tok.decode([t]).strip() for t in p1]
    p2 = next_tokens(add, [f"{s}+{c}=" for s,(a,b,c) in zip(s1, te3) if s.isdigit()])
    valid = [(s,(a,b,c)) for s,(a,b,c) in zip(s1, te3) if s.isdigit()]
    pipe_muladd = sum(tok.decode([t]).strip()==str((a*b+c)%10) for t,(s,(a,b,c)) in zip(p2, valid))/len(te3)

    # mul2add2: mul(a,b)->p, mul(c,d)->q, add(p,q)
    _, te4 = split(Q4)
    pp = [tok.decode([t]).strip() for t in next_tokens(mul, [f"{a}*{b}=" for a,b,c,d in te4])]
    qq = [tok.decode([t]).strip() for t in next_tokens(mul, [f"{c}*{d}=" for a,b,c,d in te4])]
    ok=0
    addq = [f"{p}+{q}=" for p,q in zip(pp,qq) if p.isdigit() and q.isdigit()]
    keep = [(p,q,(a,b,c,d)) for p,q,(a,b,c,d) in zip(pp,qq,te4) if p.isdigit() and q.isdigit()]
    r3 = next_tokens(add, addq)
    pipe_m2a2 = sum(tok.decode([t]).strip()==str((a*b+c*d)%10) for t,(p,q,(a,b,c,d)) in zip(r3,keep))/len(te4)

    print("\n== SUMMARY ==")
    print("single-agent test acc:")
    for k,v in res.items(): print(f"  {k:9s}: {v:.0%}")
    print(f"specialists: mul={mb:.0%}  add={ab:.0%}")
    print(f"pipeline muladd  (mul->add)        : {pipe_muladd:.0%}   vs monolith {res['muladd']:.0%}")
    print(f"pipeline mul2add2(mul,mul->add)    : {pipe_m2a2:.0%}   vs monolith {res['mul2add2']:.0%}")
    print("\nDONE")

if __name__ == "__main__":
    main()
