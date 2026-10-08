"""
M0/M1 — single-agent capability frontier vs multi-agent pipelines (grokking-aware).

Why this supersedes script 12:
  Script 12 early-stopped on `train>=0.999 and patience`, i.e. right after the model
  MEMORIZES the train set. But grokking generalization happens LATER (add2 groks at
  ep500-700, well after train saturates ~ep200). So script 12 stopped just before the
  phase transition and reported add2=30%, mul2=50% -- artifacts, not capability.

Fix: early-stop ONLY when test>=target. Otherwise keep training to max_epochs and
track the best held-out accuracy. Small batch (bs=16) = more gradient steps = grokking
(the regime where script 11's add2 reached 95%). Evaluation is one batched forward over
all prompts on GPU (teacher-forced argmax at the answer position == greedy 1-token gen).

We sweep increasing compositional depth to find where a single ~11,856-param agent caps,
then test whether a pipeline of validated specialists stays accurate past that ceiling.
"""
import os, random, time, torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))
random.seed(0); torch.manual_seed(0)
# NOTE: at 11,856 params CPU is faster than GPU (kernel-launch overhead dominates) AND
# grokking is cleaner/reproducible on CPU (GPU numerics gave noisy, weak grokking:
# add2 45% vs CPU 95%). Force CPU here; keep GPU for future larger-model work.
DEV = "cpu" if os.environ.get("FORCE_CPU", "1") == "1" else ("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
if tok.pad_token is None: tok.pad_token = tok.eos_token
BOS, EOS, PAD = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
def enc(s): return tok(s, add_special_tokens=False)["input_ids"]
def make(): return AutoModelForCausalLM.from_pretrained(LOCAL_DIR, dtype=torch.float32).to(DEV)

@torch.no_grad()
def next_tokens(model, prompts, bs=2048):
    """Greedy next-token id after each prompt, batched (GPU)."""
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

def train(train_ex, test_ex, name, max_epochs=2000, bs=16, lr=3e-3, wd=0.01,
          eval_every=100, target=0.97, log=True):
    """Grokking-aware: stop ONLY on test>=target, else run full budget tracking best."""
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
    best, bep, grok_ep, t0 = 0.0, 0, None, time.time()
    prev_te = 0.0
    for ep in range(1, max_epochs + 1):
        model.train()
        for bi, bl, bm in dl:
            opt.zero_grad()
            model(input_ids=bi.to(DEV), attention_mask=bm.to(DEV), labels=bl.to(DEV)).loss.backward()
            opt.step()
        if ep % eval_every == 0 or ep == max_epochs:
            tr, te = acc(model, train_ex), acc(model, test_ex)
            if te > best: best, bep = te, ep
            if grok_ep is None and te - prev_te >= 0.30: grok_ep = ep  # phase transition
            prev_te = te
            if log: print(f"  [{name}] ep{ep:4d} train={tr:.0%} test={te:.0%}")
            if te >= target: break
    gtxt = f" grok@{grok_ep}" if grok_ep else ""
    print(f"  [{name}] -> best test {best:.0%} @ ep{bep}{gtxt}  ({time.time()-t0:.0f}s)")
    return model, best

def split(space, frac=0.2, cap_train=None, cap_test=None):
    sp = list(space); random.shuffle(sp); n = int(len(sp) * frac)
    test, tr = sp[:n], sp[n:]
    if cap_test: test = test[:cap_test]
    if cap_train: tr = tr[:cap_train]
    return tr, test
def mk(space, fmt, fn): return [(fmt(*t), str(fn(*t))) for t in space]

def main():
    print(f"device={DEV}  {torch.cuda.get_device_name(0) if DEV=='cuda' else ''}")
    P = [(a, b) for a in range(10) for b in range(10)]
    T3 = [(a, b, c) for a in range(10) for b in range(10) for c in range(10)]
    Q4 = [(a, b, c, d) for a in range(10) for b in range(10) for c in range(10) for d in range(10)]
    res = {}

    print("\n== Single-agent capability frontier ==")
    # (name, space, fmt, fn, max_epochs, cap_train, cap_test)
    tasks = [
        ("add2",     P,  lambda a,b: f"{a}+{b}=",             lambda a,b: (a+b)%10,           1500, None, None),
        ("mul2",     P,  lambda a,b: f"{a}*{b}=",             lambda a,b: (a*b)%10,           1500, None, None),
        ("add3",     T3, lambda a,b,c: f"{a}+{b}+{c}=",       lambda a,b,c: (a+b+c)%10,       1500, 700, 200),
        ("muladd",   T3, lambda a,b,c: f"{a}*{b}+{c}=",       lambda a,b,c: (a*b+c)%10,       1500, 700, 200),
        ("add4",     Q4, lambda a,b,c,d: f"{a}+{b}+{c}+{d}=", lambda a,b,c,d: (a+b+c+d)%10,   1500, 900, 250),
        ("mul2add2", Q4, lambda a,b,c,d: f"{a}*{b}+{c}*{d}=", lambda a,b,c,d: (a*b+c*d)%10,   1500, 900, 250),
    ]
    for name, space, fmt, fn, mx, ctr, cte in tasks:
        tr, te = split(space, cap_train=ctr, cap_test=cte)
        _, b = train(mk(tr, fmt, fn), mk(te, fmt, fn), name, max_epochs=mx)
        res[name] = b

    print("\n== Validated specialists (for pipelines) ==")
    trm, tem = split(P)
    mul, mb = train(mk(trm, lambda a,b: f"{a}*{b}=", lambda a,b: (a*b)%10),
                    mk(tem, lambda a,b: f"{a}*{b}=", lambda a,b: (a*b)%10), "mulS", max_epochs=1500)
    tra, tea = split(P)
    add, ab = train(mk(tra, lambda a,b: f"{a}+{b}=", lambda a,b: (a+b)%10),
                    mk(tea, lambda a,b: f"{a}+{b}=", lambda a,b: (a+b)%10), "addS", max_epochs=1500)

    print("\n== Pipelines (composition emergent at inference) ==")
    # muladd: mul(a,b) -> add(.,c)
    _, te3 = split(T3, cap_test=300)
    s1 = [tok.decode([t]).strip() for t in next_tokens(mul, [f"{a}*{b}=" for a,b,c in te3])]
    valid = [(s, (a,b,c)) for s, (a,b,c) in zip(s1, te3) if s.isdigit()]
    p2 = next_tokens(add, [f"{s}+{c}=" for s,(a,b,c) in valid])
    pipe_muladd = sum(tok.decode([t]).strip() == str((a*b+c)%10)
                      for t,(s,(a,b,c)) in zip(p2, valid)) / len(te3)

    # mul2add2: mul(a,b)->p, mul(c,d)->q, add(p,q)
    _, te4 = split(Q4, cap_test=300)
    pp = [tok.decode([t]).strip() for t in next_tokens(mul, [f"{a}*{b}=" for a,b,c,d in te4])]
    qq = [tok.decode([t]).strip() for t in next_tokens(mul, [f"{c}*{d}=" for a,b,c,d in te4])]
    keep = [(p,q,(a,b,c,d)) for p,q,(a,b,c,d) in zip(pp,qq,te4) if p.isdigit() and q.isdigit()]
    r3 = next_tokens(add, [f"{p}+{q}=" for p,q,_ in keep])
    pipe_m2a2 = sum(tok.decode([t]).strip() == str((a*b+c*d)%10)
                    for t,(p,q,(a,b,c,d)) in zip(r3, keep)) / len(te4)

    print("\n== SUMMARY ==")
    print("single-agent test acc:")
    for k, v in res.items(): print(f"  {k:9s}: {v:.0%}")
    print(f"specialists: mul={mb:.0%}  add={ab:.0%}")
    print(f"pipeline muladd   (mul->add)        : {pipe_muladd:.0%}   vs monolith {res['muladd']:.0%}")
    print(f"pipeline mul2add2 (mul,mul->add)    : {pipe_m2a2:.0%}   vs monolith {res['mul2add2']:.0%}")
    print("\nDONE")

if __name__ == "__main__":
    main()
