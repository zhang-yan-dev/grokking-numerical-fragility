"""
M0-fixed — reliable single-agent capability frontier, then a validated pipeline.

Lesson from script 10: batch_size too large (full-batch on 100-example sets) starved
the optimizer of gradient steps, so specialists never grokked (8-11% vs the 90% we
know is reachable). Here we use a consistent small-batch recipe with weight decay,
periodic eval, and early stopping, and we LOG convergence so results are trustworthy.

We map what a single ~11,856-param agent can learn:
    add2  = (a+b) mod 10        mul2  = (a*b) mod 10
    add3  = (a+b+c) mod 10      muladd= (a*b+c) mod 10
Then we test the pipeline  multiplier->adder  on muladd, using validated components.
"""
import os, random, torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))
random.seed(0); torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
if tok.pad_token is None: tok.pad_token = tok.eos_token
BOS, EOS, PAD = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
def enc(s): return tok(s, add_special_tokens=False)["input_ids"]
def make(): return AutoModelForCausalLM.from_pretrained(LOCAL_DIR, dtype=torch.float32)

@torch.no_grad()
def pred(model, p):
    model.eval()
    out = model.generate(torch.tensor([[BOS] + enc(p)]), max_new_tokens=1,
                         do_sample=False, pad_token_id=PAD)
    return tok.decode([out[0, -1].item()]).strip()

@torch.no_grad()
def acc(model, ex):
    return sum(pred(model, p) == a for p, a in ex) / len(ex)

def train(train_ex, test_ex, name, max_epochs=2000, bs=16, lr=3e-3, wd=0.01,
          eval_every=100, target=0.98, patience=4):
    model = make()
    rows = []
    for p, a in train_ex:
        f = [BOS] + enc(p + a) + [EOS]; pl = len([BOS] + enc(p))
        lab = [-100] * len(f); lab[pl] = f[pl]; rows.append((f, lab))
    ml = max(len(f) for f, _ in rows)
    ids = torch.tensor([f + [PAD] * (ml - len(f)) for f, _ in rows])
    lab = torch.tensor([l + [-100] * (ml - len(l)) for _, l in rows])
    msk = torch.tensor([[1] * len(f) + [0] * (ml - len(f)) for f, _ in rows])
    dl = DataLoader(TensorDataset(ids, lab, msk), batch_size=bs, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, best_ep, since = 0.0, 0, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        for bi, bl, bm in dl:
            opt.zero_grad()
            model(input_ids=bi, attention_mask=bm, labels=bl).loss.backward(); opt.step()
        if ep % eval_every == 0 or ep == max_epochs:
            tr, te = acc(model, train_ex), acc(model, test_ex)
            print(f"  [{name}] ep{ep:4d} train={tr:.0%} test={te:.0%}")
            if te > best: best, best_ep, since = te, ep, 0
            else: since += 1
            if te >= target or (tr >= 0.999 and since >= patience): break
    print(f"  [{name}] -> best test {best:.0%} @ ep{best_ep}")
    return model, best

def split(space, frac_test=0.2):
    sp = list(space); random.shuffle(sp)
    n = int(len(sp) * frac_test)
    return sp[n:], sp[:n]

def make_ex(triples_or_pairs, fmt, fn):
    return [(fmt(*t), str(fn(*t))) for t in triples_or_pairs]

def main():
    pairs = [(a, b) for a in range(10) for b in range(10)]
    triples = [(a, b, c) for a in range(10) for b in range(10) for c in range(10)]

    print("== Single-agent capability frontier ==")
    results = {}

    for name, space, fmt, fn, mx in [
        ("add2",   pairs,   lambda a, b: f"{a}+{b}=",       lambda a, b: (a + b) % 10,        2000),
        ("mul2",   pairs,   lambda a, b: f"{a}*{b}=",       lambda a, b: (a * b) % 10,        3000),
        ("add3",   triples, lambda a, b, c: f"{a}+{b}+{c}=", lambda a, b, c: (a + b + c) % 10, 1500),
        ("muladd", triples, lambda a, b, c: f"{a}*{b}+{c}=", lambda a, b, c: (a * b + c) % 10, 1500),
    ]:
        tr_sp, te_sp = split(space)
        _, best = train(make_ex(tr_sp, fmt, fn), make_ex(te_sp, fmt, fn), name, max_epochs=mx)
        results[name] = best

    print("\n== Validated pipeline on muladd = mul2 -> add2 ==")
    # train fresh, well-converged specialists
    tr2, te2 = split(pairs)
    mul, mul_best = train(make_ex(tr2, lambda a, b: f"{a}*{b}=", lambda a, b: (a * b) % 10),
                          make_ex(te2, lambda a, b: f"{a}*{b}=", lambda a, b: (a * b) % 10),
                          "mul2*", max_epochs=3000)
    tr3, te3 = split(pairs)
    add, add_best = train(make_ex(tr3, lambda a, b: f"{a}+{b}=", lambda a, b: (a + b) % 10),
                          make_ex(te3, lambda a, b: f"{a}+{b}=", lambda a, b: (a + b) % 10),
                          "add2*", max_epochs=2000)
    _, te_tr = split(triples)
    ok = 0
    for a, b, c in te_tr:
        p = pred(mul, f"{a}*{b}=")
        if not p.isdigit(): continue
        v = pred(add, f"{p}+{c}=")
        ok += (v == str((a * b + c) % 10))
    pipe = ok / len(te_tr)

    print("\n== SUMMARY (single-agent test acc) ==")
    for k, v in results.items(): print(f"  {k:7s}: {v:.0%}")
    print(f"  pipeline mul2->add2 on muladd: {pipe:.0%} "
          f"(components: mul2={mul_best:.0%}, add2={add_best:.0%})")
    print("\nDONE")

if __name__ == "__main__":
    main()
