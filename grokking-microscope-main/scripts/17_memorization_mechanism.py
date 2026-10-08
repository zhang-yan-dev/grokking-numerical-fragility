"""
go/no-go part 2 — does the decomposition pipeline win by GENERALIZING or by MEMORIZING
dense sub-domains?

Claim under test: each specialist only needs to MEMORIZE its small dense 10x10 domain;
the pipeline then covers the sparse 10^4 composite via lookups, not generalization.

Evidence we extract (one seed = one job):
  1) Specialist memorization signature: acc on TRAINING pairs (seen) vs HELD-OUT pairs.
     Memorization => train ~100%, held-out low.
  2) Pipeline mul2add2 accuracy partitioned by whether the (p,q) pair the adder RECEIVES
     was in the adder's training set. If acc(seen-pair) >> acc(heldout-pair), the
     pipeline's success rides on memorized intermediates (claim supported).

Usage: python 17_memorization_mechanism.py <seed> -> results/17_mech_s<seed>.json
"""
import os, sys, json, random, time, torch
torch.set_num_threads(1)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DEV, MAX_EPOCHS = "cpu", 1500
LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", f"17_mech_s{SEED}.json"))
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
    if not ex: return None
    preds = next_tokens(model, [p for p, _ in ex])
    return sum(tok.decode([t]).strip() == a for t, (_, a) in zip(preds, ex)) / len(ex)

def train(train_ex):
    model = make(); rows = []
    for p, a in train_ex:
        f = [BOS] + enc(p + a) + [EOS]; pl = len([BOS] + enc(p))
        lab = [-100] * len(f); lab[pl] = f[pl]; rows.append((f, lab))
    ml = max(len(f) for f, _ in rows)
    ids = torch.tensor([f + [PAD] * (ml - len(f)) for f, _ in rows])
    lab = torch.tensor([l + [-100] * (ml - len(l)) for _, l in rows])
    msk = torch.tensor([[1] * len(f) + [0] * (ml - len(f)) for f, _ in rows])
    dl = DataLoader(TensorDataset(ids, lab, msk), batch_size=16, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for bi, bl, bm in dl:
            opt.zero_grad()
            model(input_ids=bi.to(DEV), attention_mask=bm.to(DEV), labels=bl.to(DEV)).loss.backward()
            opt.step()
    return model

def split_pairs(frac_test=0.2):
    P = [(a, b) for a in range(10) for b in range(10)]
    random.shuffle(P); n = int(len(P) * frac_test)
    return P[n:], P[:n]  # train, heldout

def main():
    t0 = time.time()
    # specialists on dense 80/100 domains
    mtr, mte = split_pairs()
    mul = train([(f"{a}*{b}=", str((a*b)%10)) for a, b in mtr])
    atr, ate = split_pairs()
    add = train([(f"{a}+{b}=", str((a+b)%10)) for a, b in atr])
    atr_set = set(atr)

    # 1) memorization signature
    mul_tr = acc(mul, [(f"{a}*{b}=", str((a*b)%10)) for a, b in mtr])
    mul_te = acc(mul, [(f"{a}*{b}=", str((a*b)%10)) for a, b in mte])
    add_tr = acc(add, [(f"{a}+{b}=", str((a+b)%10)) for a, b in atr])
    add_te = acc(add, [(f"{a}+{b}=", str((a+b)%10)) for a, b in ate])

    # 2) pipeline on mul2add2 held-out, partitioned by adder-input membership
    Q4 = [(a, b, c, d) for a in range(10) for b in range(10) for c in range(10) for d in range(10)]
    random.shuffle(Q4); te4 = Q4[:1000]
    pp = [tok.decode([t]).strip() for t in next_tokens(mul, [f"{a}*{b}=" for a,b,c,d in te4])]
    qq = [tok.decode([t]).strip() for t in next_tokens(mul, [f"{c}*{d}=" for a,b,c,d in te4])]
    keep = [(p,q,t) for p,q,t in zip(pp,qq,te4) if p.isdigit() and q.isdigit()]
    r = next_tokens(add, [f"{p}+{q}=" for p,q,_ in keep])
    seen_ok=seen_n=held_ok=held_n=tot_ok=0
    for tkn,(p,q,(a,b,c,d)) in zip(r, keep):
        correct = tok.decode([tkn]).strip() == str((a*b+c*d)%10)
        tot_ok += correct
        pair = (int(p), int(q))
        if pair in atr_set:
            seen_n += 1; seen_ok += correct
        else:
            held_n += 1; held_ok += correct
    out = {
        "seed": SEED,
        "mul_spec": {"train_acc": round(mul_tr,3), "heldout_acc": round(mul_te,3)},
        "add_spec": {"train_acc": round(add_tr,3), "heldout_acc": round(add_te,3)},
        "pipeline_mul2add2": {
            "overall_acc": round(tot_ok/len(te4),3),
            "acc_adder_pair_SEEN": round(seen_ok/seen_n,3) if seen_n else None,
            "acc_adder_pair_HELDOUT": round(held_ok/held_n,3) if held_n else None,
            "frac_pairs_seen": round(seen_n/(seen_n+held_n),3) if (seen_n+held_n) else None,
        },
        "secs": round(time.time()-t0,1),
    }
    with open(OUT, "w") as f: json.dump(out, f, indent=2)
    print(f"s{SEED} add_spec tr={add_tr:.0%}/te={add_te:.0%}  pipe seen={out['pipeline_mul2add2']['acc_adder_pair_SEEN']} "
          f"held={out['pipeline_mul2add2']['acc_adder_pair_HELDOUT']} ({out['secs']}s)", flush=True)

if __name__ == "__main__":
    main()
