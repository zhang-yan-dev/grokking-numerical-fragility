"""
Step 2 (glass-box core) — what circuit does a 12K agent form when it GENERALIZES vs when
it MEMORIZES? We train many add-specialists on (a+b) mod M (the seed lottery gives both
grokkers and memorizers) and read out, for each, the mechanism:

  * Fourier structure of the 0..9 digit embeddings (mod-M DFT over the digit axis).
    The generalizing modular-arithmetic circuit (Power/Nanda) represents digits on a
    CIRCLE -> spectral power concentrates at a few key frequencies. A lookup-table
    (memorizer) has no such concentration.
  * Effective rank of the digit-embedding matrix (participation ratio of singular
    values). Fourier/circular code -> LOW rank; memorized lookup -> HIGH rank.
  * Periodicity of the answer logits over the (a,b) grid (2-D mod-M DFT concentration).

Hypothesis (the paper's mechanistic punchline): generalization <=> Fourier-structured
low-rank circuit; the pipeline specialists that make decomposition "work" are largely
MEMORIZERS (high rank, no periodicity) -> decomposition succeeds via lookup, not via the
elegant circuit.

Usage: python 20_mechanism_fourier.py <M> <n_seeds>  -> results/20_fourier_m<M>.json
"""
import os, sys, json, random, time, math, torch
torch.set_num_threads(2)
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

M = int(sys.argv[1]) if len(sys.argv) > 1 else 10
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 12
SEED0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
DEV, MAX_EPOCHS = "cpu", 1500
LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", f"20_fourier_m{M}_s{SEED0}.json"))
tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
if tok.pad_token is None: tok.pad_token = tok.eos_token
BOS, EOS, PAD = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
def enc(s): return tok(s, add_special_tokens=False)["input_ids"]
DIGIT_IDS = [enc(str(d))[0] for d in range(10)]   # token id for each digit 0..9
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
        out.append(logits[torch.arange(len(ch)), idx])
    return torch.cat(out, 0)  # [N, vocab]

def acc_from_logits(L, answers):
    pred = L.argmax(-1).tolist()
    return sum(tok.decode([p]).strip() == a for p, a in zip(pred, answers)) / len(answers)

def train(train_ex, bs=16, lr=3e-3, wd=0.01):
    model = make(); rows = []
    for p, a in train_ex:
        f = [BOS] + enc(p + a) + [EOS]; pl = len([BOS] + enc(p))
        lab = [-100]*len(f); lab[pl] = f[pl]; rows.append((f, lab))
    ml = max(len(f) for f, _ in rows)
    ids = torch.tensor([f + [PAD]*(ml-len(f)) for f, _ in rows])
    lab = torch.tensor([l + [-100]*(ml-len(l)) for _, l in rows])
    msk = torch.tensor([[1]*len(f) + [0]*(ml-len(f)) for f, _ in rows])
    dl = DataLoader(TensorDataset(ids, lab, msk), batch_size=bs, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for ep in range(1, MAX_EPOCHS+1):
        model.train()
        for bi, bl, bm in dl:
            opt.zero_grad()
            model(input_ids=bi.to(DEV), attention_mask=bm.to(DEV), labels=bl.to(DEV)).loss.backward()
            opt.step()
    return model

def fourier_concentration(vec_real):
    """vec_real: [M] real signal over residues. Return fraction of non-DC power in the
    single strongest frequency (1..floor(M/2))."""
    f = np.fft.fft(vec_real)
    p = np.abs(f)**2
    nondc = p[1:]                      # drop DC
    return float(nondc.max() / (nondc.sum() + 1e-12))

def analyze(model):
    # digit embeddings restricted to residues 0..M-1
    E = model.get_input_embeddings().weight.detach().cpu().numpy()[DIGIT_IDS][:M]  # [M,16]
    # (a) per-dim Fourier concentration of the embedding code, averaged over dims weighted by variance
    cols = []
    for d in range(E.shape[1]):
        col = E[:, d] - E[:, d].mean()
        if np.abs(col).sum() < 1e-8: continue
        cols.append(fourier_concentration(col))
    emb_fourier = float(np.mean(cols)) if cols else 0.0
    # (b) effective rank (participation ratio of singular values) of centered embeddings
    Ec = E - E.mean(0, keepdims=True)
    s = np.linalg.svd(Ec, compute_uv=False)
    eff_rank = float((s.sum()**2) / ((s**2).sum() + 1e-12))
    # (c) answer-logit periodicity over the (a,b) grid: logit of the TRUE answer token
    prompts, ans = [], []
    for a in range(M):
        for b in range(M):
            prompts.append(f"{a}+{b}="); ans.append((a + b) % M)
    L = next_tokens(model, prompts)  # [M*M, vocab]
    ans_ids = [DIGIT_IDS[r] for r in ans]
    true_logit = L[torch.arange(len(ans)), torch.tensor(ans_ids)].detach().cpu().numpy().reshape(M, M)
    F2 = np.abs(np.fft.fft2(true_logit - true_logit.mean()))**2
    F2[0, 0] = 0.0
    logit_fourier = float(F2.max() / (F2.sum() + 1e-12))
    return emb_fourier, eff_rank, logit_fourier

def main():
    t0 = time.time()
    P = [(a, b) for a in range(10) for b in range(10)]
    results = []
    for seed in range(SEED0, SEED0 + NSEED):
        random.seed(seed); torch.manual_seed(seed)
        sp = P[:]; random.shuffle(sp); n = int(len(sp)*0.2)
        te, tr = sp[:n], sp[n:]
        model = train([(f"{a}+{b}=", str((a+b)%M)) for a, b in tr])
        Lte = next_tokens(model, [f"{a}+{b}=" for a, b in te])
        held = acc_from_logits(Lte, [str((a+b)%M) for a, b in te])
        Ltr = next_tokens(model, [f"{a}+{b}=" for a, b in tr])
        train_acc = acc_from_logits(Ltr, [str((a+b)%M) for a, b in tr])
        ef, er, lf = analyze(model)
        row = {"seed": seed, "train_acc": round(train_acc,3), "heldout_acc": round(held,3),
               "grok": held >= 0.70, "emb_fourier": round(ef,3), "eff_rank": round(er,2),
               "logit_fourier": round(lf,3)}
        results.append(row)
        print(f"M{M} s{seed} held={held:.0%} grok={held>=0.70} emb_fourier={ef:.2f} eff_rank={er:.2f} logit_fourier={lf:.2f}", flush=True)
    # summary: contrast grokkers vs memorizers
    gk = [r for r in results if r["grok"]]; mm = [r for r in results if not r["grok"]]
    def avg(rows, k): return round(float(np.mean([r[k] for r in rows])), 3) if rows else None
    out = {"M": M, "n_seeds": NSEED, "results": results,
           "grokkers": {"n": len(gk), "emb_fourier": avg(gk,"emb_fourier"), "eff_rank": avg(gk,"eff_rank"), "logit_fourier": avg(gk,"logit_fourier")},
           "memorizers": {"n": len(mm), "emb_fourier": avg(mm,"emb_fourier"), "eff_rank": avg(mm,"eff_rank"), "logit_fourier": avg(mm,"logit_fourier")},
           "secs": round(time.time()-t0,1)}
    json.dump(out, open(OUT,"w"), indent=2)
    print(f"\nGROKKERS  n={len(gk)} emb_fourier={out['grokkers']['emb_fourier']} eff_rank={out['grokkers']['eff_rank']} logit_fourier={out['grokkers']['logit_fourier']}")
    print(f"MEMORIZERS n={len(mm)} emb_fourier={out['memorizers']['emb_fourier']} eff_rank={out['memorizers']['eff_rank']} logit_fourier={out['memorizers']['logit_fourier']}")
    print(f"-> {OUT} ({out['secs']}s)")

if __name__ == "__main__":
    main()
