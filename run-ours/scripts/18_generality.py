"""
Generality of the thesis beyond (a*b+c*d) mod 10. We re-test BOTH halves on new settings:
  - different modulus:   (a*b+c*d) mod 7,  mod 12
  - different structure: (a+b)*(c+d) mod 10   (decomposition becomes add,add->mul)

If the coverage-gated grokking transition (mono) AND the memorization-driven pipeline
advantage (mech) replicate across these, the principle is not a mod-10 artifact.

A "family" defines the composite task + its two sub-tasks + the pipeline wiring.
One job = one (family, mode, seed[, n_train]). Launch the grid in parallel.

Usage:
  python 18_generality.py <family> mono <seed> <n_train>  -> 18_<family>_mono_s<seed>_n<n>.json
  python 18_generality.py <family> mech <seed>            -> 18_<family>_mech_s<seed>.json
families: muladd_m7  muladd_m12  addmul_m10
"""
import os, sys, json, random, time, torch
torch.set_num_threads(1)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

FAMILY = sys.argv[1]
MODE = sys.argv[2]
SEED = int(sys.argv[3])
NTRAIN = int(sys.argv[4]) if len(sys.argv) > 4 else None
GROK, DEV = 0.70, "cpu"
TEST_N, TEST_SEED, MAX_EPOCHS = 1000, 12345, 1500

# ---- family definitions -------------------------------------------------------
# comp(a,b,c,d) -> composite answer ; op1/op2 are the two specialist sub-functions ;
# pipe(a,b,c,d, f1, f2) wires specialists at inference and returns the predicted int.
def make_family(name):
    if name.startswith("muladd_m"):
        M = int(name.split("_m")[1])
        comp = lambda a,b,c,d: (a*b + c*d) % M
        comp_fmt = lambda a,b,c,d: f"{a}*{b}+{c}*{d}="
        sub_mul = ("mul", lambda x,y: f"{x}*{y}=", lambda x,y: (x*y)%M)
        sub_add = ("add", lambda x,y: f"{x}+{y}=", lambda x,y: (x+y)%M)
        specs = [sub_mul, sub_add]
        def pipe(te4, run):  # run(spec_idx, prompts)->list[str]
            p = run(0, [f"{a}*{b}=" for a,b,c,d in te4])
            q = run(0, [f"{c}*{d}=" for a,b,c,d in te4])
            keep = [(pi,qi,t) for pi,qi,t in zip(p,q,te4) if pi.isdigit() and qi.isdigit()]
            r = run(1, [f"{pi}+{qi}=" for pi,qi,_ in keep])
            return keep, r, 1  # adder is spec idx 1
        return M, comp, comp_fmt, specs, pipe
    if name.startswith("addmul_m"):
        M = int(name.split("_m")[1])
        comp = lambda a,b,c,d: ((a+b) * (c+d)) % M
        comp_fmt = lambda a,b,c,d: f"({a}+{b})*({c}+{d})="
        sub_add = ("add", lambda x,y: f"{x}+{y}=", lambda x,y: (x+y)%M)
        sub_mul = ("mul", lambda x,y: f"{x}*{y}=", lambda x,y: (x*y)%M)
        specs = [sub_add, sub_mul]
        def pipe(te4, run):
            s = run(0, [f"{a}+{b}=" for a,b,c,d in te4])
            t = run(0, [f"{c}+{d}=" for a,b,c,d in te4])
            keep = [(si,ti,tt) for si,ti,tt in zip(s,t,te4) if si.isdigit() and ti.isdigit()]
            r = run(1, [f"{si}*{ti}=" for si,ti,_ in keep])
            return keep, r, 1  # multiplier is spec idx 1
        return M, comp, comp_fmt, specs, pipe
    raise ValueError(name)

M, COMP, COMP_FMT, SPECS, PIPE = make_family(FAMILY)
OUTNAME = f"18_{FAMILY}_mono_s{SEED}_n{NTRAIN}.json" if MODE=="mono" else f"18_{FAMILY}_mech_s{SEED}.json"
LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", OUTNAME))
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

def train(train_ex, test_ex=None, eval_grok=False, bs=16, lr=3e-3, wd=0.01, eval_every=100):
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
    best, grok_ep = 0.0, None
    for ep in range(1, MAX_EPOCHS+1):
        model.train()
        for bi, bl, bm in dl:
            opt.zero_grad()
            model(input_ids=bi.to(DEV), attention_mask=bm.to(DEV), labels=bl.to(DEV)).loss.backward()
            opt.step()
        if eval_grok and (ep % eval_every == 0 or ep == MAX_EPOCHS):
            te = acc(model, test_ex)
            if te > best: best = te
            if grok_ep is None and te >= GROK: grok_ep = ep
            if te >= 0.985: break
    return model, best, grok_ep

def all_pairs(): return [(x, y) for x in range(10) for y in range(10)]

def run_mono():
    t0 = time.time()
    Q4 = [(a,b,c,d) for a in range(10) for b in range(10) for c in range(10) for d in range(10)]
    rng = random.Random(TEST_SEED); pool = Q4[:]; rng.shuffle(pool)
    test_t, train_pool = pool[:TEST_N], pool[TEST_N:]
    rj = random.Random(SEED); rj.shuffle(train_pool); train_t = train_pool[:NTRAIN]
    torch.manual_seed(SEED)
    _, best, gep = train([(COMP_FMT(*t), str(COMP(*t))) for t in train_t],
                         [(COMP_FMT(*t), str(COMP(*t))) for t in test_t], eval_grok=True)
    out = {"family": FAMILY, "mod": M, "seed": SEED, "n_train": NTRAIN,
           "coverage": round(NTRAIN/10000,3), "best": round(best,3), "grok": best>=GROK,
           "grok_ep": gep, "secs": round(time.time()-t0,1)}
    json.dump(out, open(OUT,"w"), indent=2)
    print(f"{FAMILY} mono s{SEED} cov{out['coverage']:.0%} best={best:.0%} grok={best>=GROK} ({out['secs']}s)", flush=True)

def run_mech():
    t0 = time.time()
    random.seed(SEED); torch.manual_seed(SEED)
    # train both specialists on dense 80/100 domains
    models, specinfo = [], []
    for (sname, sfmt, sfn) in SPECS:
        P = all_pairs(); random.shuffle(P); n = int(len(P)*0.2)
        tr, te = P[n:], P[:n]
        mdl, _, _ = train([(sfmt(x,y), str(sfn(x,y))) for x,y in tr])
        tr_acc = acc(mdl, [(sfmt(x,y), str(sfn(x,y))) for x,y in tr])
        te_acc = acc(mdl, [(sfmt(x,y), str(sfn(x,y))) for x,y in te])
        models.append(mdl); specinfo.append({"name": sname, "train_set": set(tr),
                                              "train_acc": round(tr_acc,3), "heldout_acc": round(te_acc,3)})
    def run(idx, prompts): return [tok.decode([t]).strip() for t in next_tokens(models[idx], prompts)]
    Q4 = [(a,b,c,d) for a in range(10) for b in range(10) for c in range(10) for d in range(10)]
    random.shuffle(Q4); te4 = Q4[:1000]
    keep, r, second_idx = PIPE(te4, run)
    seen = specinfo[second_idx]["train_set"]
    seen_ok=seen_n=held_ok=held_n=tot=0
    for tkn,(x,y,t) in zip(r, keep):
        ok = tkn == str(COMP(*t)); tot += ok  # tkn is already a decoded, stripped string
        if (int(x),int(y)) in seen: seen_n+=1; seen_ok+=ok
        else: held_n+=1; held_ok+=ok
    out = {"family": FAMILY, "mod": M, "seed": SEED,
           "specialists": [{k:v for k,v in s.items() if k!="train_set"} for s in specinfo],
           "pipeline": {"overall_acc": round(tot/len(te4),3),
                        "acc_2ndspec_pair_SEEN": round(seen_ok/seen_n,3) if seen_n else None,
                        "acc_2ndspec_pair_HELDOUT": round(held_ok/held_n,3) if held_n else None,
                        "frac_seen": round(seen_n/(seen_n+held_n),3) if (seen_n+held_n) else None},
           "secs": round(time.time()-t0,1)}
    json.dump(out, open(OUT,"w"), indent=2)
    p = out["pipeline"]
    print(f"{FAMILY} mech s{SEED} overall={p['overall_acc']:.0%} seen={p['acc_2ndspec_pair_SEEN']} held={p['acc_2ndspec_pair_HELDOUT']} ({out['secs']}s)", flush=True)

if __name__ == "__main__":
    (run_mono if MODE == "mono" else run_mech)()
