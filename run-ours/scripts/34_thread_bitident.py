"""
Reviewer-strengthening (knife-edge mechanism): verify the asserted "t=4 == t=16 bit-identical,
because the BLAS reduction tree saturates by 4 threads at hidden dim 16" claim EXPLICITLY.

We train (a+b) mod 10 to completion under a given OMP/MKL/torch thread count (env set BEFORE
importing torch, exactly like script 22) and dump the full final state_dict to a .pt file.
A separate compare step (mode=compare) loads two such dumps and runs torch.equal on every
parameter, reporting max abs diff. If t4 and t16 are bit-identical, every diff is exactly 0.0
and the "knife-edge flips come from reduction-ORDER (t1 vs t4), not thread count per se" story
is nailed down: t4 and t16 use the SAME reduction order, so they cannot differ.

Usage:
  python 34_thread_bitident.py train <threads> <seed>   -> results/34_w_t<threads>_s<seed>.pt
  python 34_thread_bitident.py compare <seed> <tA> <tB> -> results/34_compare_s<seed>.json
"""
import os, sys, json, time

MODE = sys.argv[1]

def train(threads, seed):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    import torch, random
    torch.set_num_threads(threads)
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    M, WD, MAX_EPOCHS, DEV = 10, 0.01, 1500, "cpu"
    LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))
    OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", f"34_w_t{threads}_s{seed}.pt"))
    tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    BOS, EOS, PAD = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
    def enc(s): return tok(s, add_special_tokens=False)["input_ids"]
    t0 = time.time()
    random.seed(seed); torch.manual_seed(seed)
    P = [(a, b) for a in range(10) for b in range(10)]
    random.shuffle(P); n = int(len(P) * 0.2); tr = P[n:]
    model = AutoModelForCausalLM.from_pretrained(LOCAL_DIR, dtype=torch.float32).to(DEV)
    rows = []
    for a, b in tr:
        p, ans = f"{a}+{b}=", str((a+b) % M)
        f = [BOS] + enc(p + ans) + [EOS]; pl = len([BOS] + enc(p))
        lab = [-100]*len(f); lab[pl] = f[pl]; rows.append((f, lab))
    ml = max(len(f) for f, _ in rows)
    ids = torch.tensor([f + [PAD]*(ml-len(f)) for f, _ in rows])
    lab = torch.tensor([l + [-100]*(ml-len(l)) for _, l in rows])
    msk = torch.tensor([[1]*len(f) + [0]*(ml-len(f)) for f, _ in rows])
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(ids, lab, msk), batch_size=16, shuffle=True, generator=g)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=WD)
    for ep in range(1, MAX_EPOCHS+1):
        model.train()
        for bi, bl, bm in dl:
            opt.zero_grad()
            model(input_ids=bi, attention_mask=bm, labels=bl).loss.backward()
            opt.step()
    torch.save(model.state_dict(), OUT)
    print(f"train t{threads} s{seed} done ({time.time()-t0:.0f}s) -> {OUT}", flush=True)

def compare(seed, tA, tB):
    import torch
    R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    a = torch.load(os.path.join(R, f"34_w_t{tA}_s{seed}.pt"))
    b = torch.load(os.path.join(R, f"34_w_t{tB}_s{seed}.pt"))
    keys = list(a.keys())
    maxdiff = 0.0; allequal = True; per = {}
    for k in keys:
        eq = bool(torch.equal(a[k], b[k]))
        d = float((a[k].float() - b[k].float()).abs().max())
        per[k] = {"equal": eq, "max_abs_diff": d}
        maxdiff = max(maxdiff, d); allequal = allequal and eq
    out = {"seed": seed, "tA": tA, "tB": tB, "all_bit_identical": allequal,
           "global_max_abs_diff": maxdiff, "n_params_tensors": len(keys),
           "per_param": per}
    OUT = os.path.join(R, f"34_compare_s{seed}_t{tA}vs{tB}.json")
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"compare t{tA} vs t{tB} s{seed}: all_bit_identical={allequal} "
          f"global_max_abs_diff={maxdiff:.2e}", flush=True)

if __name__ == "__main__":
    if MODE == "train":
        train(int(sys.argv[2]), int(sys.argv[3]))
    else:
        compare(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
