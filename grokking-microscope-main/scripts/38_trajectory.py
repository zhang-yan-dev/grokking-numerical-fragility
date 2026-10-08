"""
Grokking TEMPORAL DYNAMICS (trajectory) for (a+b) mod 10 -- the train-first / test-delayed
signature that our best-held-out grok definition (tau=0.70) summarizes. Logs train AND test
answer-token accuracy and answer-token cross-entropy every LOG_EVERY epochs, with NO early
stop, so both the delayed-generalization transition and the test-loss plateau -> collapse
(plateau near ln(M)=ln 10 ~= 2.30) are visible end to end.

Reviewer-2 control: shows we examined the temporal dynamics, not only the best-acc summary.

Usage: python 38_trajectory.py <seed> [wd]   ->  results/38_traj_s<seed>.json
"""
import os, sys, json, random, time, torch
torch.set_num_threads(1)  # one core per process; launch seeds in parallel
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
WD   = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1  # grok-friendly regime (Omnigrok peak)
M, DEV, GROK = 10, "cpu", 0.70
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", 1200))  # override for smoke tests
LOG_EVERY, LR, BS = 10, 3e-3, 16
LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", f"38_traj_wd{WD}_s{SEED}.json"))
random.seed(SEED); torch.manual_seed(SEED)
tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
if tok.pad_token is None: tok.pad_token = tok.eos_token
BOS, EOS, PAD = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
def enc(s): return tok(s, add_special_tokens=False)["input_ids"]
def make(): return AutoModelForCausalLM.from_pretrained(LOCAL_DIR, dtype=torch.float32).to(DEV)

def build(ex):
    rows = []
    for p, a in ex:
        f = [BOS] + enc(p + a) + [EOS]; pl = len([BOS] + enc(p))
        lab = [-100] * len(f); lab[pl] = f[pl]; rows.append((f, lab))
    ml = max(len(f) for f, _ in rows)
    ids = torch.tensor([f + [PAD] * (ml - len(f)) for f, _ in rows])
    lab = torch.tensor([l + [-100] * (ml - len(l)) for _, l in rows])
    msk = torch.tensor([[1] * len(f) + [0] * (ml - len(f)) for f, _ in rows])
    return ids, lab, msk

@torch.no_grad()
def evaluate(model, ids, lab, msk, bs=4096):
    """answer-token accuracy and mean cross-entropy over the single supervised position."""
    model.eval(); n = ids.size(0); correct = 0; loss_sum = 0.0; cnt = 0
    for i in range(0, n, bs):
        bi, bl, bm = ids[i:i+bs], lab[i:i+bs], msk[i:i+bs]
        logits = model(input_ids=bi, attention_mask=bm).logits
        pos = (bl != -100).int().argmax(1)           # the lone answer position per row
        ar = torch.arange(bi.size(0))
        lg = logits[ar, pos]; tgt = bl[ar, pos]
        correct += (lg.argmax(-1) == tgt).sum().item()
        loss_sum += torch.nn.functional.cross_entropy(lg, tgt, reduction="sum").item()
        cnt += bi.size(0)
    return correct / cnt, loss_sum / cnt

def main():
    t0 = time.time()
    pairs = [(a, b) for a in range(10) for b in range(10)]
    random.shuffle(pairs); n = int(len(pairs) * 0.2)
    test, train = pairs[:n], pairs[n:]
    mk = lambda S: [(f"{a}+{b}=", str((a + b) % M)) for a, b in S]
    tr_ids, tr_lab, tr_msk = build(mk(train))
    te_ids, te_lab, te_msk = build(mk(test))
    model = make()
    dl = DataLoader(TensorDataset(tr_ids, tr_lab, tr_msk), batch_size=BS, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    hist = {"epoch": [], "train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}
    grok_ep = train_sat_ep = None; best = 0.0
    for ep in range(0, MAX_EPOCHS + 1):
        if ep > 0:
            model.train()
            for bi, bl, bm in dl:
                opt.zero_grad()
                model(input_ids=bi, attention_mask=bm, labels=bl).loss.backward()
                opt.step()
        if ep % LOG_EVERY == 0 or ep == MAX_EPOCHS:
            tra, trl = evaluate(model, tr_ids, tr_lab, tr_msk)
            tea, tel = evaluate(model, te_ids, te_lab, te_msk)
            hist["epoch"].append(ep)
            hist["train_acc"].append(round(tra, 4)); hist["test_acc"].append(round(tea, 4))
            hist["train_loss"].append(round(trl, 4)); hist["test_loss"].append(round(tel, 4))
            if tea > best: best = tea
            if train_sat_ep is None and tra >= 0.99: train_sat_ep = ep
            if grok_ep is None and tea >= GROK: grok_ep = ep
    out = {"task": "add2", "M": M, "seed": SEED, "wd": WD, "grok_thresh": GROK,
           "best_test": round(best, 3), "grok": best >= GROK, "grok_ep": grok_ep,
           "train_sat_ep": train_sat_ep,
           "delay": (grok_ep - train_sat_ep) if (grok_ep and train_sat_ep) else None,
           "n_train": len(train), "hist": hist, "secs": round(time.time() - t0, 1)}
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"traj s{SEED} wd{WD}: best={best:.0%} grok_ep={grok_ep} "
          f"train_sat={train_sat_ep} delay={out['delay']} ({out['secs']}s)", flush=True)

if __name__ == "__main__":
    main()
