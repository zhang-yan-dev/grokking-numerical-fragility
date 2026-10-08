"""
Corrected capacity test. Bug in 03b: AutoTokenizer auto-appended <eos> to the
prompt, shifting the answer position by one (so we measured "<eos> after a digit",
not arithmetic). Here we add BOS/EOS manually with add_special_tokens=False.

Task: c = (a+b) mod 10. Metric: teacher-forced next-token accuracy at the answer
position + free 1-token greedy generation. BASE vs FINE-TUNED, held-out test split.
"""
import os, random, torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))
random.seed(0); torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
if tok.pad_token is None: tok.pad_token = tok.eos_token
BOS, EOS, PAD = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id

def prompt(a, b): return f"{a}+{b}="
def full(a, b):   return f"{a}+{b}={(a + b) % 10}"
def enc(s):       return tok(s, add_special_tokens=False)["input_ids"]
def enc_prompt(a, b): return [BOS] + enc(prompt(a, b))
def enc_full(a, b):   return [BOS] + enc(full(a, b)) + [EOS]
def gold_tok(a, b):   return enc(str((a + b) % 10))[0]

# sanity
pf, ff = enc_prompt(3, 4), enc_full(3, 4)
print("full   '3+4=7' ->", ff, tok.convert_ids_to_tokens(ff))
print("prompt '3+4='  ->", pf, tok.convert_ids_to_tokens(pf))
print("clean prefix:", ff[:len(pf)] == pf, "| answer token at idx", len(pf), "=",
      tok.convert_ids_to_tokens([ff[len(pf)]]))

all_pairs = [(a, b) for a in range(10) for b in range(10)]
random.shuffle(all_pairs)
test_pairs, train_pairs = all_pairs[:20], all_pairs[20:]

@torch.no_grad()
def tf_acc(model, pairs):
    model.eval(); c = 0
    for a, b in pairs:
        f = enc_full(a, b); pl = len(enc_prompt(a, b))
        logits = model(torch.tensor([f])).logits[0]
        c += (logits[pl - 1].argmax().item() == f[pl])
    return c / len(pairs)

@torch.no_grad()
def gen_acc(model, pairs, show=0):
    model.eval(); c = 0; s = []
    for a, b in pairs:
        ids = torch.tensor([enc_prompt(a, b)])
        out = model.generate(ids, max_new_tokens=1, do_sample=False, pad_token_id=PAD)
        pred = out[0, -1].item(); ok = pred == gold_tok(a, b); c += ok
        if len(s) < show:
            s.append(f"{prompt(a,b)} -> {tok.decode([pred])!r} (want {(a+b)%10}) {'OK' if ok else 'x'}")
    return c / len(pairs), s

def main():
    model = AutoModelForCausalLM.from_pretrained(LOCAL_DIR, dtype=torch.float32)
    print(f"\n== BASE ==  tf train={tf_acc(model, train_pairs):.0%} test={tf_acc(model, test_pairs):.0%}")
    g, s = gen_acc(model, test_pairs, show=5); print(f"   free-gen test={g:.0%}")
    for x in s: print("   ", x)

    ex = []
    for a, b in train_pairs:
        f = enc_full(a, b); pl = len(enc_prompt(a, b))
        lab = [-100] * len(f); lab[pl] = f[pl]
        ex.append((f, lab))
    ml = max(len(f) for f, _ in ex)
    ids = torch.tensor([f + [PAD] * (ml - len(f)) for f, _ in ex])
    lab = torch.tensor([l + [-100] * (ml - len(l)) for _, l in ex])
    msk = torch.tensor([[1] * len(f) + [0] * (ml - len(f)) for f, _ in ex])
    dl = DataLoader(TensorDataset(ids, lab, msk), batch_size=16, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    for ep in range(600):
        model.train(); tot = 0.0
        for bi, bl, bm in dl:
            opt.zero_grad(); out = model(input_ids=bi, attention_mask=bm, labels=bl)
            out.loss.backward(); opt.step(); tot += out.loss.item()
        if (ep + 1) % 100 == 0:
            print(f"epoch {ep+1:3d} loss={tot/len(dl):.4f} "
                  f"tf_train={tf_acc(model, train_pairs):.0%} tf_test={tf_acc(model, test_pairs):.0%}")

    print(f"\n== FINE-TUNED ==  tf train={tf_acc(model, train_pairs):.0%} test={tf_acc(model, test_pairs):.0%}")
    g, s = gen_acc(model, test_pairs, show=5); print(f"   free-gen test={g:.0%}")
    for x in s: print("   ", x)

if __name__ == "__main__":
    main()
