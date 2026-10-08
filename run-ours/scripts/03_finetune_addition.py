"""
Can fine-tuning give Glimmer-1-Base a real, measurable ability?
Narrow task: single-digit addition  "a+b=c"  (a,b in 0..9, c in 0..18).
We measure exact-match accuracy of the BASE model vs the FINE-TUNED model,
on a held-out test split, to distinguish capacity from chance.
"""
import os
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))
random.seed(0)
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ---- dataset ----
all_pairs = [(a, b) for a in range(10) for b in range(10)]
random.shuffle(all_pairs)
test_pairs = all_pairs[:20]
train_pairs = all_pairs[20:]

def prompt(a, b):   return f"{a}+{b}="
def full(a, b):     return f"{a}+{b}={a+b}"

def encode_example(a, b):
    p_ids = tok(prompt(a, b))["input_ids"]          # includes <bos>
    f_ids = tok(full(a, b))["input_ids"] + [tok.eos_token_id]
    labels = list(f_ids)
    for i in range(len(p_ids)):                      # mask the prompt; learn only the answer
        labels[i] = -100
    return f_ids, labels

def build_batch(pairs):
    ex = [encode_example(a, b) for a, b in pairs]
    maxlen = max(len(x[0]) for x in ex)
    ids, lab, mask = [], [], []
    for f_ids, labels in ex:
        pad = maxlen - len(f_ids)
        ids.append(f_ids + [tok.pad_token_id] * pad)
        lab.append(labels + [-100] * pad)
        mask.append([1] * len(f_ids) + [0] * pad)
    return (torch.tensor(ids), torch.tensor(lab), torch.tensor(mask))

@torch.no_grad()
def accuracy(model, pairs):
    model.eval()
    correct = 0
    samples = []
    for a, b in pairs:
        inp = tok(prompt(a, b), return_tensors="pt")
        out = model.generate(**inp, max_new_tokens=4, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0], skip_special_tokens=True)
        pred = gen[len(prompt(a, b)):].strip().split()[0] if gen[len(prompt(a, b)):].strip() else ""
        ok = pred == str(a + b)
        correct += ok
        if len(samples) < 6:
            samples.append(f"{prompt(a,b)} -> {gen[len(prompt(a,b)):]!r} (want {a+b}) {'OK' if ok else 'x'}")
    return correct / len(pairs), samples

def main():
    model = AutoModelForCausalLM.from_pretrained(LOCAL_DIR, dtype=torch.float32)

    base_train_acc, _ = accuracy(model, train_pairs)
    base_test_acc, base_samples = accuracy(model, test_pairs)
    print("== BASE model ==")
    print(f"train acc={base_train_acc:.2%}  test acc={base_test_acc:.2%}")
    for s in base_samples:
        print("   ", s)

    ids, lab, mask = build_batch(train_pairs)
    ds = TensorDataset(ids, lab, mask)
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    EPOCHS = 400
    model.train()
    for ep in range(EPOCHS):
        tot = 0.0
        for bi, bl, bm in dl:
            opt.zero_grad()
            out = model(input_ids=bi, attention_mask=bm, labels=bl)
            out.loss.backward()
            opt.step()
            tot += out.loss.item()
        if (ep + 1) % 50 == 0:
            tr, _ = accuracy(model, train_pairs)
            te, _ = accuracy(model, test_pairs)
            model.train()
            print(f"epoch {ep+1:3d}  loss={tot/len(dl):.4f}  train_acc={tr:.2%}  test_acc={te:.2%}")

    ft_train_acc, _ = accuracy(model, train_pairs)
    ft_test_acc, ft_samples = accuracy(model, test_pairs)
    print("\n== FINE-TUNED model ==")
    print(f"train acc={ft_train_acc:.2%}  test acc={ft_test_acc:.2%}")
    for s in ft_samples:
        print("   ", s)

    print("\n== SUMMARY ==")
    print(f"train: {base_train_acc:.0%} -> {ft_train_acc:.0%}    test: {base_test_acc:.0%} -> {ft_test_acc:.0%}")


if __name__ == "__main__":
    main()
