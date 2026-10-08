"""
Cleaner capacity test: can fine-tuning teach Glimmer-1-Base a rule?
Task: c = (a + b) mod 10  -> single-digit answer (one token, no parsing ambiguity).
Primary metric: teacher-forced next-token accuracy at the answer position
  (argmax of the logits that predict the answer == gold answer token).
Secondary: free 1-token greedy generation accuracy.
Report BASE vs FINE-TUNED on a held-out test split.
"""
import os
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))
random.seed(0); torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def prompt(a, b): return f"{a}+{b}="
def full(a, b):   return f"{a}+{b}={(a + b) % 10}"

# sanity: confirm the answer is a single token and the prompt is a clean prefix
demo_full = tok(full(3, 4))["input_ids"]
demo_prompt = tok(prompt(3, 4))["input_ids"]
print("full   '3+4=7' ->", demo_full, tok.convert_ids_to_tokens(demo_full))
print("prompt '3+4='  ->", demo_prompt, tok.convert_ids_to_tokens(demo_prompt))
print("prompt is clean prefix of full:", demo_full[:len(demo_prompt)] == demo_prompt)

all_pairs = [(a, b) for a in range(10) for b in range(10)]
random.shuffle(all_pairs)
test_pairs, train_pairs = all_pairs[:20], all_pairs[20:]

def ans_token_id(a, b):
    return tok(str((a + b) % 10), add_special_tokens=False)["input_ids"][0]

@torch.no_grad()
def tf_accuracy(model, pairs):
    """teacher-forced: does argmax at answer position equal the gold answer token?"""
    model.eval(); correct = 0
    for a, b in pairs:
        f_ids = tok(full(a, b))["input_ids"]
        p_len = len(tok(prompt(a, b))["input_ids"])   # answer token sits at index p_len
        logits = model(torch.tensor([f_ids])).logits[0]
        pred = logits[p_len - 1].argmax().item()
        correct += (pred == f_ids[p_len])
    return correct / len(pairs)

@torch.no_grad()
def gen_accuracy(model, pairs, show=0):
    model.eval(); correct = 0; samples = []
    for a, b in pairs:
        inp = tok(prompt(a, b), return_tensors="pt")
        out = model.generate(**inp, max_new_tokens=1, do_sample=False, pad_token_id=tok.pad_token_id)
        pred_tok = out[0, -1].item()
        gold = ans_token_id(a, b)
        ok = pred_tok == gold
        correct += ok
        if len(samples) < show:
            samples.append(f"{prompt(a,b)} -> {tok.decode([pred_tok])!r} (want {(a+b)%10}) {'OK' if ok else 'x'}")
    return correct / len(pairs), samples

def main():
    model = AutoModelForCausalLM.from_pretrained(LOCAL_DIR, dtype=torch.float32)
    print(f"\n== BASE ==")
    print(f"teacher-forced  train={tf_accuracy(model, train_pairs):.2%}  test={tf_accuracy(model, test_pairs):.2%}")
    g, s = gen_accuracy(model, test_pairs, show=5)
    print(f"free-gen        test={g:.2%}")
    for x in s: print("   ", x)

    # build training batch (mask prompt; learn only the answer token)
    ex = []
    for a, b in train_pairs:
        f_ids = tok(full(a, b))["input_ids"]
        p_len = len(tok(prompt(a, b))["input_ids"])
        labels = [-100] * len(f_ids); labels[p_len] = f_ids[p_len]
        ex.append((f_ids, labels))
    maxlen = max(len(f) for f, _ in ex)
    ids = torch.tensor([f + [tok.pad_token_id] * (maxlen - len(f)) for f, _ in ex])
    lab = torch.tensor([l + [-100] * (maxlen - len(l)) for _, l in ex])
    msk = torch.tensor([[1] * len(f) + [0] * (maxlen - len(f)) for f, _ in ex])
    dl = DataLoader(TensorDataset(ids, lab, msk), batch_size=16, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    for ep in range(600):
        model.train(); tot = 0.0
        for bi, bl, bm in dl:
            opt.zero_grad()
            out = model(input_ids=bi, attention_mask=bm, labels=bl)
            out.loss.backward(); opt.step(); tot += out.loss.item()
        if (ep + 1) % 100 == 0:
            print(f"epoch {ep+1:3d} loss={tot/len(dl):.4f} "
                  f"tf_train={tf_accuracy(model, train_pairs):.2%} tf_test={tf_accuracy(model, test_pairs):.2%}")

    print(f"\n== FINE-TUNED ==")
    print(f"teacher-forced  train={tf_accuracy(model, train_pairs):.2%}  test={tf_accuracy(model, test_pairs):.2%}")
    g, s = gen_accuracy(model, test_pairs, show=5)
    print(f"free-gen        test={g:.2%}")
    for x in s: print("   ", x)

if __name__ == "__main__":
    main()
