"""
M0 — division of labor vs the single-agent capacity ceiling.

Each agent is an independent Glimmer-1 instance (~11,856 params), fine-tuned for a
narrow role. We compare a MONOLITHIC agent against a PIPELINE of specialists.

Part 1 (pure addition):  y = (a+b+c) mod 10
    monolith: "a+b+c="                vs   A:(a+b) -> B:(·+c)
Part 2 (mixed ops = real division of labor):  y = (a*b+c) mod 10
    monolith: "a*b+c="                vs   A multiplier:"a*b="  ->  B adder:"·+c="

Answers are single digits (one token). Metric: exact-match accuracy on a held-out
test split of the input space. Composition is emergent at inference — the pipeline
agents never see the composite task.
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

def make_agent():
    return AutoModelForCausalLM.from_pretrained(LOCAL_DIR, dtype=torch.float32)

def train_agent(examples, epochs=800, lr=3e-3, bs=128, tag=""):
    """examples: list of (prompt_str, answer_str). Learns only the first answer token."""
    model = make_agent()
    rows = []
    for p, ans in examples:
        f = [BOS] + enc(p + ans) + [EOS]
        pl = len([BOS] + enc(p))
        lab = [-100] * len(f); lab[pl] = f[pl]
        rows.append((f, lab))
    ml = max(len(f) for f, _ in rows)
    ids = torch.tensor([f + [PAD] * (ml - len(f)) for f, _ in rows])
    lab = torch.tensor([l + [-100] * (ml - len(l)) for _, l in rows])
    msk = torch.tensor([[1] * len(f) + [0] * (ml - len(f)) for f, _ in rows])
    dl = DataLoader(TensorDataset(ids, lab, msk), batch_size=bs, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        for bi, bl, bm in dl:
            opt.zero_grad()
            out = model(input_ids=bi, attention_mask=bm, labels=bl)
            out.loss.backward(); opt.step()
    return model

@torch.no_grad()
def predict_digit(model, prompt_str):
    model.eval()
    ids = torch.tensor([[BOS] + enc(prompt_str)])
    out = model.generate(ids, max_new_tokens=1, do_sample=False, pad_token_id=PAD)
    return tok.decode([out[0, -1].item()]).strip()

@torch.no_grad()
def acc(model, examples):
    return sum(predict_digit(model, p) == ans for p, ans in examples) / len(examples)

def split(space, n_test):
    random.shuffle(space)
    return space[n_test:], space[:n_test]

# ---------------- Part 1: (a+b+c) mod 10 ----------------
def part1():
    print("\n##### Part 1:  y = (a+b+c) mod 10 #####")
    triples = [(a, b, c) for a in range(10) for b in range(10) for c in range(10)]
    train, test = split(triples, 200)

    mono_train = [(f"{a}+{b}+{c}=", str((a + b + c) % 10)) for a, b, c in train]
    mono_test = [(f"{a}+{b}+{c}=", str((a + b + c) % 10)) for a, b, c in test]
    mono = train_agent(mono_train, epochs=1000)
    print(f"[monolith]  test acc = {acc(mono, mono_test):.1%}")

    # specialist adder (same module reused as A and B): "x+y=" -> (x+y)%10
    add_ex = [(f"{x}+{y}=", str((x + y) % 10)) for x in range(10) for y in range(10)]
    adder = train_agent(add_ex, epochs=800)
    print(f"[adder solo] test acc = {acc(adder, add_ex):.1%} (on its own task)")

    correct = 0
    for a, b, c in test:
        u = predict_digit(adder, f"{a}+{b}=")
        if not u.isdigit():
            continue
        v = predict_digit(adder, f"{u}+{c}=")
        correct += (v == str((a + b + c) % 10))
    print(f"[pipeline A->B] test acc = {correct/len(test):.1%}")

# ---------------- Part 2: (a*b+c) mod 10 ----------------
def part2():
    print("\n##### Part 2:  y = (a*b+c) mod 10  (mixed ops) #####")
    print("token check '3*4=':", tok.convert_ids_to_tokens([BOS] + enc("3*4=")))
    triples = [(a, b, c) for a in range(10) for b in range(10) for c in range(10)]
    train, test = split(triples, 200)

    mono_train = [(f"{a}*{b}+{c}=", str((a * b + c) % 10)) for a, b, c in train]
    mono_test = [(f"{a}*{b}+{c}=", str((a * b + c) % 10)) for a, b, c in test]
    mono = train_agent(mono_train, epochs=1200)
    print(f"[monolith]  test acc = {acc(mono, mono_test):.1%}")

    mul_ex = [(f"{x}*{y}=", str((x * y) % 10)) for x in range(10) for y in range(10)]
    add_ex = [(f"{x}+{y}=", str((x + y) % 10)) for x in range(10) for y in range(10)]
    multiplier = train_agent(mul_ex, epochs=1200)
    adder = train_agent(add_ex, epochs=800)
    print(f"[multiplier solo] acc = {acc(multiplier, mul_ex):.1%}")
    print(f"[adder solo]      acc = {acc(adder, add_ex):.1%}")

    correct = 0
    for a, b, c in test:
        p = predict_digit(multiplier, f"{a}*{b}=")
        if not p.isdigit():
            continue
        v = predict_digit(adder, f"{p}+{c}=")
        correct += (v == str((a * b + c) % 10))
    print(f"[pipeline mult->add] test acc = {correct/len(test):.1%}")

if __name__ == "__main__":
    part1()
    part2()
    print("\nDONE")
