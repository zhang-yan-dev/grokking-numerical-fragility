"""
Glimmer-1-Base: is it "completely random"?
Tests whether the model learned ANY structure by comparing its loss on:
  (a) real English text   (b) token-shuffled text   (c) uniformly random tokens
A model that learned structure assigns LOWER loss to real text than to random.
Also: mean next-token entropy vs the uniform ceiling, and minimal-pair grammar.
"""
import os
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base"))


def seq_loss(model, ids):
    """Mean per-token cross-entropy (nats) for a 1D LongTensor of token ids."""
    ids = ids.unsqueeze(0)
    with torch.no_grad():
        out = model(ids, labels=ids)
    return out.loss.item()


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_DIR, dtype=torch.float32)
    model.eval()
    V = model.config.vocab_size
    uniform_bpt = math.log2(V)  # bits/token if predictions were uniform
    print(f"vocab={V}  uniform ceiling = {uniform_bpt:.3f} bits/token "
          f"(loss {math.log(V):.3f} nats)\n")

    real_text = (
        "The sun rose over the quiet village as the children walked to school. "
        "Water is made of hydrogen and oxygen. Many animals sleep during the day "
        "and hunt at night. Reading books helps people learn new ideas."
    )
    real_ids = torch.tensor(tok(real_text)["input_ids"])
    n = real_ids.numel()

    # (b) shuffled real tokens
    perm = torch.randperm(n)
    shuf_ids = real_ids[perm]
    # (c) uniform random tokens
    rand_ids = torch.randint(0, V, (n,))

    print("== Loss comparison (lower = model finds it more predictable) ==")
    for name, ids in [("real English", real_ids), ("shuffled tokens", shuf_ids),
                      ("uniform random", rand_ids)]:
        loss = seq_loss(model, ids)
        print(f"  {name:16s}: loss={loss:.4f} nats  ppl={math.exp(loss):,.1f}  "
              f"bits/token={loss/math.log(2):.3f}")
    print(f"  {'(uniform model)':16s}: loss={math.log(V):.4f} nats  ppl={V:,.1f}  "
          f"bits/token={uniform_bpt:.3f}")

    print("\n== Mean next-token entropy on real text ==")
    with torch.no_grad():
        logits = model(real_ids.unsqueeze(0)).logits[0]
    probs = F.softmax(logits, dim=-1)
    ent_bits = -(probs * torch.log2(probs + 1e-12)).sum(-1)
    print(f"  mean entropy = {ent_bits.mean():.3f} bits  (uniform = {uniform_bpt:.3f}); "
          f"{'CONCENTRATED -> learned structure' if ent_bits.mean() < uniform_bpt - 0.5 else 'near-uniform'}")

    print("\n== Minimal-pair grammar (lower loss should go to the grammatical one) ==")
    pairs = [
        ("The keys are on the table.", "The keys is on the table."),
        ("She has eaten her dinner.", "She have eaten her dinner."),
        ("The children are playing outside.", "The children is playing outside."),
        ("He does not know the answer.", "He do not knows the answer."),
    ]
    correct = 0
    for good, bad in pairs:
        lg = seq_loss(model, torch.tensor(tok(good)["input_ids"]))
        lb = seq_loss(model, torch.tensor(tok(bad)["input_ids"]))
        ok = lg < lb
        correct += ok
        print(f"  [{'OK ' if ok else 'MISS'}] good={lg:.3f} bad={lb:.3f}  | {good!r}")
    print(f"  grammar preference: {correct}/{len(pairs)}")


if __name__ == "__main__":
    main()
