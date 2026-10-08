"""
Glimmer-1-Base: load, inspect, and generate.
Step 1 of the empirical investigation — see what the model actually produces.
"""
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

MODEL_ID = "Glint-Research/Glimmer-1-Base"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "Glimmer-1-Base")
LOCAL_DIR = os.path.abspath(LOCAL_DIR)


def main():
    torch.manual_seed(0)
    print("== Downloading model (tiny, ~2.4MB) ==")
    snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_DIR)
    print("Local dir:", LOCAL_DIR)

    tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(LOCAL_DIR, dtype=torch.float32)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n== Model ==")
    print(f"Total parameters: {n_params:,}")
    cfg = model.config
    print(f"hidden_size={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
          f"vocab={cfg.vocab_size} ctx={cfg.max_position_embeddings}")

    print("\n== Tokenizer ==")
    print("vocab_size:", tok.vocab_size)
    print("special_tokens_map:", tok.special_tokens_map)
    sample = "The cat sat on the mat."
    ids = tok(sample)["input_ids"]
    print(f"encode {sample!r} -> {ids}")
    print("tokens:", tok.convert_ids_to_tokens(ids))
    print("decode back:", repr(tok.decode(ids)))

    print("\n== Generation samples ==")
    prompts = ["The", "Once upon a time", "The sun is", "Water is made of",
               "The capital of France is", "2 + 2 ="]
    for p in prompts:
        inp = tok(p, return_tensors="pt")
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=40, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            s = model.generate(**inp, max_new_tokens=40, do_sample=True,
                               temperature=0.8, top_k=50, pad_token_id=tok.pad_token_id)
        print(f"\nPROMPT: {p!r}")
        print("  greedy:", repr(tok.decode(g[0], skip_special_tokens=True)))
        print("  sample:", repr(tok.decode(s[0], skip_special_tokens=True)))


if __name__ == "__main__":
    main()
