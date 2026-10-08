# grokking-microscope

Reproducibility code and per-seed records for the paper

> **Grokking Is Conditional and Fragile: A Fully-Tractable, Multi-Seed Study at 12K Parameters**
> ([arXiv:2607.05104](https://arxiv.org/abs/2607.05104))

We treat an ~11,856-parameter transformer as a *microscope*: a model small enough to
enumerate and read its weights, attention, and full input–output map directly, used to ask
what actually governs grokking when nothing is hidden and every claim is a multi-seed **rate**
rather than a single run.

Related (different research question, same enumerable-world methodology):
[otanl/microground](https://github.com/otanl/microground) asks how the *input pathway* shapes
compositional binding; this repository asks what governs *grokking* and how fragile single-run
grokking claims are.

---

## Model (third-party — *not* a contribution of this work)

This paper **studies a publicly released model that we did not create or train**:

**Glimmer-1-Base** — by CompactAI / Glint Research
<https://huggingface.co/Glint-Research/Glimmer-1-Base> (MIT license)

An ~11,856-parameter (≈11.9K) Llama-style decoder: hidden size 16, 2 layers, 4 query heads /
1 KV head (GQA), SiLU MLP width 24, RMSNorm, RoPE, tied embeddings, vocab 512, context 512,
byte-level BPE, pretrained on 500K tokens of FineWeb-Edu (base only, no SFT). The scripts
fetch it at runtime via `huggingface_hub.snapshot_download`; **the weights are not committed
to this repository.** If you use the model, please credit its authors (see
[Crediting the model](#crediting-the-model-third-party)).

---

## What the paper shows (abstract in brief)

Grokking is the delayed onset of generalization long after a network has fit its training set.
Measuring it as a multi-seed rate at 12K parameters, six findings emerge:

1. **Coverage gates the transition.** The grok threshold is governed *primarily by output
   cardinality* (the modulus) and only *secondarily by composition structure* — an ordering
   that replicates across a ten-fold change in domain size.
2. **Weight decay reproduces the Omnigrok inverted-U** at 12K params (grok-rate 20% → 90% →
   0%) — a positive control on the rate measurement.
3. **Grokking sits on a numerical knife-edge.** Two distinct floating-point perturbations —
   CPU thread count (a pure reduction-order change) and CPU-vs-GPU execution — each flip a
   *minority* of seeds (≈5/30) with **no detectable change in the aggregate rate**.
4. **The mechanism is mixed.** Generalizing solutions have a more periodic output map (a
   partly definitional consistency check); the genuinely independent result is a **negative**
   one — the dim-16 model does **not** form the textbook Fourier embedding circle.
5. **Decomposition helps by coverage, not capability.** At a matched data budget a two-
   specialist pipeline groks where the monolith cannot (**10/10 vs 0/10**); a same-budget
   scratchpad monolith carrying identical decomposed supervision still fails (0/10),
   isolating *coverage* rather than supervision density as the driver.
6. **Multi-seed control overturns three single-run narratives** in our own data — a hard task
   "wall", a "thread count flips grokking" effect, and a "GPU suppresses grokking" effect —
   each a seed confound.

---

## Setup

```bash
# Python 3.12.9
python -m venv .venv && . .venv/Scripts/activate        # Windows
pip install -r requirements.txt
```

`requirements.txt` pins the exact environment (torch 2.12.1, transformers 5.12.1,
huggingface_hub 1.20.1, numpy 2.4.6, matplotlib 3.11.0).

- **Run on CPU.** At ~12K parameters CPU is *faster* than a GPU (kernel-launch overhead
  dominates); only the CPU-vs-GPU device-control leg (`scripts/26`) needs CUDA. To reproduce
  the exact GPU leg, install the CUDA 12.6 build:
  `pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126`.
- **Multi-seed sweeps** run seeds as parallel processes with `torch.set_num_threads(1)` and
  `OMP_NUM_THREADS=1` (tiny matmuls do not benefit from intra-op threads). The `run_*.ps1`
  launchers in `scripts/` do this.
- **Windows console:** set `PYTHONUTF8=1` (byte-BPE tokens contain non-cp932 characters).

---

## Repository layout

```
scripts/        all experiment, aggregation, figure, and statistics code (run order in REPRODUCE.md)
results/        per-seed JSON records + aggregated .txt tables — the released evidence
paper/          LaTeX source (main.tex, main.bib) + compiled PDF + figures
figures/        generated figures (PDF/PNG), copied into paper/figs/
requirements.txt, LICENSE, REPRODUCE.md
```

Not tracked (see `.gitignore`): `.venv/`, `models/` (fetched from HF), run logs
(`results/*.log`, `*.err`), and LaTeX build artifacts.

---

## Reproducing the results

See **[REPRODUCE.md](REPRODUCE.md)** for a claim-by-claim map (paper section → script →
output) and the exact commands. Short version:

```powershell
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe scripts\<name>.py | Tee-Object results\<name>.txt
```

Because grokking here is a **seed- and numerical-environment-sensitive knife-edge**, single
runs are not reproducible point-for-point across machines; the *rates* are. We therefore
release every per-seed JSON in `results/` so the aggregate statistics can be recomputed
exactly without retraining.

---

## License

- **This repository's code** (`scripts/`, analysis, figures): MIT — see [LICENSE](LICENSE).
- **The model** (Glimmer-1-Base) is the property of its authors (CompactAI / Glint Research)
  and is distributed by them under MIT on HuggingFace. It is **not** redistributed here.

## Citing this work

```bibtex
@article{ootani2026grokking,
  author  = {Ootani, Yoshiyuki},
  title   = {Grokking Is Conditional and Fragile: A Fully-Tractable, Multi-Seed Study at 12K Parameters},
  journal = {arXiv preprint arXiv:2607.05104},
  year    = {2026}
}
```

## Crediting the model (third-party)

This model is **not** a contribution of this work; we neither created nor trained it. If your
work uses **Glimmer-1-Base**, please cite its original authors (CompactAI / Glint Research):

```bibtex
@misc{glimmer1base2026,
  author    = {CompactAI},
  title     = {Glimmer-1: An 11.9K-Parameter Llama-Style Transformer},
  year      = {2026},
  publisher = {Glint Research},
  url       = {https://huggingface.co/Glint-Research}
}
```
