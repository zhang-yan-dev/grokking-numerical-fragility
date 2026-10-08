# Reproduction map

Every reported number is reproducible from the scripts below and the per-seed records in
`results/`. The model (`Glint-Research/Glimmer-1-Base`) is fetched automatically from
HuggingFace on first run. Grok is defined as held-out accuracy ≥ τ = 0.70; a **grok-rate** is
the fraction of seeds that grok within the epoch budget.

Run pattern (PowerShell, Windows):

```powershell
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe scripts\<name>.py | Tee-Object results\<name>.txt
```

Multi-seed sweeps launch one process per seed (`torch.set_num_threads(1)`,
`OMP_NUM_THREADS=1`). The `run_*.ps1` launchers do this for the grids.

---

## Claim → script → output

| Paper location | Claim | Script(s) (launcher) | Output |
|---|---|---|---|
| Intro; base probes | Base model generates incoherently; "not random" only at the unigram level (~0.6 bit/token over uniform), ~zero word order | `01_inspect_and_generate`, `02_randomness_probe` | `results/01_generate.txt`, `02_randomness.txt` |
| §2 fine-tune sanity | `(a+b) mod 10`: base 0% → ~90% held-out, grokking-style phase transition | `03c_finetune_modadd_fixed` | `results/03c_finetune_modadd_fixed.txt` |
| Capability frontier; seed sensitivity | Per-task multi-seed grok-rates (the trustworthy view; single-seed frontiers are lucky draws) | `14_grok_probability <seed>` (parallel) → `15_aggregate_grok` | `results/14_seed*.json` |
| §coverage — finding (i) | Monolith grok-rate **rises with training coverage**; sharp 0→full threshold | `16_coverage_sweep`, `18_generality` → fig F1 | `results/16_*`, `18_*` |
| finding (i), cardinality | Threshold tracks **output cardinality (M)** primarily, **structure** secondarily; replicates at domain 10³ (3-input) and in a 2×2 dissociation | `33_cardinality_3input`, `18_generality` → `35_strengthen_aggregate` | `results/33_*`, `35_*` |
| §decomp mechanism | Pipeline advantage comes from specialists **memorizing dense sub-domains**, not generalizing | `17_memorization_mechanism` | `results/17_*` |
| §wd — finding (ii) | Weight-decay **Omnigrok inverted-U** (grok-rate 20%→90%→0%) | `22_numerical_fragility` (`run_22_wdgrid.ps1`) → `23_aggregate_fragility` → fig F2 | `results/22_*`, `23_*` |
| §knife-edge — finding (iii) | **Thread count** (pure float reduction-order change) flips a minority of seeds, no aggregate bias; t=4 ≡ t=16 **bit-identical** | `22_numerical_fragility` (`run_22_threadgrid.ps1`) → `23`; `34_thread_bitident` → fig F3 | `results/22_*`, `34_*` |
| §knife-edge — finding (iii) | **CPU-vs-GPU** device flips a minority of seeds (paired, same seed/data/init) | `26_device_control` (`run_26_device.ps1`) → fig F3 | `results/26_*` |
| §mechanism — finding (iv) | Generalizers have a more periodic **logit** map; **negative result**: no Fourier **embedding** circle at dim 16 | `20_mechanism_fourier` → `21_aggregate_mechanism`; `36_review2_stats` (Fisher-z, Spearman) → fig F4 | `results/20_fourier_m*.json`, `21_*` |
| §decomp — finding (v) | The single-agent "wall" is robust to weight decay; at matched budget **pipeline 10/10 vs monolith 0/10**; same-budget **scratchpad monolith 0/10** isolates coverage (not supervision density) | `24_wall_vs_wd` (`run_24_wall.ps1`), `25_decomp_vs_wd` (`run_25_decomp.ps1`), `37_decomp_scratchpad` → fig F5 | `results/24_*`, `25_*`, `37_*` |
| Fig 7 trajectory | Train-first / test-delayed signature; transition epoch ≈320–760; held-out loss plateaus at ln(10)≈2.30 then collapses | `38_trajectory` → fig F7 | `results/38_*` |
| §discussion — finding (vi) | Story invariant to grok threshold τ∈{0.6,0.7,0.8}; paired stats: exact McNemar, Newcombe (1998) paired-difference CI, Fisher-z, Spearman | `32_review_reanalysis`, `36_review2_stats` | `results/32_*`, `36_*` |
| All figures | F1 coverage, F2 weight decay, F3 paired flip scatter, F4 logit-Fourier, F5 pipeline vs monolith, F6/F7 | `30_figures` | `figures/fig*.pdf` → copy to `paper/figs/` |

---

## Aggregators and launchers

- **Aggregators** (pool the per-seed JSONs into paper tables): `15`, `21`, `23`, `35`.
- **Statistics** (zero-cost re-analysis from stored `best` accuracies, no retraining): `32`, `36`.
- **Grid launchers** (`scripts/*.ps1`): `run_22_threadgrid`, `run_22_wdgrid`, `run_22_phase2`,
  `run_24_wall`, `run_25_decomp`, `run_26_device`, `run_2x2_n2000`, `run_strengthen`.

## Superseded scripts (kept for transparency)

These are **not** part of the final results; they are retained to document evaluation pitfalls
discussed in the paper / `CLAUDE.md`:

- `03`, `03b` — teacher-forced accuracy inflated by the tokenizer auto-appending `<eos>`
  (off-by-one at the answer position). Fixed in `03c`.
- `10`, `11` — early multi-agent / capacity probes superseded by `13`–`18`.
- `12` — early-stopped on train saturation, cutting off the grokking transition; superseded by
  `13`. Use `13`+ only.

## Figures

```powershell
$env:PYTHONUTF8=1; .\.venv\Scripts\python.exe scripts\30_figures.py
# then copy figures\fig*.pdf -> paper\figs\
```
