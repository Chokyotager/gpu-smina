# A faster search for GPU smina: `--fast`

`smina --gpu --fast` is an opt-in search that is not smina's. It combines
three changes, each measured against CPU smina on this machine (Ryzen 9
5950X; RTX 4080 SUPER + RTX 4070 Ti SUPER):

1. **QuickVina 2's history check** (ported from QuickVina 2.1): a chain skips
   a local optimization when its recent history says the starting point will
   fall back into a known minimum.
2. **Split chains:** every Monte Carlo chain becomes 4–32 shorter concurrent
   chains, chosen per batch so that a GPU's chains fit in 4 waves (32 for a
   single complex, 4–8 for screens).
3. **Half of smina's steps on screens** (split below 32). Single complexes
   keep the full budget, because their runs are mostly fixed overhead.

## Results

| | CPU smina | `--gpu` (1:1) | `--gpu --fast` |
|---|---|---|---|
| 1HSG redock, exhaustiveness 8 | 43 s | 26 s | 1.2 s |
| 1HSG, exhaustiveness 128 | 173 s | 47 s | 5.7 s |
| 25 Astex complexes, per complex | | 16.8 s | 1.5 s |
| 210-ligand screen, 1 GPU / 2 GPUs | 236 s (32 processes) | 32 / 19.5 s | 7.9 / 5.4 s |
| 840-ligand screen, 1 GPU / 2 GPUs | | 57 / 47 s | 28.6 / 16.9 s |

Quality (Astex set, 25 complexes × 5 seeds, success = top pose < 2 Å;
p-values from permutation tests within complexes):

| | success | mean top score (kcal/mol) |
|---|---|---|
| CPU smina | 92/125 | −9.835 |
| `--gpu` (1:1) | 91/125 (p = 1.00) | −9.816 (p = 0.43) |
| `--gpu --fast`, single complex | 93/125 (p = 1.00) | −9.838 (p = 0.79) |
| `--fast` screening form, split 8 | 90/125 (p = 0.62) | −9.825 (p = 0.39) |

On the 210-ligand screen, `--fast` top scores differ from the CPU's by s.d.
0.081–0.083 kcal/mol (means within 0.002). Two CPU runs with different
seeds differ by 0.073.

## How much search does smina need?

Traces of 125 full 1:1 redocking runs, giving each ligand's best energy over
its 8 chains at each tenth of the steps:

| budget | 10% | 20% | 30% | 50% | 80% |
|---|---|---|---|---|---|
| ligand best within 0.1 kcal/mol of the full search | 70% | 81% | 87% | 93% | 97% |
| … within 0.5 kcal/mol | 96% | 99% | 99% | 99% | 100% |

Half of the chains make their last improvement larger than 0.1 kcal/mol
within the first 10% of their steps. Most of smina's budget goes to a few
hard cases. Cutting the budget uniformly, though, eventually loses those
cases (redocking, split 16):

| budget | 1 | ½ | ¼ | ⅛ | 1/16 | 1/64 |
|---|---|---|---|---|---|---|
| success | 91 | 89 | 89 | 84 | 85 | 76 |
| Δ mean top score | −0.008 | +0.002 | +0.023 | +0.095 | +0.157 | +0.466 |

From ⅛ down, the losses are significant (p ≤ 0.06 for success, p < 0.01
for scores).

## Published strategies

| tool | what it changes | here |
|---|---|---|
| QuickVina 2 (Alhossary et al. 2015) | first-order consistency check on a 5n-point history; skip BFGS when no nearby point is consistent | ported; ~2× alone (Astex 16.8 → 8.5 s, same quality) |
| QuickVina-W (2017) | QuickVina 2 plus a global octree of minima shared by all chains, for blind docking | not ported (box docking; cross-chain sharing) |
| Uni-Dock (2023) | many short chains: fast 128 × 20, balance 384 × 40, detail 512 × 40 Monte Carlo steps (≈1–8% of smina's) | equivalents lose 7–16 successes and 0.16–0.47 kcal/mol (384 chains at 6%: 85/125, +0.235) |
| Vina-GPU 2.1 (2023) | 5000 chains × 1 step by default; RILC-BFGS (L-BFGS with one correction pair, Lewis–Overton line search) | budgets this small fail as above; RILC-BFGS saves optimizer bookkeeping, ~10% of our time; not ported |
| AutoDock-GPU (2021) | autostop: stop when the top scores' s.d. falls below 0.15 kcal/mol | simulated on the traces, see below; not implemented |

**Early stopping, simulated.** Stopping a ligand once m of its 8 chains are
within ε of its best beats a fixed budget on the traces:

| rule | budget used | final best within 0.1 | worst gap |
|---|---|---|---|
| fixed 40% | 40% | 90% | 1.38 kcal/mol |
| 4 chains within 0.1 kcal/mol | 40% | 98% | 0.16 kcal/mol |

Rules based on stalled improvement do no better than a fixed cut.
Implementing the agreement rule needs cross-chain checks during the search;
the resumable chains support that, but the phase barriers would cost
screens part of the gain.

## Other findings along the way

- **The split must adapt to the batch.** It pays off while a GPU's chains
  fit in a few waves and hurts beyond that, because shorter chains need
  more evaluations: 19% more at split 8. Measured on the 840-ligand screen
  on one GPU: split 1 with the check takes 48.9 s, split 4 51.9 s.
- **Half the steps need a split of at least 4.** With split 1, screen top
  scores drift (s.d. 0.124, one ligand more than 1 kcal/mol worse).
- **Host work in parallel:** finishing screened ligands (final scoring,
  clustering) now runs on all host threads, with output unchanged.

## Caveats

- **Validation scope:** validated with rigid receptors, the default scoring
  function, the 25 Astex complexes and one screen (210 ZINC ligands, 1HSG box).
  Flexible residues and custom scoring run with `--fast` but haven't been
  validated.
- **DEKOIS enrichment:** not measured yet (EF1%, pROC-AUC, BEDROC). Redocking
  success and top scores don't guarantee unchanged enrichment.
- **Reproducibility:** results are reproducible for the same input and GPUs.
  They differ, statistically equivalently, from `--gpu` and between batch
  sizes or GPU sets, because the split and threads per chain depend on how
  many chains share a GPU.

## Reproducing

- **Hidden switches:** `--gpu_qvina`, `--gpu_split N` and `--gpu_budget F`
  set the components individually.
- **Tracing:** `GPU_TRACE_STEPS=file` records per-chain convergence.
- **Benchmarks:** `tools/redock_benchmark.py --gpu-only --gpu-args "..."
  --label name` runs the redocking set; `tools/compare_redock.py` and
  `tools/compare_screen.py` give the comparisons above.
