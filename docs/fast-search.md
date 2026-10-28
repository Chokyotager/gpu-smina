# A faster search for GPU smina: `--fast`

`smina --gpu --fast` is an opt-in search that is not smina's. It combines
five changes, each measured against CPU smina on this machine (Ryzen 9
5950X; RTX 4080 SUPER + RTX 4070 Ti SUPER):

1. **QuickVina 2's history check** (ported from QuickVina 2.1): a chain skips
   a local optimization when its recent history says the starting point will
   fall back into a known minimum.
2. **Split chains:** every Monte Carlo chain becomes 4–32 shorter concurrent
   chains, chosen per batch so that a GPU's chains fit in 4 waves (32 for a
   single complex, 4–8 for screens).
3. **Half of smina's steps on screens** (split below 32). Single complexes
   keep the full budget, because their runs are mostly fixed overhead.
4. **Per-ligand early stopping** (screens): a ligand stops once half of its
   exhaustiveness units agree on its best energy.
5. **Two-pass screening** (20 or more ligands): every ligand is docked
   briefly, and only those ranked near the top get the full search
   (`--fast_top`, default 0.1; `--fast_top 1` turns it off).

## Results

| | CPU smina | `--gpu` (1:1) | `--gpu --fast` |
|---|---|---|---|
| 1HSG redock, exhaustiveness 8 | 43 s | 26 s | 1.2 s |
| 1HSG, exhaustiveness 128 | 173 s | 47 s | 5.7 s |
| 25 Astex complexes, per complex | | 16.8 s | 1.5 s |
| 210-ligand screen, 1 GPU / 2 GPUs | 236 s (32 processes) | 32 / 19.5 s | 4.5 / 4.8 s |
| 840-ligand screen, 1 GPU / 2 GPUs | | 57 / 47 s | 13.3 / 8.8 s |

How the screen times came about (840 ligands, one GPU / two GPUs):

| `--fast` with | time |
|---|---|
| components 1–3 | 28.6 / 16.9 s |
| + early stopping (`--fast_top 1`) | 19.4 / 12.1 s |
| + two-pass screening (default) | 13.3 / 8.8 s |

Redocking quality (Astex set, 25 complexes × 5 seeds, success = top pose
< 2 Å; p-values from permutation tests within complexes):

| | success | mean top score (kcal/mol) |
|---|---|---|
| CPU smina | 92/125 | −9.835 |
| `--gpu` (1:1) | 90/125 (p = 0.60) | −9.818 (p = 0.51) |
| `--gpu --fast`, single complex | 93/125 (p = 1.00) | −9.838 (p = 0.79) |
| `--fast` screening form: split 8, half the steps, early stopping | 92/125 (p = 1.00) | −9.817 (p = 0.16) |

Screening quality (210 ZINC ligands, 1HSG box). With every ligand fully
searched (`--fast_top 1`), `--fast` top scores differ from the CPU's by
s.d. 0.085 (one GPU) and 0.088 kcal/mol (two GPUs); two CPU runs with
different seeds differ by 0.073. With two-pass screening, the overlap of the
top k with the CPU's top k is:

| k = 10 / 21 / 42, summed over seeds 1–4 | overlap with CPU smina |
|---|---|
| `--fast`, two-pass (default) | 34/40, 76/84, 165/168 |
| `--fast --fast_top 1` | 36/40, 77/84, 165/168 |
| CPU smina, another seed (one run) | 9/10, 19/21, 42/42 |

No ligand of the CPU's top 10% scores more than 0.5 kcal/mol worse. The
ligands outside the top keep their first-pass scores: over all ligands the
s.d. against the CPU is 0.11–0.14 kcal/mol.

## How much search does smina need?

Traces of 125 full 1:1 redocking runs, giving each ligand's best energy over
its 8 chains at each tenth of the steps:

| budget | 10% | 20% | 30% | 50% | 80% |
|---|---|---|---|---|---|
| ligand best within 0.1 kcal/mol of the full search | 70% | 81% | 87% | 93% | 97% |
| … within 0.5 kcal/mol | 96% | 99% | 99% | 99% | 100% |

Most of smina's budget goes to a few hard cases. Cutting the budget
uniformly, though, eventually loses those cases (redocking, split 16):

| budget | 1 | ½ | ¼ | ⅛ | 1/16 | 1/64 |
|---|---|---|---|---|---|---|
| success | 91 | 89 | 89 | 84 | 85 | 76 |
| Δ mean top score | −0.008 | +0.002 | +0.023 | +0.095 | +0.157 | +0.466 |

From ⅛ down, the losses are significant (p ≤ 0.06 for success, p < 0.01
for scores). Hence two adaptive cuts instead of a uniform one: early
stopping spends the budget where chains still disagree, and two-pass
screening spends it on the ligands whose rank matters.

## Early stopping

The rule was chosen on traces of the `--fast` runs, simulated offline: after
each tenth of the chains' steps, a ligand stops when the best energies of at
least half of its exhaustiveness units (the chains of one smina task; at
least 2) are within ε of its best. Checking from the second tenth on:

| rule (screen: split 4, 210 ligands) | steps used | best within 0.1 of the full search | worst gap |
|---|---|---|---|
| fixed 50% (components 1–3) | 50% | 99.0% | 0.23 kcal/mol |
| half the units within 0.05, cap 50% | 28% | 99.0% | 0.23 kcal/mol |
| half the units within 0.05, cap 100% | 34% | 99.5% | 0.14 kcal/mol |

The chains run in phases ending at the checkpoints; between phases the
host reads every chain's best energy and drops the chains of stopped
ligands, and the remaining chains resume exactly where they stopped, so
results are reproducible. The full-budget cap would search better for the
same work, but it was slower: the few hard ligands left after the first
checks keep the GPU mostly idle for the rest of their longer chains (210
ligands: 9.1 s against 6.7 s with the 50% cap). For a single complex the GPU
is idle anyway and the checkpoints only add launches (1HSG: 1.38 s instead
of 1.23 s), so early stopping applies only below split 32.

## Two-pass screening

A screen ranks ligands, and only the top of the ranking is acted on. So a
screen of 20 or more ligands first docks every ligand of a batch with 1/16
of smina's steps (`--fast_first`), scores each one as the final output
would, and gives the normal `--fast` search only to the ligands within 0.3
kcal/mol (`--fast_top_margin`) of the best 10% (`--fast_top`) of all
first-pass scores so far: about 20% of the ligands. The others keep their
first-pass results. Simulated on finished runs first, first passes of 1/32 to
1/8 of the steps all kept the top k about as well as one pass; 1/16 was
chosen.

## Tried and rejected

| idea | result |
|---|---|
| Uni-Dock-like budgets (≈1–8% of smina's steps for every ligand) | 7–16 fewer redocking successes, +0.16–0.47 kcal/mol |
| Half the steps with split 1 | screen top scores drift (s.d. 0.124, one ligand > 1 kcal/mol worse) |
| Giving up a candidate's local optimization once its energy is above the Metropolis threshold drawn in advance | local optimizations descend steeply and 79% of a screen's candidates are accepted: the safest rule saved 7% of iterations and no measurable time (840 ligands: 19.3–20.3 s against 19.3–19.8 s); aggressive rules lost accuracy (s.d. 0.14–0.20) |
| Vina-GPU 2.1's RILC-BFGS (one-pair L-BFGS) | optimizer bookkeeping is ~18% of the search time; at most ~10% to gain; not ported |
| QuickVina-W's global history of minima | aimed at blind docking; not ported |

## Published strategies

| tool | what it changes | here |
|---|---|---|
| QuickVina 2 (Alhossary et al. 2015) | first-order consistency check on a 5n-point history; skip BFGS when no nearby point is consistent | ported; ~2× alone (Astex 16.8 → 8.5 s, same quality) |
| Uni-Dock (2023) | many short chains: fast 128 × 20, balance 384 × 40, detail 512 × 40 Monte Carlo steps | equivalent budgets lose accuracy (above) |
| Vina-GPU 2.1 (2023) | 5000 chains × 1 step; RILC-BFGS | as above |
| AutoDock-GPU (2021) | autostop: stop when the top scores' s.d. falls below 0.15 kcal/mol | the idea behind early stopping; the agreement rule did better on our traces than stall rules |
| Hierarchical screening (e.g. Glide HTVS → SP) | cheap pass for all, thorough pass for the best | the idea behind two-pass screening |

## Caveats

- **Validation scope:** validated with rigid receptors, the default scoring
  function, the 25 Astex complexes and one screen (210 ZINC ligands, 1HSG
  box). Flexible residues and custom scoring run with `--fast` but haven't
  been validated.
- **Screens report rough scores outside the top:** with two-pass screening,
  ligands outside the top 10% (plus margin) keep first-pass results. Use
  `--fast_top 1` when every ligand's score matters (e.g. as training data).
- **DEKOIS enrichment:** not measured yet (EF1%, pROC-AUC, BEDROC). The top-k
  overlap above comes from a single screen of 210 ligands; enrichment on
  actives and decoys is the test two-pass screening most needs.
- **Reproducibility:** results are reproducible for the same input and GPUs.
  They differ, statistically equivalently, from `--gpu` and between batch
  sizes or GPU sets, because the split, threads per chain, early stopping and
  the two-pass threshold depend on how the ligands are batched.

## Reproducing

- **Components:** `--gpu_split N` and the hidden `--gpu_qvina`,
  `--gpu_budget F`, `--gpu_early_stop`, `--fast_first F` and
  `--fast_top_margin D` set them individually; `GPU_EARLY_STOP="fraction,
  eps,first,checkpoints"` tests other stopping rules.
- **Tracing:** `GPU_TRACE_STEPS=file` records per-chain convergence;
  `GPU_CHAIN_STATS=1` prints every phase.
- **Benchmarks:** `tools/redock_benchmark.py --gpu-only --gpu-args "..."
  --label name` runs the redocking set; `tools/compare_redock.py` and
  `tools/compare_screen.py` give the comparisons above.
