# smina on the GPU

`smina --gpu` runs smina's search on NVIDIA GPUs. It is a 1:1 port: the
Monte Carlo search, the BFGS minimizer and its line searches, mutation,
Metropolis acceptance, output clustering, the precalculated pair tables
(linear and spline), the grid maps, the direct receptor evaluation and the
step counts, seeding and parameters are those of CPU smina. Without `--gpu`
the program behaves exactly as before.

`--fast` additionally offers a GPU-style search that is *not* smina's: it
skips local optimizations that QuickVina 2's history check deems redundant,
splits every Monte Carlo chain into several shorter concurrent ones and, on
screens, runs half of smina's steps. It docks a single complex 10-20 times
faster than the 1:1 port and screens 2-4 times faster, with the same
redocking success and screening scores as CPU smina (see below and
[docs/fast-search.md](docs/fast-search.md)).

## Building

Dependencies: Boost (≥ 1.70), OpenBabel (2.x or 3.x), CMake ≥ 3.18 and the
CUDA toolkit (developed with 12.6, sm_89).

```
cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=<boost/openbabel prefix> \
      -DCMAKE_CUDA_HOST_COMPILER=g++-13 [-DCMAKE_CUDA_ARCHITECTURES=89]
ninja -C build smina
```

`-DSMINA_GPU=OFF` builds the CPU-only program. CUDA 12.6 installs a
`bin/gcc` link to gcc-10, which cannot compile its headers in C++17 mode; set
`CMAKE_CUDA_HOST_COMPILER` to a supported g++.

## Using it

```
smina -r rec.pdb -l ligands.sdf --autobox_ligand ref.sdf --gpu -o out.sdf
smina ... --gpu --device 0,1          # several GPUs
smina ... --gpu --fast                # GPU-style search (not 1:1, see below)
smina ... --minimize --gpu            # batched local optimization
```

All other options work as usual. `--gpu_split N` sets `--fast`'s chain split
by hand; `--fast` then runs half of smina's steps for N from 4 to 31 and all
of them otherwise. Hidden options (`--help_hidden`): `--gpu_threads` (threads per
search chain; chosen automatically), `--gpu_batch` (ligands per batch; chosen
automatically), `--gpu_fp64` (double precision kernels, for validation), and
`--fast`'s other components for experiments, `--gpu_qvina` (QuickVina 2's
check) and `--gpu_budget F` (fraction of smina's steps).

What runs where:

| step | with `--gpu` |
|---|---|
| reading molecules, typing, torsion trees | CPU (unchanged) |
| grid maps (`cache::populate`) | GPU, summed in double precision |
| Monte Carlo search (`parallel_mc`) | GPU, single precision |
| refinement of the search results (`refine_structure`) | GPU, single precision |
| `--minimize` / `--local_only` | GPU, double precision (see below) |
| final scoring, clustering, RMSD table, output | CPU in double precision (unchanged) |
| `--score_only`, `--randomize_only` | CPU (unchanged) |

With `--gpu`, ligands are docked in batches large enough to fill all GPUs
(the search chains of every ligand in a batch run concurrently); each ligand is
then finished and written in input order. A ligand's chains are seeded exactly
as on the CPU (a fresh generator from `--seed` per ligand, tasks seeded from
it in order). Results are reproducible for the same input and GPUs. In single
precision they can differ (statistically equivalently) between batch sizes or
GPU sets, because the number of threads per chain depends on how many chains
share a GPU and changes the rounding of sums; the double precision kernels do
not depend on it.

## Validation

* **Exactness.** The double precision kernels, with transcendental functions
  shared with a validation build of the CPU library (`portable_math.h`,
  `-DSMINA_PORTABLE_MATH`), reproduce CPU Monte Carlo and BFGS trajectories bit
  for bit (1HSG 300 steps × 8 chains; flexible residues with ad4 scoring;
  vinardo with splines; other complexes). Against the production CPU build
  (glibc libm), single evaluations agree to ~1e-14 and the grid maps are
  bit-identical.
* **Single precision, per evaluation** (energy and gradient vs CPU double):
  ≥ 99% of evaluations within 1e-4 relative for vina, vinardo, dkoes, ad4
  (spline), the 25-term `examples/kitchensink.score`, flexible residues and
  user grids; ad4 with linear tables on the direct receptor path reaches 98.7%
  (worst 2.3e-4). The remaining differences come from float coordinate
  resolution (~5e-6 Å) at discontinuities of the CPU scoring function (grid
  cell faces, the hard cutoff) and in steep regions.
* **Redocking** (25 Astex complexes × 5 seeds, top pose < 2 Å): CPU 92/125
  (73.6%), GPU 90/125 (72.0%); mean top score −9.835 vs −9.818 kcal/mol;
  per-complex outcomes agree; the differences are within run-to-run noise
  (permutation tests p = 0.60 and 0.51).
* **Screening** (210 ligands, 1HSG box, exhaustiveness 8): per-ligand top
  score GPU − CPU has median +0.001 and s.d. 0.074 kcal/mol; two CPU runs with
  different seeds differ by s.d. 0.073.
* **`--minimize`**, 1890 docked poses: output identical to the CPU's (energies
  and coordinates). 1890 random, mostly clashing poses (`--randomize_only`
  output): these minimizations are chaotic — two CPU builds that differ only
  in the last bit of sin/cos (glibc vs `portable_math.h`) agree within
  0.01 kcal/mol on 70.5% of them; the GPU agrees with production CPU smina on
  70.4%. Against the portable CPU build, the GPU's double precision
  minimizations are bit-identical for all 1890 once the CPU cell-list quirk
  described below is excluded (`GPU_TEST_FULL_CELLS`), 1888/1890 with it.

* **`--fast`**, the same 25 Astex complexes × 5 seeds: 93/125 (74.4%), mean
  top score −9.838 kcal/mol; against the CPU (92/125, −9.835), +1 success
  (permutation p = 1.00) and −0.003 kcal/mol (p = 0.79). Its screening forms,
  with half the steps, redock within noise as well: split 8, 90/125 (p = 0.62;
  +0.010 kcal/mol, p = 0.39); split 16, 92/125 (+0.024, p = 0.13). On the
  210-ligand screen (split 4 on one GPU, 8 on two), top scores differ from the
  CPU's by s.d. 0.083 (one GPU) and 0.081 kcal/mol (two GPUs), means within
  0.002; two CPU runs with different seeds differ by 0.073. Validated with
  rigid receptors and the default scoring function.

Tools: `src/gpu/tests/gpu_eval_test` (grids and evaluations),
`src/gpu/tests/gpu_mc_test` (trajectories, `--minimize`, determinism,
throughput), `smina_portable` (smina built against the portable-math library,
with and without `--gpu --gpu_fp64`), `tools/redock_benchmark.py` with
`tools/compare_redock.py`, `tools/compare_screen.py`,
`tools/compare_minimize.py`. `GPU_ITEM_TIMES=1` prints when each pose of a
local optimization finished; `GPU_CHAIN_STATS=1` prints, for every search
launch, the last chains to finish with their evaluation counts and the mean
time per evaluation; `-DGPU_PROFILE` builds report per-section cycles.

## Performance

Ryzen 9 5950X (16 cores, 32 threads); RTX 4080 SUPER and RTX 4070 Ti SUPER;
CPU smina built from this tree (`-O3`, same compiler); idle machine, wall
clock times including input and output.

| workload | CPU smina | `--gpu` (1:1) | `--gpu --fast` |
|---|---|---|---|
| 1HSG redocking, exhaustiveness 8 | 43 s (`--cpu 8`) | 26 s | 1.2 s |
| … exhaustiveness 32 | 58 s (`--cpu 32`) | 44 s | 2.5 s |
| … exhaustiveness 128 | 173 s (`--cpu 32`) | 47 s | 5.7 s |
| Astex set, exhaustiveness 8, per complex | | 16.8 s | 1.5 s |
| 1HSG, flexible residues A:50 and B:50, exhaustiveness 8 | 41 s (`--cpu 8`) | 48 s | 1.2 s (not validated) |
| screening 210 ligands (1HSG box, exhaustiveness 8): one GPU / both | 236 s (32 single-threaded processes) | 32 s / 19.5 s | 7.9 s / 5.4 s |
| screening 840 ligands (the 210, four times): one GPU / both | not measured | 57 s / 47 s | 28.6 s / 16.9 s |
| `--minimize`, 1890 docked poses | 0.47 s (32 processes), 4.8 s (1 process) | 2.3 s | |
| `--minimize`, 1890 random poses | 4.4 s (32 processes), 49 s (1 process) | 13 s | |

Single-complex runs used one GPU (the default, device 0, an RTX 4080 SUPER);
the Astex runs used one GPU per complex; "one GPU" screens the RTX 4080
SUPER. Times are single runs; a run takes as long as its
slowest search chain, and single precision trajectories differ between seeds,
so times vary by about ±10%.

The search is a set of independent but internally sequential Monte Carlo
chains (exhaustiveness 8 means 8 chains of tens of thousands of steps). A
single ligand at the default exhaustiveness therefore keeps only a few
streaming multiprocessors busy and gains little; the GPU pays off when many
chains run at once — many ligands, or higher exhaustiveness.

A launch lasts as long as its slowest chain, and chains differ severalfold in
work: the number of steps grows with the ligand's size and flexibility, and
the number of evaluations per step varies with the trajectory, up to threefold
between chains of the same ligand. Per chain, an evaluation takes about as
long with 128 threads as with 256, 1.2 times as long with 64 and 2-5 times as
long with 32 (which has the best throughput). So the GPU code uses the most
threads per chain that keep every chain resident (single complexes:
exhaustiveness 128 takes 47 s, against 62 s in two waves at 256 threads); 64
threads when the chains would make a single wave at 32 (the longest chains
start first and run 2-4 times faster, the short ones fill in behind: 210
ligands on one GPU take 32 s instead of 37 s); and 32 threads beyond that.
Giving the remaining chains more threads as others finish (chains can stop and
resume exactly) did not do better: the boundaries are barriers held by the
slowest chain, whose excess work no estimate made in advance predicts.

`--fast` attacks both problems, and the search's redundancy. QuickVina 2's
history check skips local optimizations whose starting point the chain's
recent history says will fall back into a known minimum (about 2× on its own:
Astex 16.8 s → 8.5 s). Splitting every chain into several shorter ones fills
the GPU for a single complex and shortens a screen's tail; the split is
chosen per batch, the largest power of two from 4 to 32 that keeps a GPU's
chains within 4 waves (32 for a single complex, 4 or 8 for the screens
above), because shorter chains need more evaluations for the same steps.
Once the GPU is full (split below 32), half of smina's steps are run, which
halves a screen's time; a single complex's run is mostly fixed overhead, so
it keeps the full budget. Budgets below half, and the budgets of Uni-Dock
(about 1-8% of smina's steps), cost redocking success and top scores.
[docs/fast-search.md](docs/fast-search.md) has the measurements.

`--minimize` gains nothing over a multi-core CPU. Minimizing a docked pose
costs about 2 ms of CPU time; the GPU does all 1890 in 0.14 s, but the rest of
the run — reading the molecules with OpenBabel (0.9 s), final rescoring and
output (0.3 s), start-up — stays serial on the host, whereas 32 CPU processes
share all of it. Random, clashing poses are the opposite case: a few of them
need tens of thousands of BFGS iterations in double precision, which consumer
GPUs execute at 1/64 of the single precision rate, and the slowest pose sets
the run time. (A single precision descent finished in double precision was
tried: it did not shorten that tail, and it matched the CPU less often.) For
local optimization of many poses, run several CPU processes.

## Differences and limitations

* Single precision search: trajectories diverge from the CPU's after the first
  steps; results are statistically equivalent, not pose-for-pose identical.
* Local optimization (`--minimize`, `--local_only`) uses the double precision
  kernels: it iterates BFGS until |g|² < 1e-4, which float coordinate
  resolution cannot reliably reach (single precision minimizations stall at
  the 10000 iteration limit). In double precision they match the CPU.
* Direct receptor evaluation (refinement, `--minimize`): smina's
  `szv_grid_cache` fills each 3 Å cell with the receptor atoms near the brick
  around the first coordinate that falls into it. When that coordinate lies
  exactly on a cell plane (an input atom at, say, z = −6.0000), the brick is
  flat and the cell misses receptor atoms for the rest of the run. The GPU's
  cell lists are complete. This changed 2 of 1890 random-pose minimizations
  and none of the docked poses.
* `--approximation exact` is not supported with `--gpu`.
* `--user_grid` while docking: smina folds the user grid into the grid maps
  evaluated at grid *indices* with the 1e6 out-of-box slope, producing map
  values up to ~1e7; in single precision their interpolated gradients lose
  about three digits. The behaviour itself is reproduced.
* Progress is shown as one bar per batch of ligands, ahead of the ligands'
  result tables.
* Some molecules (e.g. heme, PDB 1g9v) are converted by OpenBabel into
  different torsion trees in every process, so their results vary between runs
  of CPU smina as well as of the GPU version.
