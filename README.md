# GPU smina

[smina](https://sourceforge.net/projects/smina/) (a fork of AutoDock Vina)
with CUDA acceleration. `--gpu` runs smina's own search on NVIDIA GPUs: the
same Monte Carlo search, BFGS minimizer, scoring tables and parameters, so
results match CPU smina statistically. Without `--gpu` the program behaves
exactly as before.

## Build

Requires Boost, OpenBabel 3, CMake ≥ 3.18 and the CUDA toolkit (tested with
12.6 on sm_89).

```
cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=<boost/openbabel prefix> \
      -DCMAKE_CUDA_HOST_COMPILER=g++-13
ninja -C build smina
```

## Use

```
smina -r rec.pdb -l ligands.sdf --autobox_ligand ref.sdf --gpu -o out.sdf
smina ... --gpu --device 0,1        # several GPUs
smina ... --gpu --fast              # much faster GPU-style search (not smina's)
```

All other smina options work as usual, including custom scoring, flexible
residues and per-atom term output. `--fast` skips redundant local
optimizations (QuickVina 2's check) and splits the search into many shorter
chains. On screens it also runs at most half of smina's steps, stops a
ligand once its runs agree, and gives the full search only to the ligands a
quick first pass ranks in the top 10% (`--fast_top 1`: every ligand). It
redocks as well as smina and finds the same top of a screen (validated), but
it is not the original algorithm.

## Performance

Ryzen 9 5950X (16 cores) vs RTX 4080 SUPER (+ RTX 4070 Ti SUPER):

| | CPU smina | `--gpu` | `--gpu --fast` |
|---|---|---|---|
| redock 1HSG, exhaustiveness 8 | 43 s | 26 s | 1.2 s |
| … exhaustiveness 128 | 173 s | 47 s | 5.7 s |
| screen 210 ligands, 1 GPU / 2 GPUs | 236 s (32 processes) | 32 s / 19.5 s | 4.5 s / 4.8 s |
| screen 840 ligands, 1 GPU / 2 GPUs | | 57 s / 47 s | 13.3 s / 8.8 s |

## Validation

- Redocking 25 Astex complexes × 5 seeds (top pose < 2 Å): CPU 73.6%,
  `--gpu` 72.0%, `--gpu --fast` 74.4%; differences within run-to-run noise.
- Screening top scores differ from CPU smina by s.d. 0.074 kcal/mol with
  `--gpu` and 0.085-0.088 with `--fast --fast_top 1`; two CPU runs with
  different seeds differ by 0.073. With two-pass screening the CPU's top
  ligands stay in `--fast`'s top about as often as without it; ligands
  outside the top keep rougher first-pass scores.
- The double precision kernels (`--gpu_fp64`) reproduce CPU smina's
  trajectories bit for bit, against a CPU build that shares their sin/cos/exp.

## More

- [README.GPU.md](README.GPU.md): details, validation, performance analysis
  and limitations of the GPU version
- [docs/fast-search.md](docs/fast-search.md): how `--fast` was chosen and
  validated
- [README.smina](README.smina): the original smina README and option
  reference

## License and citation

smina is distributed under the Apache 2.0 ([LICENSE.APACHE](LICENSE.APACHE))
and GPL v2 ([LICENSE.GNU](LICENSE.GNU)) licenses, and so is this version.
`src/lib/portable_math.h` derives from fdlibm (Sun Microsystems notice in the
file). If you use smina, please cite
[Koes et al., J. Chem. Inf. Model. 2013](http://pubs.acs.org/doi/abs/10.1021/ci300604z).
`--fast` uses QuickVina 2's history check; if you use it, please also cite
[Alhossary et al., Bioinformatics 2015](https://doi.org/10.1093/bioinformatics/btv082).
