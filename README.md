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
smina ... --gpu --gpu_split 16      # faster GPU-style search (not smina's)
```

All other smina options work as usual, including custom scoring, flexible
residues and per-atom term output. `--gpu_split N` replaces every search chain
with N shorter chains; it is much faster for single complexes and small
screens and redocks as well as smina (validated), but it is not the original
algorithm.

## Performance

Ryzen 9 5950X (16 cores) vs RTX 4080 SUPER (+ RTX 4070 Ti SUPER):

| | CPU smina | `--gpu` | `--gpu_split` |
|---|---|---|---|
| redock 1HSG, exhaustiveness 8 | 43 s | 26 s | 2.9 s (N = 32) |
| … exhaustiveness 128 | 173 s | 47 s | |
| screen 210 ligands, 1 GPU / 2 GPUs | 236 s (32 processes) | 32 s / 19.5 s | 18.5 s / 12.4 s (N = 8) |

## Validation

- Redocking 25 Astex complexes × 5 seeds (top pose < 2 Å): CPU 73.6%,
  `--gpu` 72.0%, `--gpu_split 8/16/32` 72.8%; differences within run-to-run
  noise.
- Screening top scores differ from CPU smina by s.d. 0.074 kcal/mol, the same
  as two CPU runs with different seeds (0.073).
- The double precision kernels (`--gpu_fp64`) reproduce CPU smina's
  trajectories bit for bit, against a CPU build that shares their sin/cos/exp.

## More

- [README.GPU.md](README.GPU.md): details, validation, performance analysis
  and limitations of the GPU version
- [README.smina](README.smina): the original smina README and option
  reference

## License and citation

smina is distributed under the Apache 2.0 ([LICENSE.APACHE](LICENSE.APACHE))
and GPL v2 ([LICENSE.GNU](LICENSE.GNU)) licenses, and so is this version.
`src/lib/portable_math.h` derives from fdlibm (Sun Microsystems notice in the
file). If you use smina, please cite
[Koes et al., J. Chem. Inf. Model. 2013](http://pubs.acs.org/doi/abs/10.1021/ci300604z).
