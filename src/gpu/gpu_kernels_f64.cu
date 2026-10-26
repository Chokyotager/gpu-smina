// double-precision (validation) instantiation of the GPU kernels; built
// with -fmad=false so arithmetic matches the CPU (no FMA contraction)
#define GPU_PRECISION double
#include "gpu_kernels_impl.cuh"
