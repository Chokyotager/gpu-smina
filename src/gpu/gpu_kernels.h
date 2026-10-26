/*
 * Host entry points of the GPU kernels.  Each is instantiated for float
 * (production) and double (validation, CPU-ordered reductions, no FMA).
 */

#ifndef SMINA_GPU_KERNELS_H
#define SMINA_GPU_KERNELS_H

#include <vector>
#include "gpu_device.h"

namespace gpusmina
{

// model::eval_deriv for many conformations of one model (one warp each).
// confs: nconf * conf_size; outputs e (nconf) and g (nconf * change_size)
// and optionally the movable atom coordinates (nconf * 3 * num_movable).
// Returns false if a ligand atom exactly overlapped a receptor atom
// (non_cache's runtime error).
template<typename T>
bool eval_confs(const DeviceModels<T>& models, int model,
		const DeviceTables<T>& tables, const InterContext<T>& ic,
		const double v[3], const std::vector<double>& confs,
		std::vector<double>& e, std::vector<double>& g,
		std::vector<double>* coords = 0);

} // namespace gpusmina

#endif
