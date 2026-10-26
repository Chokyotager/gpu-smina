/*
 * Host-side glue between smina's search objects (parallel_mc, monte_carlo,
 * output containers) and the GPU kernels.
 */

#ifndef SMINA_GPU_SEARCH_H
#define SMINA_GPU_SEARCH_H

#include "gpu_flatten.h"
#include "gpu_kernels.h"
#include "monte_carlo.h"

namespace gpusmina
{

// boost::mt19937 state, canonicalized by its operator<< (i == n)
MTState export_rng(const rng& g);

// monte_carlo::operator() setup for one task: a conf of the model's size
// randomized in the box with the task's generator (seeded like parallel_mc)
ChainInput make_chain(const model& m, const FlatModel& fm, int model_index,
		int seed, const vec& corner1, const vec& corner2);

MCParams mc_params(const monte_carlo& mc);

// ChainResult -> output_container (sorted, with conf, e and heavy coords)
void chain_to_container(const ChainResult& r, const FlatModel& fm,
		output_container& out);

} // namespace gpusmina

#endif
