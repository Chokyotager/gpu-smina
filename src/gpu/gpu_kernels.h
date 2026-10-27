/*
 * Host entry points of the GPU kernels.  Each is instantiated for float
 * (production) and double (validation, CPU-ordered reductions, no FMA).
 */

#ifndef SMINA_GPU_KERNELS_H
#define SMINA_GPU_KERNELS_H

#include <vector>
#include "gpu_device.h"
#include "gpu_mc_structs.h"
#include "gpu_rng.cuh"

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
		std::vector<double>* coords = 0, int threads_per_chain = 32);

// threads_per_chain (32, 64 or 128): 32 maximizes throughput with many
// chains; more threads lower the latency of each chain.

// one Monte Carlo chain: its model, initial (randomized) conf and the state
// of its random number generator after the randomization
struct ChainInput
{
	int model;
	std::vector<double> conf;
	MTState rng;
	unsigned num_steps; // per-ligand search length (0: params.num_steps)
	unsigned maxiters; // per-ligand bfgs iterations (0: params.min.maxiters)
	ChainInput() : model(0), num_steps(0), maxiters(0) {}
};

// a chain's output container, sorted by energy
struct ChainResult
{
	std::vector<double> e;
	std::vector<std::vector<double> > conf;
	std::vector<std::vector<double> > coords; // heavy atoms, 3 per atom
};

// load every kernel of the precision on the current device (see the
// definition)
template<typename T>
void preload_kernels();

// How many items (chains, conformations) a kernel keeps resident on the
// current device with 32, 64, 128 and 256 threads per item, as the
// occupancy calculator gives it from the kernels' registers.
struct Residency
{
	size_t items[4];
	// the most threads per item that keep n items resident, else 32
	int tpc(size_t n) const
	{
		for (int i = 3; i > 0; i--)
			if (n <= items[i])
				return 32 << i;
		return 32;
	}
};
template<typename T>
Residency mc_residency();
template<typename T>
Residency bfgs_residency();

// monte_carlo::operator() for every chain, using the cache grids in ic.
// progress, if given, counts completed MC steps (host-mapped).  With
// threads_per_chain == 0 and the device's residency given, the threads per
// chain are chosen for the chains (see mc_tpc).
template<typename T>
void run_monte_carlo(const DeviceModels<T>& models,
		const DeviceTables<T>& tables, const InterContext<T>& ic,
		const MCParams& params, const std::vector<ChainInput>& chains,
		std::vector<ChainResult>& results, unsigned* progress = 0,
		int threads_per_chain = 32, const Residency* residency = 0);

// quasi_newton::operator() (bfgs) on every conf in place (one warp each);
// returns the minimized energies.  Uses ic's back end (cache or non_cache).
template<typename T>
bool run_bfgs(const DeviceModels<T>& models, const DeviceTables<T>& tables,
		const InterContext<T>& ic, const MinParams& params, const double v[3],
		const std::vector<int>& model_of, std::vector<std::vector<double> >& confs,
		std::vector<double>& e, int threads_per_chain = 32);

// main.cpp refine_structure for every conf (non_cache back end in ic, v =
// cap): bfgs with increasing slope until in the box.  Energies of poses that
// never got inside are the precision's max; coords receives the final heavy
// atom coordinates.
// maxiters, if not empty, gives each conf its own iteration limit and boxes,
// if not empty, its own non_cache box (otherwise ic.box).
template<typename T>
bool run_refine(const DeviceModels<T>& models, const DeviceTables<T>& tables,
		const InterContext<T>& ic, const MinParams& params, const double cap[3],
		const std::vector<int>& model_of, std::vector<std::vector<double> >& confs,
		std::vector<double>& e, std::vector<std::vector<double> >& coords,
		int threads_per_chain = 32,
		const std::vector<unsigned>& maxiters = std::vector<unsigned>(),
		const std::vector<FlatGridGeometry>& boxes =
				std::vector<FlatGridGeometry>());

} // namespace gpusmina

#endif
