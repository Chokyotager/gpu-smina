/*
 * GPU replacements for the expensive parts of smina's search, used by main
 * when --gpu is given:
 *   parallel_mc::operator()   -> GpuDocking::parallel_mc
 *   refine_structure (main)   -> GpuDocking::refine
 * Everything else (input, final scoring in double precision, clustering,
 * output) stays on the CPU code path.
 */

#ifndef SMINA_GPU_DOCKING_H
#define SMINA_GPU_DOCKING_H

#include <string>
#include <vector>
#include "model.h"
#include "parallel_mc.h"
#include "precalculate.h"
#include "grid.h"

namespace gpusmina
{

struct GpuSettings
{
	std::vector<int> devices; // CUDA device ordinals
	bool fp64; // double precision kernels (validation)
	int threads_per_chain; // 0: automatic
	// GPU-style search (not smina's): every Monte Carlo chain is replaced by
	// this many chains with 1/chain_split of its steps, run concurrently
	int chain_split;
	GpuSettings() : fp64(false), threads_per_chain(0), chain_split(1) {}
};

// parse "0,1" into device ordinals
std::vector<int> parse_devices(const std::string& s);

struct EngineBase;

class GpuDocking
{
	GpuSettings settings;
	std::vector<EngineBase*> engines; // one per device (search precision)
	// Local optimization (--minimize / --local_only) always runs in double
	// precision: it iterates bfgs until |g|^2 < 1e-4, which float coordinate
	// resolution (~5e-6 A) cannot reliably reach, so float minimizations
	// stall at the iteration limit.  Created on first use.
	std::vector<EngineBase*> local_engines;
	GpuDocking(const GpuDocking&);
	GpuDocking& operator=(const GpuDocking&);
public:
	explicit GpuDocking(const GpuSettings& s);
	~GpuDocking();

	// parallel_mc::operator(): the same chains (seeded from `generator`),
	// merged into `out` like merge_output_containers
	void parallel_mc(const ::parallel_mc& par, const model& m,
			output_container& out, const precalculate& p, const grid_dims& gd,
			const vec& corner1, const vec& corner2, rng& generator,
			const grid& user_grid, double cache_slope);

	// refine_structure(m, p, nc, out[i], cap, minparm, user_grid) for every
	// pose in `poses`; the non_cache box is `gd`
	void refine(const model& m, const precalculate& p, const grid_dims& gd,
			std::vector<output_type*>& poses, const vec& cap,
			const minimization_params& minparm, const grid& user_grid);

	// Batched docking of many ligands (virtual screening): for every job, the
	// equivalent of parallel_mc (seeded with a fresh rng(seed) like
	// do_search) followed by refine_structure of every pose.  Jobs are split
	// over the devices.
	struct DockJob
	{
		const model* m;
		::parallel_mc par; // this ligand's search parameters
		output_container out; // merged, refined poses
	};
	void dock_batch(std::vector<DockJob*>& jobs, const precalculate& p,
			const grid_dims& gd, const vec& corner1, const vec& corner2,
			int seed, const grid& user_grid, double cache_slope, const vec& cap,
			bool display_progress);

	// Batched local optimization (--local_only / --minimize): for every job,
	// refine_structure of job.out (in: initial conf; out: conf, e, heavy
	// atom coords) in the job's own box.
	struct MinimizeJob
	{
		const model* m;
		grid_dims gd;
		minimization_params minparm;
		output_type* out;
	};
	void minimize_batch(std::vector<MinimizeJob>& jobs, const precalculate& p,
			const vec& cap, const grid& user_grid);

	// search tasks (exhaustiveness units, each chain_split chains) that keep
	// all devices busy; batches should provide this many
	size_t chains_per_batch() const;

	std::string description() const;
};

} // namespace gpusmina

#endif
