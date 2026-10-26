/*
 * Host-side conversion of smina's data structures (model, precalculate,
 * receptor, grids) into the flat double-precision arrays that are uploaded
 * to the GPU.  Compiled by the host compiler only; never included by .cu
 * files.
 */

#ifndef SMINA_GPU_FLATTEN_H
#define SMINA_GPU_FLATTEN_H

#include <vector>
#include "gpu_defs.h"
#include "gpu_flatten_data.h"
#include "model.h"
#include "precalculate.h"
#include "grid.h"
#include "cache.h"

// global namespace: smina classes befriend ::gpu_model_flattener
struct gpu_model_flattener
{
	typedef gpusmina::FlatModel FlatModel;
	typedef gpusmina::FlatTables FlatTables;
	typedef gpusmina::FlatReceptor FlatReceptor;
	typedef gpusmina::FlatGridGeometry FlatGridGeometry;
	typedef gpusmina::FlatUserGrid FlatUserGrid;

	static FlatModel flatten_model(const model& m);
	static FlatTables flatten_tables(const precalculate& p);
	// relevant receptor atoms for box gd, with cell lists for non_cache
	static FlatReceptor flatten_receptor(const model& m, const grid_dims& gd,
			double cutoff_sqr);
	// cell lists exactly as cache::populate builds them for its probes
	static FlatReceptor flatten_receptor_for_grid(const model& m,
			const grid_dims& gd, double cutoff_sqr);
	static FlatGridGeometry grid_geometry(const grid_dims& gd);
	static FlatUserGrid flatten_user_grid(const grid& g);

	// test access to cache internals: data / chargedata of type t
	static const grid& cache_grid(const cache& c, int t) { return c.grids[t]; }
	static const array3d<fl>& grid_data(const grid& g) { return g.data; }
	static const array3d<fl>& grid_chargedata(const grid& g) { return g.chargedata; }
};

namespace gpusmina
{

// conf / change <-> flat vectors in the FlatModel layout
void conf_to_flat(const conf& c, const FlatModel& fm, double* out);
void flat_to_conf(const double* in, const FlatModel& fm, conf& c);
void change_to_flat(const change& g, const FlatModel& fm, double* out);
void flat_to_change(const double* in, const FlatModel& fm, change& g);

} // namespace gpusmina

#endif
