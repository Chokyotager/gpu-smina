/*
 * Flat, double-precision host representations of smina data destined for
 * the GPU.  Plain std::vector data only, so .cu files may include this.
 */

#ifndef SMINA_GPU_FLATTEN_DATA_H
#define SMINA_GPU_FLATTEN_DATA_H

#include <vector>
#include "gpu_defs.h"

namespace gpusmina
{

// a model (ligand(s) + flexible residues) laid out for the GPU
struct FlatModel
{
	int num_movable = 0;
	int num_atoms = 0;
	std::vector<int> atom_type; // movable atoms
	std::vector<double> atom_charge;
	std::vector<double> atom_local; // 3 per movable atom, frame-local coords
	std::vector<int> atom_frame;
	std::vector<double> init_coords; // 3 per movable atom: model coords now
	std::vector<double> inflex_coords; // 3 per inflex atom
	std::vector<Frame<double> > frames;
	std::vector<double> init_frame_origin; // 3 per frame: origin now
	std::vector<int> children;
	std::vector<int> level_frames; // frames grouped by depth
	std::vector<int> level_start; // num_levels + 1
	std::vector<Pair<double> > pairs;
	std::vector<int> list_start; // list 0 = other_pairs, then each ligand
	std::vector<ConfBlock> blocks; // ligands then flex
	std::vector<int> heavy; // movable non-hydrogen atoms
	int conf_size = 0, change_size = 0;
	int num_ligands = 0, num_flex = 0;

	int num_lists() const { return int(list_start.size()) - 1; }
	int num_levels() const { return int(level_start.size()) - 1; }
};

// precalculated pair tables (precalculate_linear or precalculate_splines)
struct FlatTables
{
	enum Kind { Linear = 0, Spline = 1 };
	int kind = Linear;
	int num_components = 1;
	double cutoff = 0, cutoff_sqr = 0;

	// precalculate_linear: n points per type pair
	double factor = 0;
	int n = 0;
	std::vector<double> lin_fast; // [pair][i][c]
	std::vector<double> lin_smooth; // [pair][i][c][e, dor]

	// precalculate_splines: nseg segments per type pair and component
	double fraction = 0;
	int nseg = 0;
	std::vector<double> spl_cutoff; // [pair][c] (0 => component absent)
	std::vector<double> spl_data; // [pair][c][seg][x, a, b, c, d]
};

// receptor atoms that can interact with the search box, with the
// szv_grid-style 3 Angstrom cell lists used to enumerate them
struct FlatReceptor
{
	std::vector<double> coords; // 3 per atom
	std::vector<int> type;
	std::vector<double> charge;
	int offset[3] = {0, 0, 0}; // szv_grid_cache::get_local_dims
	int dims[3] = {0, 0, 0};
	std::vector<int> cell_start;
	std::vector<int> cell_atoms;
};

// grid::init parameters for a grid_dims (shared by every cache grid)
struct FlatGridGeometry
{
	int dim[3] = {0, 0, 0}; // gd[i].n + 1 sample points
	double init[3], factor[3], dim_fl_minus_1[3], factor_inv[3];
	double gd_begin[3], gd_end[3]; // non_cache box (grid_dims begin/end)
	int gd_enabled[3]; // gd[i].n > 0
};

// a user-supplied grid (--user_grid)
struct FlatUserGrid
{
	bool initialized = false;
	FlatGridGeometry geom;
	std::vector<double> data; // x fastest, like array3d
};

} // namespace gpusmina

#endif
