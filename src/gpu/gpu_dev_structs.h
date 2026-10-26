/*
 * Descriptors of device-resident data, shared between host code (which fills
 * them in after uploading) and the kernels.  Plain data only.
 */

#ifndef SMINA_GPU_DEV_STRUCTS_H
#define SMINA_GPU_DEV_STRUCTS_H

#include "gpu_defs.h"

namespace gpusmina
{

template<typename T>
struct DevTables
{
	int kind; // FlatTables::Kind
	int nc; // components
	T cutoff_sqr;
	// linear
	T factor;
	int n;
	const T* lin_fast; // [pair][i][c]
	const T* lin_smooth; // [pair][i][c][2]
	// spline
	T fraction;
	int nseg;
	const T* spl_cutoff; // [pair][c]
	const T* spl_data; // [pair][c][seg][5]
};

template<typename T>
struct DevGridGeometry
{
	int dim[3];
	T init[3], factor[3], dim_fl_minus_1[3], factor_inv[3];
};

// cache: one grid per smina atom type (null when not needed)
template<typename T>
struct DevGrids
{
	DevGridGeometry<T> geo;
	const T* data[NUM_TYPES];
	const T* chargedata[NUM_TYPES];
};

template<typename T>
struct DevUserGrid
{
	int initialized;
	DevGridGeometry<T> geo;
	const T* data;
};

// non_cache: receptor atoms near the box, 3 A cell lists, box bounds
template<typename T>
struct DevReceptor
{
	const T* coords;
	const int* type;
	const T* charge;
	int offset[3], dims[3];
	const int* cell_start;
	const int* cell_atoms;
	T gd_begin[3], gd_end[3];
	int gd_enabled[3];
};

// one model of the batch, resolved to direct pointers
template<typename T>
struct ModelView
{
	ModelDesc d;
	const int* atom_type;
	const T* atom_charge;
	const T* atom_local;
	const int* atom_frame;
	const T* inflex_coords;
	const Frame<T>* frames;
	const int* children;
	const int* level_frames;
	const int* level_start;
	const Pair<T>* pairs;
	const int* list_start;
	const ConfBlock* blocks;
	const int* heavy;
	const int* nbr_start;
	const int* nbr;
};

// per-warp scratch (global or shared memory)
template<typename T>
struct EvalScratch
{
	T* coords; // 3 * num_movable
	T* forces; // 3 * num_movable (model::minus_forces)
	T* frame_origin; // 3 per frame
	T* frame_q; // 4 per frame
	T* frame_m; // 9 per frame
	T* frame_axis; // 3 per frame (lab)
	T* frame_ft; // 6 per frame: force, torque
	T* pair_buf; // ORDERED only: 4 per pair (e, f) plus per-atom energies
	int* error; // set when a ligand atom sits on a receptor atom
};

// which intermolecular back end a model is evaluated with
template<typename T>
struct InterContext
{
	int use_cache; // cache (grids) or non_cache (receptor atoms)
	DevGrids<T> grids;
	DevReceptor<T> rec;
	DevUserGrid<T> ug;
	T slope;
};

// element offsets of the EvalScratch arrays inside one warp's scratch block
struct ScratchOffsets
{
	int coords, forces, frame_origin, frame_q, frame_m, frame_axis, frame_ft,
			total; // hot part
	int pair_total; // pair buffer, allocated separately
};

// all models of a batch: per-model descriptors plus concatenated arrays
template<typename T>
struct DevModels
{
	int num_models;
	const ModelDesc* desc;
	const int* atom_type;
	const T* atom_charge;
	const T* atom_local;
	const int* atom_frame;
	const T* init_coords;
	const T* inflex_coords;
	const Frame<T>* frames;
	const T* init_frame_origin;
	const int* children;
	const int* level_frames;
	const int* level_start;
	const Pair<T>* pairs;
	const int* list_start;
	const ConfBlock* blocks;
	const int* heavy;
	const int* nbr_start;
	const int* nbr;
};

} // namespace gpusmina

#endif
