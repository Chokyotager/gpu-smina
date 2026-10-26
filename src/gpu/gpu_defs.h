/*
 * Plain data structures shared by the host-side flattening code (g++) and
 * the CUDA kernels (nvcc).  Nothing in here may depend on boost or on the
 * smina model classes, so that .cu files stay free of those headers.
 *
 * The GPU path is a 1:1 port of smina's search.  Every structure here mirrors
 * a piece of the CPU data model; comments name the CPU counterpart.
 */

#ifndef SMINA_GPU_DEFS_H
#define SMINA_GPU_DEFS_H

#ifdef __CUDACC__
#define GPU_HD __host__ __device__ __forceinline__
#define GPU_D __device__ __forceinline__
#else
#define GPU_HD inline
#define GPU_D inline
#endif

namespace gpusmina
{

// smina_atom_type::NumTypes
const int NUM_TYPES = 27;
const int NUM_TYPE_PAIRS = NUM_TYPES * (NUM_TYPES + 1) / 2;
// result_components::Last
const int MAX_COMPONENTS = 4;

GPU_HD int tri_index(int i, int j) // triangular_matrix_index, requires i <= j
{
	return i + j * (j + 1) / 2;
}

GPU_HD int tri_index_permissive(int i, int j)
{
	return (i <= j) ? tri_index(i, j) : tri_index(j, i);
}

GPU_HD bool type_is_hydrogen(int t) // ::is_hydrogen(smt)
{
	return t == 0 || t == 1;
}

enum FrameKind
{
	FRAME_RIGID = 0, // rigid_body: root of a ligand
	FRAME_FIRST_SEGMENT = 1, // first_segment: root of a flexible residue
	FRAME_SEGMENT = 2 // segment: any rotatable branch
};

// one node of a ligand / residue torsion tree (tree.h); frames are stored
// per model in CPU set_conf order (ligands then flex, each in preorder)
template<typename T>
struct Frame
{
	int parent; // model-local frame index, -1 for roots
	int kind; // FrameKind
	int begin, end; // movable atoms owned by this node
	int child_begin, child_end; // range in the model's child index array
	int conf_idx; // rigid: position in conf; torsion frames: torsion in conf
	int change_idx; // same, in change (gradient) layout
	T rel_origin[3]; // segment: relative_origin
	T rel_axis[3]; // segment: relative_axis
	T origin[3]; // first_segment: fixed origin
	T axis[3]; // first_segment: fixed axis
};

// interacting_pair with its type-pair table resolved on the host.  (a, b)
// keep the CPU orientation (force on a is -f, on b is +f).  `tpair` indexes
// the table of (min type, max type); `swapped` is set when type(a) > type(b).
// The linear and spline backends combine charge components in different
// orders for swapped pairs, so the raw charge factors are kept and each
// backend applies its own CPU expression order.
template<typename T>
struct Pair
{
	int a, b;
	int tpair;
	int swapped;
	T abs_qa, abs_qb, qab; // |q_a|, |q_b|, q_a*q_b
};

// ligand_conf / residue_conf block layout inside the flat conf and change
// vectors (conf.h).  A ligand occupies 7 + ntors conf values (position,
// quaternion, torsions) and 6 + ntors change values.
struct ConfBlock
{
	int conf_offset;
	int change_offset;
	int ntors;
	int is_ligand;
	int root_frame; // model-local frame index of the block's root
	int atom_begin, atom_end; // ligand atom range (model::gyration_radius)
};

// per-model offsets into the batch-wide arrays (see DeviceModels)
struct ModelDesc
{
	int num_movable; // model::m_num_movable_atoms
	int num_atoms; // movable + inflex
	int atom_offset; // movable atom arrays (type, charge, local coords, frame)
	int inflex_offset; // inflex coordinates (index into array of 3-vectors)
	int frame_offset, num_frames;
	int child_offset;
	int level_offset, num_levels; // frames grouped by tree depth
	int pair_offset; // pair lists: other_pairs, then each ligand's pairs
	int list_offset, num_lists; // list_start entries (num_lists + 1 of them)
	int block_offset, num_blocks; // ConfBlocks: ligands then flex
	int conf_size, change_size;
	int heavy_offset, num_heavy; // movable heavy atoms (output coords, within)
	int nbr_start_offset, nbr_offset; // per-atom pair lists
	int num_pairs, num_nbr, num_children;
	int num_ligands, num_flex;
};

} // namespace gpusmina

#endif
