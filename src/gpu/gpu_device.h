/*
 * Ownership of device memory for the GPU code path.  The classes here take
 * the flat host representations (gpu_flatten.h) and expose device
 * descriptors (gpu_dev_structs.h) for kernels.  Templates are explicitly
 * instantiated for float (production) and double (validation) in gpu_device.cu.
 */

#ifndef SMINA_GPU_DEVICE_H
#define SMINA_GPU_DEVICE_H

#include <vector>
#include <string>
#include <stdexcept>
#include "gpu_defs.h"
#include "gpu_dev_structs.h"

namespace gpusmina
{

struct FlatModel;
struct FlatTables;
struct FlatReceptor;
struct FlatGridGeometry;
struct FlatUserGrid;

void cuda_check(int err, const char* what);

// a device allocation
class DevBuf
{
	void* ptr;
	size_t bytes;
	DevBuf(const DevBuf&);
	DevBuf& operator=(const DevBuf&);
public:
	DevBuf() : ptr(0), bytes(0) {}
	~DevBuf() { release(); }
	void release();
	void alloc(size_t n);
	void upload(const void* src, size_t n); // allocates n bytes
	void download(void* dst, size_t n) const;
	void zero();
	template<typename X> X* get() const { return static_cast<X*>(ptr); }
	size_t size() const { return bytes; }
};

// upload a host vector converted element-wise to X
template<typename X, typename Y>
void upload_converted(DevBuf& b, const std::vector<Y>& v)
{
	std::vector<X> tmp(v.begin(), v.end());
	if (tmp.empty())
		tmp.push_back(X());
	b.upload(&tmp[0], tmp.size() * sizeof(X));
}

template<typename T>
class DeviceTables
{
	DevBuf fast, smooth, spl_cutoff, spl_data;
public:
	DevTables<T> dev;
	void upload(const FlatTables& t);
};

template<typename T>
class DeviceReceptor
{
	DevBuf coords, type, charge, cell_start, cell_atoms;
public:
	DevReceptor<T> dev;
	void upload(const FlatReceptor& r, const FlatGridGeometry& box);
};

template<typename T>
class DeviceUserGrid
{
	DevBuf data;
public:
	DevUserGrid<T> dev;
	DeviceUserGrid() { dev.initialized = 0; dev.data = 0; }
	void upload(const FlatUserGrid& u);
};

// the cache grids (one per atom type), computed on the GPU exactly like
// cache::populate (sums in double, stored as T)
template<typename T>
class DeviceGrids
{
	DevBuf data[NUM_TYPES], chargedata[NUM_TYPES];
public:
	DevGrids<T> dev;
	DeviceGrids();
	// build grids for `types`; user grid (if initialized) is added like
	// cache::populate does; slope is the cache's slope
	void build(const FlatReceptor& rec_for_grid, const FlatGridGeometry& geo,
			const FlatTables& tables, const std::vector<int>& types,
			const FlatUserGrid& ug, double slope);
	void download(int t, std::vector<T>& d, std::vector<T>& cd) const;
	bool has(int t) const { return dev.data[t] != 0; }
};

// a batch of models in concatenated device arrays
template<typename T>
class DeviceModels
{
	DevBuf desc, atom_type, atom_charge, atom_local, atom_frame, init_coords,
			inflex_coords, frames, init_frame_origin, children, level_frames,
			level_start, pairs, list_start, blocks, heavy, nbr_start, nbr;
public:
	DevModels<T> dev;
	std::vector<ModelDesc> host_desc;
	int max_movable, max_frames, max_list, max_change, max_conf, max_heavy,
			max_pairs;
	void upload(const std::vector<const FlatModel*>& models);
};

// per-warp scratch sizing for EvalScratch (in elements of T)
ScratchOffsets scratch_layout(int max_movable, int max_frames, int max_list,
		int max_pairs, bool ordered);

} // namespace gpusmina

#endif
