/*
 * Kernel templates; included once per precision by gpu_kernels_f32.cu and
 * gpu_kernels_f64.cu, which explicitly instantiate the host entry points.
 */

#include "gpu_kernels.h"
#include "gpu_eval.cuh"
#include <cuda_runtime.h>
#include <type_traits>

namespace gpusmina
{

template<typename T>
struct Ordered
{
	static const bool value = std::is_same<T, double>::value;
};

template<typename T>
__global__ void eval_confs_kernel(DevModels<T> dm, int model, DevTables<T> tb,
		InterContext<T> ic, T v0, T v1, T v2, const T* confs, int nconf,
		T* out_e, T* out_g, T* out_coords, T* scratch, ScratchOffsets so,
		int* error)
{
	const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
	const int lane = threadIdx.x & 31;
	if (warp >= nconf)
		return;
	const ModelView<T> mv = make_view(dm, model);
	EvalScratch<T> ws = make_scratch(scratch + size_t(warp) * so.total, so,
			error);
	const T v[3] = {v0, v1, v2};
	T* g = out_g + size_t(warp) * mv.d.change_size;
	T e = eval_deriv<T, Ordered<T>::value>(mv, tb, ic, v,
			confs + size_t(warp) * mv.d.conf_size, g, ws, lane);
	if (lane == 0)
		out_e[warp] = e;
	if (out_coords)
		for (int i = lane; i < 3 * mv.d.num_movable; i += 32)
			out_coords[size_t(warp) * 3 * mv.d.num_movable + i] = ws.coords[i];
}

template<typename T>
bool eval_confs(const DeviceModels<T>& models, int model,
		const DeviceTables<T>& tables, const InterContext<T>& ic,
		const double v[3], const std::vector<double>& confs,
		std::vector<double>& e, std::vector<double>& g,
		std::vector<double>* coords)
{
	const ModelDesc& d = models.host_desc[model];
	const int nconf = int(confs.size() / d.conf_size);
	ScratchOffsets so = scratch_layout(models.max_movable, models.max_frames,
			models.max_list, Ordered<T>::value);
	DevBuf dconfs, de, dg, dscratch, derr, dcoords;
	if (coords)
		dcoords.alloc(sizeof(T) * size_t(nconf) * 3 * d.num_movable);
	upload_converted<T>(dconfs, confs);
	de.alloc(sizeof(T) * nconf);
	dg.alloc(sizeof(T) * size_t(nconf) * d.change_size);
	dscratch.alloc(sizeof(T) * size_t(nconf) * so.total);
	dscratch.zero();
	int zero = 0;
	derr.upload(&zero, sizeof(int));

	const int threads = 128;
	const int blocks = (nconf * 32 + threads - 1) / threads;
	eval_confs_kernel<T><<<blocks, threads>>>(models.dev, model, tables.dev, ic,
			T(v[0]), T(v[1]), T(v[2]), dconfs.get<T>(), nconf, de.get<T>(),
			dg.get<T>(), coords ? dcoords.get<T>() : (T*) 0, dscratch.get<T>(),
			so, derr.get<int>());
	cuda_check(cudaGetLastError(), "eval_confs_kernel launch");
	cuda_check(cudaDeviceSynchronize(), "eval_confs_kernel");

	std::vector<T> he(nconf), hg(size_t(nconf) * d.change_size);
	de.download(&he[0], he.size() * sizeof(T));
	if (!hg.empty())
		dg.download(&hg[0], hg.size() * sizeof(T));
	int err = 0;
	derr.download(&err, sizeof(int));
	if (coords)
	{
		std::vector<T> hc(size_t(nconf) * 3 * d.num_movable);
		dcoords.download(&hc[0], hc.size() * sizeof(T));
		coords->assign(hc.begin(), hc.end());
	}
	e.assign(he.begin(), he.end());
	g.assign(hg.begin(), hg.end());
	return err == 0;
}

template bool eval_confs<GPU_PRECISION>(const DeviceModels<GPU_PRECISION>&,
		int, const DeviceTables<GPU_PRECISION>&,
		const InterContext<GPU_PRECISION>&, const double[3],
		const std::vector<double>&, std::vector<double>&, std::vector<double>&,
		std::vector<double>*);

} // namespace gpusmina
