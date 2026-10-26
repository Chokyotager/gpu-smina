/*
 * Kernel templates; included once per precision by gpu_kernels_f32.cu and
 * gpu_kernels_f64.cu, which explicitly instantiate the host entry points.
 */

#include "gpu_kernels.h"
#include "gpu_eval.cuh"
#include "gpu_mc.cuh"
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

template<typename T>
__device__ ChainView<T> make_chain_view(T* mem, int* imem,
		const ChainOffsets& co)
{
	ChainView<T> cv;
	cv.tmp = mem + co.tmp;
	cv.cand = mem + co.cand;
	cv.coords = mem + co.coords;
	cv.work.x_new = mem + co.x_new;
	cv.work.x_orig = mem + co.x_orig;
	cv.work.g = mem + co.g;
	cv.work.g_new = mem + co.g_new;
	cv.work.g_orig = mem + co.g_orig;
	cv.work.p = mem + co.p;
	cv.work.y = mem + co.y;
	cv.work.minus_hy = mem + co.minus_hy;
	cv.work.h = mem + co.h;
	cv.out_conf = mem + co.out_conf;
	cv.out_e = mem + co.out_e;
	cv.out_coords = mem + co.out_coords;
	cv.order = imem + co.order;
	cv.count = imem + co.count;
	return cv;
}

template<typename T>
__global__ void monte_carlo_kernel(DevModels<T> dm, const int* chain_model,
		int nchains, DevTables<T> tb, InterContext<T> ic, MCParams prm,
		MTState* rngs, T* chain_mem, int* chain_imem, ChainOffsets co,
		T* scratch, ScratchOffsets so, int* error, unsigned* progress)
{
	const int chain = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
	const int lane = threadIdx.x & 31;
	if (chain >= nchains)
		return;
	const ModelView<T> mv = make_view(dm, chain_model[chain]);
	EvalScratch<T> ws = make_scratch(scratch + size_t(chain) * so.total, so,
			error);
	// the task's model starts in the parsed (input) state
	const T* ic0 = dm.init_coords + 3 * mv.d.atom_offset;
	for (int i = lane; i < 3 * mv.d.num_movable; i += 32)
		ws.coords[i] = ic0[i];
	const T* fo0 = dm.init_frame_origin + 3 * mv.d.frame_offset;
	for (int i = lane; i < 3 * mv.d.num_frames; i += 32)
		ws.frame_origin[i] = fo0[i];
	__syncwarp();
	ChainView<T> cv = make_chain_view(chain_mem + size_t(chain) * co.total,
			chain_imem + size_t(chain) * co.int_total, co);
	monte_carlo_chain<T, Ordered<T>::value>(mv, tb, ic, prm, cv, ws,
			rngs[chain], lane, progress);
}

template<typename T>
void run_monte_carlo(const DeviceModels<T>& models,
		const DeviceTables<T>& tables, const InterContext<T>& ic,
		const MCParams& params, const std::vector<ChainInput>& chains,
		std::vector<ChainResult>& results, unsigned* progress)
{
	const int nchains = int(chains.size());
	results.assign(nchains, ChainResult());
	if (nchains == 0)
		return;
	ScratchOffsets so = scratch_layout(models.max_movable, models.max_frames,
			models.max_list, Ordered<T>::value);
	ChainOffsets co = chain_layout(models.max_conf, models.max_change,
			models.max_heavy, params.num_saved_mins);

	std::vector<int> cmodel(nchains);
	std::vector<MTState> rng(nchains);
	std::vector<T> mem(size_t(nchains) * co.total, T(0));
	for (int c = 0; c < nchains; c++)
	{
		cmodel[c] = chains[c].model;
		rng[c] = chains[c].rng;
		for (size_t i = 0; i < chains[c].conf.size(); i++)
			mem[size_t(c) * co.total + co.tmp + i] = T(chains[c].conf[i]);
	}
	DevBuf dmodel, drng, dmem, dimem, dscratch, derr;
	dmodel.upload(&cmodel[0], sizeof(int) * nchains);
	drng.upload(&rng[0], sizeof(MTState) * nchains);
	dmem.upload(&mem[0], sizeof(T) * mem.size());
	dimem.alloc(sizeof(int) * size_t(nchains) * co.int_total);
	dimem.zero();
	dscratch.alloc(sizeof(T) * size_t(nchains) * so.total);
	dscratch.zero();
	int zero = 0;
	derr.upload(&zero, sizeof(int));

	unsigned* dprogress = 0;
	if (progress)
		cuda_check(cudaHostGetDevicePointer((void**) &dprogress, progress, 0),
				"progress pointer");

	const int threads = 128;
	const int blocks = (nchains * 32 + threads - 1) / threads;
	monte_carlo_kernel<T><<<blocks, threads>>>(models.dev, dmodel.get<int>(),
			nchains, tables.dev, ic, params, drng.get<MTState>(), dmem.get<T>(),
			dimem.get<int>(), co, dscratch.get<T>(), so, derr.get<int>(),
			dprogress);
	cuda_check(cudaGetLastError(), "monte_carlo_kernel launch");
	cuda_check(cudaDeviceSynchronize(), "monte_carlo_kernel");

	dmem.download(&mem[0], sizeof(T) * mem.size());
	std::vector<int> imem(size_t(nchains) * co.int_total);
	dimem.download(&imem[0], sizeof(int) * imem.size());
	for (int c = 0; c < nchains; c++)
	{
		const ModelDesc& d = models.host_desc[chains[c].model];
		const T* m = &mem[size_t(c) * co.total];
		const int* im = &imem[size_t(c) * co.int_total];
		const int count = im[co.count];
		ChainResult& r = results[c];
		for (int k = 0; k < count; k++)
		{
			const int slot = im[co.order + k];
			r.e.push_back(m[co.out_e + slot]);
			// slots are strided by this chain's model sizes (see add_to_output)
			const T* oc = m + co.out_conf + size_t(slot) * d.conf_size;
			r.conf.push_back(std::vector<double>(oc, oc + d.conf_size));
			const T* ox = m + co.out_coords + size_t(slot) * 3 * d.num_heavy;
			r.coords.push_back(std::vector<double>(ox, ox + 3 * d.num_heavy));
		}
	}
}

template<typename T>
__global__ void bfgs_kernel(DevModels<T> dm, const int* conf_model, int nconf,
		DevTables<T> tb, InterContext<T> ic, MinParams prm, T v0, T v1, T v2,
		T* chain_mem, int* chain_imem, ChainOffsets co, T* scratch,
		ScratchOffsets so, int* error, T* out_e)
{
	const int c = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
	const int lane = threadIdx.x & 31;
	if (c >= nconf)
		return;
	const ModelView<T> mv = make_view(dm, conf_model[c]);
	EvalScratch<T> ws = make_scratch(scratch + size_t(c) * so.total, so, error);
	ChainView<T> cv = make_chain_view(chain_mem + size_t(c) * co.total,
			chain_imem + size_t(c) * co.int_total, co);
	const T v[3] = {v0, v1, v2};
	const Objective<T> f(mv, tb, ic, v, ws);
	T e = bfgs<T, Ordered<T>::value>(f, cv.tmp, cv.work, prm, lane);
	if (lane == 0)
		out_e[c] = e;
}

template<typename T>
bool run_bfgs(const DeviceModels<T>& models, const DeviceTables<T>& tables,
		const InterContext<T>& ic, const MinParams& params, const double v[3],
		const std::vector<int>& model_of, std::vector<std::vector<double> >& confs,
		std::vector<double>& e)
{
	const int n = int(confs.size());
	e.assign(n, 0);
	if (n == 0)
		return true;
	ScratchOffsets so = scratch_layout(models.max_movable, models.max_frames,
			models.max_list, Ordered<T>::value);
	ChainOffsets co = chain_layout(models.max_conf, models.max_change,
			models.max_heavy, 0);
	std::vector<T> mem(size_t(n) * co.total, T(0));
	for (int c = 0; c < n; c++)
		for (size_t i = 0; i < confs[c].size(); i++)
			mem[size_t(c) * co.total + co.tmp + i] = T(confs[c][i]);
	DevBuf dmodel, dmem, dimem, dscratch, derr, de;
	dmodel.upload(&model_of[0], sizeof(int) * n);
	dmem.upload(&mem[0], sizeof(T) * mem.size());
	dimem.alloc(sizeof(int) * size_t(n) * co.int_total);
	dscratch.alloc(sizeof(T) * size_t(n) * so.total);
	dscratch.zero();
	de.alloc(sizeof(T) * n);
	int zero = 0;
	derr.upload(&zero, sizeof(int));
	const int threads = 128;
	const int blocks = (n * 32 + threads - 1) / threads;
	bfgs_kernel<T><<<blocks, threads>>>(models.dev, dmodel.get<int>(), n,
			tables.dev, ic, params, T(v[0]), T(v[1]), T(v[2]), dmem.get<T>(),
			dimem.get<int>(), co, dscratch.get<T>(), so, derr.get<int>(),
			de.get<T>());
	cuda_check(cudaGetLastError(), "bfgs_kernel launch");
	cuda_check(cudaDeviceSynchronize(), "bfgs_kernel");
	dmem.download(&mem[0], sizeof(T) * mem.size());
	std::vector<T> he(n);
	de.download(&he[0], sizeof(T) * n);
	for (int c = 0; c < n; c++)
	{
		e[c] = he[c];
		for (size_t i = 0; i < confs[c].size(); i++)
			confs[c][i] = mem[size_t(c) * co.total + co.tmp + i];
	}
	int err = 0;
	derr.download(&err, sizeof(int));
	return err == 0;
}

template bool run_bfgs<GPU_PRECISION>(const DeviceModels<GPU_PRECISION>&,
		const DeviceTables<GPU_PRECISION>&, const InterContext<GPU_PRECISION>&,
		const MinParams&, const double[3], const std::vector<int>&,
		std::vector<std::vector<double> >&, std::vector<double>&);

template void run_monte_carlo<GPU_PRECISION>(
		const DeviceModels<GPU_PRECISION>&, const DeviceTables<GPU_PRECISION>&,
		const InterContext<GPU_PRECISION>&, const MCParams&,
		const std::vector<ChainInput>&, std::vector<ChainResult>&, unsigned*);

template bool eval_confs<GPU_PRECISION>(const DeviceModels<GPU_PRECISION>&,
		int, const DeviceTables<GPU_PRECISION>&,
		const InterContext<GPU_PRECISION>&, const double[3],
		const std::vector<double>&, std::vector<double>&, std::vector<double>&,
		std::vector<double>*);

} // namespace gpusmina
