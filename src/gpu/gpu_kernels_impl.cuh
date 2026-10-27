/*
 * Kernel templates; included once per precision by gpu_kernels_f32.cu and
 * gpu_kernels_f64.cu, which explicitly instantiate the host entry points.
 *
 * All kernels share one organization: work items (conformations or Monte
 * Carlo chains) are grouped into blocks of BLOCK_THREADS / TPC items of the
 * same model, one thread group of TPC threads per item.
 *
 * Dynamic shared memory of a block, in order:
 *   ModelView of the block's model          (always)
 *   per group: descriptors + reduction pad  (always)
 *   the model's static data                 (if it fits: stage_model)
 *   per group: hot state                    (if it fits: hot_shared)
 *   per group: pair buffer                  (if it fits: pair_shared)
 * Whatever does not fit lives in global memory.
 */

#include "gpu_kernels.h"
#include "gpu_eval.cuh"
#include "gpu_mc.cuh"
#include <cuda_runtime.h>
#include <type_traits>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace gpusmina
{

template<typename T>
struct Ordered
{
	static const bool value = std::is_same<T, double>::value;
};

const int BLOCK_THREADS = 256;

// Resident blocks per SM requested for throughput launches (TPC 32); bounds
// registers and the shared memory budget accordingly.
#ifndef GPU_MIN_BLOCKS
#define GPU_MIN_BLOCKS 3
#endif
template<int TPC> struct MinBlocks
{
	static const int value = (TPC == 32) ? GPU_MIN_BLOCKS : 1;
};

// a block of work items sharing one model
struct BlockTask
{
	int model;
	int first; // first item
	int count; // items (groups) used in the block
};

// per-group descriptors kept in shared memory
template<typename T>
struct GroupDesc
{
	EvalScratch<T> ws;
	ChainView<T> cv;
	T* mem; // the item's global chain block (if any)
};

__host__ __device__ inline size_t align16(size_t x)
{
	return (x + 15) & ~size_t(15);
}

// shared memory plan of a launch
struct SharedPlan
{
	size_t view_off, desc_off, red_off, model_off, hot_off, pair_off, total;
	int stage_model, hot_shared, pair_shared;
	int hot_elems; // per group, elements of T
	int pair_elems; // per group, elements of T
};

static int shared_optin()
{
	static int optin = -1;
	if (optin < 0)
	{
		int dev = 0;
		cudaGetDevice(&dev);
		cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
				dev);
	}
	return optin;
}

// Throughput launches (TPC 32) keep a block within 1 / MinBlocks of an SM's
// shared memory; latency launches may use all of it.
template<typename T, int TPC>
SharedPlan plan_shared(const void* kernel, const DeviceModels<T>& models,
		const std::vector<BlockTask>& tasks, int hot_elems, int pair_elems)
{
	const int groups = BLOCK_THREADS / TPC;
	size_t model_bytes = 0;
	for (size_t i = 0; i < tasks.size(); i++)
		model_bytes = std::max(model_bytes,
				staged_layout<T>(models.host_desc[tasks[i].model]).bytes);
	const size_t optin = size_t(shared_optin());
	const size_t budget = (TPC == 32) ? std::min(optin,
			size_t(100 * 1024 / MinBlocks<TPC>::value - 1024)) : optin;

	SharedPlan p;
	p.hot_elems = hot_elems;
	p.pair_elems = pair_elems;
	p.view_off = 0;
	p.desc_off = align16(sizeof(ModelView<T>));
	p.red_off = align16(p.desc_off + groups * sizeof(GroupDesc<T>));
	const size_t fixed = align16(p.red_off + groups * GROUP_SCRATCH_BYTES);
	const size_t hot = align16(size_t(hot_elems) * sizeof(T)) * groups;
	const size_t pair = align16(size_t(pair_elems) * sizeof(T)) * groups;
	// priority: hot state, then the staged model, then pair buffers
	p.hot_shared = fixed + hot <= budget;
	const size_t after_hot = fixed + (p.hot_shared ? hot : 0);
	p.stage_model = after_hot + model_bytes <= budget;
	const size_t after_model = after_hot + (p.stage_model ? model_bytes : 0);
	p.pair_shared = pair_elems > 0 && after_model + pair <= budget;
	p.model_off = fixed;
	p.hot_off = p.model_off + (p.stage_model ? model_bytes : 0);
	p.pair_off = p.hot_off + (p.hot_shared ? hot : 0);
	p.total = p.pair_off + (p.pair_shared ? pair : 0);
#ifdef GPU_PROFILE
	printf("shared plan: budget %zu, fixed %zu, hot %zu (%s), model %zu (%s),"
			" pairs %zu (%s)\n", budget, fixed, hot, p.hot_shared ? "shared"
			: "global", model_bytes, p.stage_model ? "shared" : "global", pair,
			p.pair_shared ? "shared" : "global");
#endif
	if (p.total > 48 * 1024)
		cuda_check(cudaFuncSetAttribute(kernel,
				cudaFuncAttributeMaxDynamicSharedMemorySize, int(p.total)),
				"shared memory opt-in");
	return p;
}

// Blocks are dispatched in index order as SMs become free; ordering them by
// decreasing estimated cost (longest first) shortens the tail of a launch
// whose items differ in length.  cost(item) is any monotone estimate.
static void order_by_cost(std::vector<BlockTask>& tasks,
		const std::vector<double>& item_cost)
{
	std::vector<std::pair<double, int> > order;
	for (size_t t = 0; t < tasks.size(); t++)
	{
		double c = 0;
		for (int i = 0; i < tasks[t].count; i++)
			c = std::max(c, item_cost[tasks[t].first + i]);
		order.push_back(std::make_pair(-c, int(t)));
	}
	std::stable_sort(order.begin(), order.end());
	std::vector<BlockTask> sorted;
	for (size_t t = 0; t < order.size(); t++)
		sorted.push_back(tasks[order[t].second]);
	tasks.swap(sorted);
}

// work of one evaluation of a model
static double eval_cost(const ModelDesc& d)
{
	return double(d.num_pairs) + 8.0 * d.num_movable + 4.0 * d.num_frames;
}

// items -> blocks of at most `per_block` consecutive items of one model
static std::vector<BlockTask> make_tasks(const std::vector<int>& item_model,
		int per_block)
{
	std::vector<BlockTask> tasks;
	for (int i = 0; i < int(item_model.size()); i++)
	{
		if (tasks.empty() || tasks.back().model != item_model[i]
				|| tasks.back().count == per_block)
		{
			BlockTask t;
			t.model = item_model[i];
			t.first = i;
			t.count = 0;
			tasks.push_back(t);
		}
		tasks.back().count++;
	}
	return tasks;
}

// global memory used by items whose state does not fit in shared memory
template<typename T>
struct GlobalScratch
{
	T* hot; // plan.hot_elems per item
	T* pair; // plan.pair_elems per item
};

// Per-block setup: the model view (staged when planned) and this group's
// descriptors, all in shared memory.  Returns false for groups without an
// item; they still take part in staging.
template<typename T, int TPC>
__device__ bool block_setup(const DevModels<T>& dm, const BlockTask* tasks,
		const SharedPlan& plan, const ScratchOffsets& so,
		const GlobalScratch<T>& gs, int* error, Group<TPC>& G, int& item,
		ModelView<T>*& mv, GroupDesc<T>*& gd)
{
	extern __shared__ __align__(16) unsigned char smem_raw[];
	const BlockTask t = tasks[blockIdx.x];
	mv = reinterpret_cast<ModelView<T>*>(smem_raw + plan.view_off);
	if (threadIdx.x == 0)
		*mv = make_view(dm, t.model);
	__syncthreads();
	if (plan.stage_model)
	{
		const ModelView<T> g = *mv;
		ModelView<T> s = stage_model(g, smem_raw + plan.model_off);
		__syncthreads();
		if (threadIdx.x == 0)
			*mv = s;
		__syncthreads();
	}
	G.gid = threadIdx.x / TPC;
	G.rank = threadIdx.x % TPC;
	G.red = reinterpret_cast<double*>(smem_raw + plan.red_off
			+ GROUP_SCRATCH_BYTES * G.gid);
	item = t.first + G.gid;
	gd = reinterpret_cast<GroupDesc<T>*>(smem_raw + plan.desc_off) + G.gid;
	if (G.gid >= t.count)
		return false;
	if (G.rank == 0)
	{
		T* hot = plan.hot_shared ? reinterpret_cast<T*>(smem_raw + plan.hot_off
				+ align16(size_t(plan.hot_elems) * sizeof(T)) * G.gid)
				: gs.hot + size_t(item) * plan.hot_elems;
		T* pair = plan.pair_shared ? reinterpret_cast<T*>(smem_raw
				+ plan.pair_off + align16(size_t(plan.pair_elems) * sizeof(T))
						* G.gid) : gs.pair + size_t(item) * plan.pair_elems;
		gd->ws = make_scratch(hot, pair, so, error);
	}
	G.sync();
	return true;
}

template<typename T>
static void alloc_global_scratch(const SharedPlan& plan, size_t items,
		DevBuf& hot, DevBuf& pair, GlobalScratch<T>& gs)
{
	gs.hot = gs.pair = 0;
	if (!plan.hot_shared)
	{
		hot.alloc(sizeof(T) * items * plan.hot_elems);
		hot.zero();
		gs.hot = hot.get<T>();
	}
	if (!plan.pair_shared && plan.pair_elems > 0)
	{
		pair.alloc(sizeof(T) * items * plan.pair_elems);
		gs.pair = pair.get<T>();
	}
}

////////////////////////////// eval_confs //////////////////////////////

template<typename T, int TPC>
__global__ void __launch_bounds__(BLOCK_THREADS, MinBlocks<TPC>::value) eval_confs_kernel(
		const __grid_constant__ DevModels<T> dm, const BlockTask* tasks,
		const __grid_constant__ SharedPlan plan,
		const __grid_constant__ DevTables<T> tb,
		const __grid_constant__ InterContext<T> ic, T v0, T v1, T v2,
		const T* confs, T* out_e, T* out_g, T* out_coords,
		const __grid_constant__ GlobalScratch<T> gs,
		const __grid_constant__ ScratchOffsets so, int* error)
{
	Group<TPC> G;
	int c;
	ModelView<T>* mvp;
	GroupDesc<T>* gd;
	if (!block_setup(dm, tasks, plan, so, gs, error, G, c, mvp, gd))
		return;
	const ModelView<T>& mv = *mvp;
	const T v[3] = {v0, v1, v2};
	T* g = out_g + size_t(c) * mv.d.change_size;
	T e = eval_deriv<T, Ordered<T>::value>(mv, tb, ic, v, ic.slope, ic.box,
			confs + size_t(c) * mv.d.conf_size, g, gd->ws, G);
	if (G.rank == 0)
		out_e[c] = e;
	if (out_coords)
		for (int i = G.rank; i < 3 * mv.d.num_movable; i += TPC)
			out_coords[size_t(c) * 3 * mv.d.num_movable + i] = gd->ws.coords[i];
}

template<typename T, int TPC>
static void launch_eval_confs(const DeviceModels<T>& models, int nconf,
		int model, const DeviceTables<T>& tables, const InterContext<T>& ic,
		const double v[3], const ScratchOffsets& so, DevBuf& dconfs, DevBuf& de,
		DevBuf& dg, DevBuf* dcoords, DevBuf& derr)
{
	std::vector<BlockTask> tasks = make_tasks(std::vector<int>(nconf, model),
			BLOCK_THREADS / TPC);
	SharedPlan plan = plan_shared<T, TPC>((const void*) eval_confs_kernel<T, TPC>,
			models, tasks, so.total, so.pair_total);
	DevBuf dtasks, hot, pair;
	GlobalScratch<T> gs;
	alloc_global_scratch(plan, nconf, hot, pair, gs);
	dtasks.upload(&tasks[0], sizeof(BlockTask) * tasks.size());
	eval_confs_kernel<T, TPC><<<int(tasks.size()), BLOCK_THREADS, plan.total>>>(
			models.dev, dtasks.get<BlockTask>(), plan, tables.dev, ic, T(v[0]),
			T(v[1]), T(v[2]), dconfs.get<T>(), de.get<T>(), dg.get<T>(),
			dcoords ? dcoords->get<T>() : (T*) 0, gs, so, derr.get<int>());
	cuda_check(cudaGetLastError(), "eval_confs_kernel launch");
	cuda_check(cudaDeviceSynchronize(), "eval_confs_kernel");
}

template<typename T>
bool eval_confs(const DeviceModels<T>& models, int model,
		const DeviceTables<T>& tables, const InterContext<T>& ic,
		const double v[3], const std::vector<double>& confs,
		std::vector<double>& e, std::vector<double>& g,
		std::vector<double>* coords, int threads_per_chain)
{
	const ModelDesc& d = models.host_desc[model];
	const int nconf = int(confs.size() / d.conf_size);
	ScratchOffsets so = scratch_layout(models.max_movable, models.max_frames,
			models.max_list, models.max_pairs, Ordered<T>::value);
	DevBuf dconfs, de, dg, derr, dcoords;
	if (coords)
		dcoords.alloc(sizeof(T) * size_t(nconf) * 3 * d.num_movable);
	upload_converted<T>(dconfs, confs);
	de.alloc(sizeof(T) * nconf);
	dg.alloc(sizeof(T) * size_t(nconf) * d.change_size);
	int zero = 0;
	derr.upload(&zero, sizeof(int));
	DevBuf* pc = coords ? &dcoords : 0;
	if (threads_per_chain >= 256)
		launch_eval_confs<T, 256>(models, nconf, model, tables, ic, v, so, dconfs,
				de, dg, pc, derr);
	else if (threads_per_chain >= 128)
		launch_eval_confs<T, 128>(models, nconf, model, tables, ic, v, so, dconfs,
				de, dg, pc, derr);
	else if (threads_per_chain >= 64)
		launch_eval_confs<T, 64>(models, nconf, model, tables, ic, v, so, dconfs,
				de, dg, pc, derr);
	else
		launch_eval_confs<T, 32>(models, nconf, model, tables, ic, v, so, dconfs,
				de, dg, pc, derr);

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

////////////////////////////// chains //////////////////////////////

// the hot part of a chain's state (offsets below co.out_conf) is at `hot`,
// which is either the chain's global block or shared memory
template<typename T>
__device__ ChainView<T> make_chain_view(T* hot, T* mem, int* imem,
		const ChainOffsets& co)
{
	ChainView<T> cv;
	cv.tmp = hot + co.tmp;
	cv.cand = hot + co.cand;
	cv.coords = hot + co.coords;
	cv.work.x_new = hot + co.x_new;
	cv.work.x_orig = hot + co.x_orig;
	cv.work.g = hot + co.g;
	cv.work.g_new = hot + co.g_new;
	cv.work.g_orig = hot + co.g_orig;
	cv.work.p = hot + co.p;
	cv.work.y = hot + co.y;
	cv.work.minus_hy = hot + co.minus_hy;
	cv.work.h = hot + co.h;
	cv.out_conf = mem + co.out_conf;
	cv.out_e = mem + co.out_e;
	cv.out_coords = mem + co.out_coords;
	cv.order = imem + co.order;
	cv.count = imem + co.count;
	return cv;
}

// chain descriptors for an item; the hot chain state follows the eval
// scratch in the group's hot region
template<typename T, int TPC>
__device__ void setup_chain(GroupDesc<T>* gd, const SharedPlan& plan,
		const ScratchOffsets& so, const ChainOffsets& co, T* chain_mem,
		int* chain_imem, int item, const Group<TPC>& G)
{
	if (G.rank == 0)
	{
		T* mem = chain_mem + size_t(item) * co.total;
		T* hot = plan.hot_shared ? gd->ws.coords + so.total : mem;
		gd->cv = make_chain_view(hot, mem,
				chain_imem + size_t(item) * co.int_total, co);
		gd->mem = mem;
	}
	G.sync();
}

#ifdef GPU_PROFILE
static void report_profile()
{
	unsigned long long prof[PROF_COUNT];
	cudaMemcpyFromSymbol(prof, g_prof, sizeof(prof));
	const char* names[PROF_COUNT] = {"set_conf", " set atoms", " set frames", "inter", "pairs",
			" pair pass1", " pair pass2", "deriv", "eval_total", "bfgs_total",
			"mutate", "output", "mc_total", "evals"};
	double evals = double(prof[PROF_EVALS]);
	printf("profile (warp 0): %llu evals, %.0f cycles/eval\n", prof[PROF_EVALS],
			prof[PROF_EVAL_TOTAL] / evals);
	for (int i = 0; i < PROF_EVALS; i++)
		printf("  %-11s %6.1f%% of mc  %10.0f cycles/eval\n", names[i],
				100.0 * prof[i] / prof[PROF_MC_TOTAL], prof[i] / evals);
	unsigned long long zero[PROF_COUNT] = {0};
	cudaMemcpyToSymbol(g_prof, zero, sizeof(zero));
}
#endif

__device__ unsigned long long* g_chain_stats;

template<typename T, int TPC>
__global__ void __launch_bounds__(BLOCK_THREADS, MinBlocks<TPC>::value) monte_carlo_kernel(
		const __grid_constant__ DevModels<T> dm, const BlockTask* tasks,
		const __grid_constant__ SharedPlan plan,
		const __grid_constant__ DevTables<T> tb,
		const __grid_constant__ InterContext<T> ic,
		const __grid_constant__ MCParams prm, MTState* rngs, T* chain_mem,
		int* chain_imem, const __grid_constant__ ChainOffsets co,
		const __grid_constant__ GlobalScratch<T> gs,
		const __grid_constant__ ScratchOffsets so, int* error,
		unsigned* progress, const uint2* chain_limits)
{
	Group<TPC> G;
	int chain;
	ModelView<T>* mvp;
	GroupDesc<T>* gd;
	if (!block_setup(dm, tasks, plan, so, gs, error, G, chain, mvp, gd))
		return;
	const ModelView<T>& mv = *mvp;
	setup_chain(gd, plan, so, co, chain_mem, chain_imem, chain, G);
	if (plan.hot_shared)
		copy_n(gd->cv.tmp, gd->mem + co.tmp, mv.d.conf_size, G);
	// the task's model starts in the parsed (input) state
	const T* ic0 = dm.init_coords + 3 * mv.d.atom_offset;
	for (int i = G.rank; i < 3 * mv.d.num_movable; i += TPC)
		gd->ws.coords[i] = ic0[i];
	const T* fo0 = dm.init_frame_origin + 3 * mv.d.frame_offset;
	for (int i = G.rank; i < 3 * mv.d.num_frames; i += TPC)
		gd->ws.frame_origin[i] = fo0[i];
	G.sync();
	if (g_chain_stats && G.rank == 0)
	{
		unsigned long long t;
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
		g_chain_stats[4 * chain + 3] = t;
	}
	MCParams p = prm;
	p.num_steps = chain_limits[chain].x;
	p.min.maxiters = chain_limits[chain].y;
	monte_carlo_chain<T, Ordered<T>::value>(mv, tb, ic, p, gd->cv, gd->ws,
			rngs[chain], G, progress);
	if (g_chain_stats && G.rank == 0)
	{
		unsigned long long t;
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
		g_chain_stats[4 * chain] = t;
		g_chain_stats[4 * chain + 1] =
				g_group_evals[blockIdx.x * (blockDim.x / TPC) + G.gid];
		g_chain_stats[4 * chain + 2] = blockIdx.x;
	}
}

template<typename T, int TPC>
static void launch_monte_carlo(const DeviceModels<T>& models,
		const std::vector<int>& cmodel, const DeviceTables<T>& tables,
		const InterContext<T>& ic, const MCParams& params,
		const ScratchOffsets& so, const ChainOffsets& co, DevBuf& drng,
		DevBuf& dmem, DevBuf& dimem, DevBuf& derr, unsigned* dprogress,
		DevBuf& dlimits, const std::vector<double>& cost)
{
	std::vector<BlockTask> tasks = make_tasks(cmodel, BLOCK_THREADS / TPC);
	order_by_cost(tasks, cost);
	SharedPlan plan = plan_shared<T, TPC>(
			(const void*) monte_carlo_kernel<T, TPC>, models, tasks,
			so.total + co.out_conf, so.pair_total);
	DevBuf dtasks, hot, pair;
	GlobalScratch<T> gs;
	alloc_global_scratch(plan, cmodel.size(), hot, pair, gs);
	dtasks.upload(&tasks[0], sizeof(BlockTask) * tasks.size());
	monte_carlo_kernel<T, TPC><<<int(tasks.size()), BLOCK_THREADS, plan.total>>>(
			models.dev, dtasks.get<BlockTask>(), plan, tables.dev, ic, params,
			drng.get<MTState>(), dmem.get<T>(), dimem.get<int>(), co, gs, so,
			derr.get<int>(), dprogress, dlimits.get<uint2>());
	cuda_check(cudaGetLastError(), "monte_carlo_kernel launch");
	cuda_check(cudaDeviceSynchronize(), "monte_carlo_kernel");
}

template<typename T>
void run_monte_carlo(const DeviceModels<T>& models,
		const DeviceTables<T>& tables, const InterContext<T>& ic,
		const MCParams& params, const std::vector<ChainInput>& chains,
		std::vector<ChainResult>& results, unsigned* progress,
		int threads_per_chain)
{
	const int nchains = int(chains.size());
	results.assign(nchains, ChainResult());
	if (nchains == 0)
		return;
	ScratchOffsets so = scratch_layout(models.max_movable, models.max_frames,
			models.max_list, models.max_pairs, Ordered<T>::value);
	ChainOffsets co = chain_layout(models.max_conf, models.max_change,
			models.max_heavy, params.num_saved_mins);

	std::vector<int> cmodel(nchains);
	std::vector<MTState> rng(nchains);
	std::vector<uint2> limits(nchains);
	std::vector<double> cost(nchains);
	std::vector<T> mem(size_t(nchains) * co.total, T(0));
	for (int c = 0; c < nchains; c++)
	{
		cmodel[c] = chains[c].model;
		rng[c] = chains[c].rng;
		limits[c].x = chains[c].num_steps ? chains[c].num_steps
				: params.num_steps;
		limits[c].y = chains[c].maxiters ? chains[c].maxiters
				: params.min.maxiters;
		cost[c] = double(limits[c].x) * limits[c].y
				* eval_cost(models.host_desc[cmodel[c]]);
		for (size_t i = 0; i < chains[c].conf.size(); i++)
			mem[size_t(c) * co.total + co.tmp + i] = T(chains[c].conf[i]);
	}
	DevBuf drng, dmem, dimem, derr, dlimits;
	drng.upload(&rng[0], sizeof(MTState) * nchains);
	dlimits.upload(&limits[0], sizeof(uint2) * nchains);
	dmem.upload(&mem[0], sizeof(T) * mem.size());
	dimem.alloc(sizeof(int) * size_t(nchains) * co.int_total);
	dimem.zero();
	int zero = 0;
	derr.upload(&zero, sizeof(int));

	DevBuf dstats, dgroup;
	const bool chain_stats = getenv("GPU_CHAIN_STATS") != 0;
	unsigned long long t0 = 0;
	if (chain_stats)
	{
		dstats.alloc(sizeof(unsigned long long) * 4 * nchains);
		dgroup.alloc(sizeof(unsigned long long) * 8 * nchains + 4096);
		dgroup.zero();
		unsigned long long* ps = dstats.get<unsigned long long>();
		unsigned long long* pg = dgroup.get<unsigned long long>();
		cudaMemcpyToSymbol(g_chain_stats, &ps, sizeof(ps));
		cudaMemcpyToSymbol(g_group_evals, &pg, sizeof(pg));
		t0 = 0;
	}
	unsigned* dprogress = 0;
	if (progress)
		cuda_check(cudaHostGetDevicePointer((void**) &dprogress, progress, 0),
				"progress pointer");

	if (threads_per_chain >= 256)
		launch_monte_carlo<T, 256>(models, cmodel, tables, ic, params, so, co,
				drng, dmem, dimem, derr, dprogress, dlimits, cost);
	else if (threads_per_chain >= 128)
		launch_monte_carlo<T, 128>(models, cmodel, tables, ic, params, so, co,
				drng, dmem, dimem, derr, dprogress, dlimits, cost);
	else if (threads_per_chain >= 64)
		launch_monte_carlo<T, 64>(models, cmodel, tables, ic, params, so, co,
				drng, dmem, dimem, derr, dprogress, dlimits, cost);
	else
		launch_monte_carlo<T, 32>(models, cmodel, tables, ic, params, so, co,
				drng, dmem, dimem, derr, dprogress, dlimits, cost);
#ifdef GPU_PROFILE
	report_profile();
#endif
	if (chain_stats)
	{
		std::vector<unsigned long long> st(4 * nchains);
		dstats.download(&st[0], sizeof(unsigned long long) * st.size());
		unsigned long long first = ~0ULL;
		for (int c = 0; c < nchains; c++)
			first = std::min(first, st[4 * c]);
		std::vector<std::pair<unsigned long long, int> > order;
		for (int c = 0; c < nchains; c++)
			order.push_back(std::make_pair(st[4 * c], c));
		std::sort(order.begin(), order.end());
		fprintf(stderr, "chains by end time (ms after first finished):");
		for (int i = nchains - 1; i >= 0 && i >= nchains - 8; i--)
		{
			const int c = order[i].second;
			fprintf(stderr, " [chain %d model %d: %.0f ms, %llu evals, block %llu]",
					c, chains[c].model, (st[4 * c] - first) * 1e-6, st[4 * c + 1],
					st[4 * c + 2]);
		}
		unsigned long long tot = 0;
		double us = 0;
		for (int c = 0; c < nchains; c++)
		{
			tot += st[4 * c + 1];
			us += (st[4 * c] - st[4 * c + 3]) * 1e-3 / double(st[4 * c + 1]);
		}
		fprintf(stderr, "\nmean evals per chain %.0f, mean %.2f us per eval\n",
				double(tot) / nchains, us / nchains);
		unsigned long long* z = 0;
		cudaMemcpyToSymbol(g_chain_stats, &z, sizeof(z));
		cudaMemcpyToSymbol(g_group_evals, &z, sizeof(z));
	}

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

////////////////////////////// bfgs //////////////////////////////

static __global__ void stamp_kernel(unsigned long long* t)
{
	unsigned long long v;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
	*t = v;
}

template<typename T>
static void report_item_times(DevBuf& dend, int n, unsigned long long t0,
		const std::vector<int>& model_of, const DeviceModels<T>& models)
{
	std::vector<unsigned long long> end(n);
	dend.download(&end[0], sizeof(unsigned long long) * n);
	std::vector<std::pair<double, int> > t(n);
	for (int c = 0; c < n; c++)
		t[c] = std::make_pair((end[c] - t0) * 1e-6, c);
	std::sort(t.begin(), t.end());
	const double q[] = {0.5, 0.9, 0.99, 0.999};
	fprintf(stderr, "item end times (ms after launch), %d items:", n);
	for (int i = 0; i < 4; i++)
		fprintf(stderr, " p%g %.1f", q[i] * 100, t[size_t(q[i] * (n - 1))].first);
	fprintf(stderr, " max %.1f\nslowest:", t[n - 1].first);
	for (int i = n - 1; i >= 0 && i >= n - 10; i--)
	{
		const ModelDesc& d = models.host_desc[model_of[t[i].second]];
		fprintf(stderr, " #%d(%.0fms,%d atoms,%d pairs)", t[i].second,
				t[i].first, d.num_movable, d.num_pairs);
	}
	fprintf(stderr, "\n");
}

template<typename T, int TPC>
__global__ void __launch_bounds__(BLOCK_THREADS, MinBlocks<TPC>::value) bfgs_kernel(
		const __grid_constant__ DevModels<T> dm, const BlockTask* tasks,
		const __grid_constant__ SharedPlan plan,
		const __grid_constant__ DevTables<T> tb,
		const __grid_constant__ InterContext<T> ic,
		const __grid_constant__ MinParams prm, T v0, T v1, T v2, T* chain_mem,
		int* chain_imem, const __grid_constant__ ChainOffsets co,
		const __grid_constant__ GlobalScratch<T> gs,
		const __grid_constant__ ScratchOffsets so, int* error, T* out_e,
		int refine, T* out_coords, int max_heavy, const unsigned* item_maxiters,
		const DevBox<T>* item_box, unsigned long long* item_end)
{
	Group<TPC> G;
	int c;
	ModelView<T>* mvp;
	GroupDesc<T>* gd;
	if (!block_setup(dm, tasks, plan, so, gs, error, G, c, mvp, gd))
		return;
	const ModelView<T>& mv = *mvp;
	setup_chain(gd, plan, so, co, chain_mem, chain_imem, c, G);
	if (plan.hot_shared)
		copy_n(gd->cv.tmp, gd->mem + co.tmp, mv.d.conf_size, G);
	const T v[3] = {v0, v1, v2};
	MinParams p = prm;
	if (item_maxiters)
		p.maxiters = item_maxiters[c];
	const DevBox<T> box = item_box ? item_box[c] : ic.box;
	T e;
	if (refine)
		e = refine_structure<T, Ordered<T>::value>(mv, tb, ic, box, v,
				gd->cv.tmp, gd->cv.work, p, gd->ws, gd->cv.coords, G);
	else
	{
		const Objective<T> f(mv, tb, ic, v, ic.slope, box, gd->ws);
		e = bfgs<T, Ordered<T>::value>(f, gd->cv.tmp, gd->cv.work, p, G);
	}
	if (refine)
		for (int i = G.rank; i < 3 * mv.d.num_heavy; i += TPC)
			out_coords[size_t(c) * 3 * max_heavy + i] = gd->cv.coords[i];
	if (plan.hot_shared)
		copy_n(gd->mem + co.tmp, gd->cv.tmp, mv.d.conf_size, G);
	if (G.rank == 0)
	{
		out_e[c] = e;
		if (item_end)
		{
			unsigned long long t;
			asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
			item_end[c] = t;
		}
	}
}

template<typename T, int TPC>
static void launch_bfgs(const DeviceModels<T>& models,
		const std::vector<int>& model_of, const DeviceTables<T>& tables,
		const InterContext<T>& ic, const MinParams& params, const double v[3],
		const ScratchOffsets& so, const ChainOffsets& co, DevBuf& dmem,
		DevBuf& dimem, DevBuf& derr, DevBuf& de, int refine, DevBuf& dcoords,
		const unsigned* dmaxiters, const DevBox<T>* dboxes,
		unsigned long long* ditem_end)
{
	std::vector<BlockTask> tasks = make_tasks(model_of, BLOCK_THREADS / TPC);
	SharedPlan plan = plan_shared<T, TPC>((const void*) bfgs_kernel<T, TPC>,
			models, tasks, so.total + co.out_conf, so.pair_total);
	DevBuf dtasks, hot, pair;
	GlobalScratch<T> gs;
	alloc_global_scratch(plan, model_of.size(), hot, pair, gs);
	dtasks.upload(&tasks[0], sizeof(BlockTask) * tasks.size());
	bfgs_kernel<T, TPC><<<int(tasks.size()), BLOCK_THREADS, plan.total>>>(
			models.dev, dtasks.get<BlockTask>(), plan, tables.dev, ic, params,
			T(v[0]), T(v[1]), T(v[2]), dmem.get<T>(), dimem.get<int>(), co, gs, so,
			derr.get<int>(), de.get<T>(), refine, dcoords.get<T>(),
			models.max_heavy, dmaxiters, dboxes, ditem_end);
	cuda_check(cudaGetLastError(), "bfgs_kernel launch");
	cuda_check(cudaDeviceSynchronize(), "bfgs_kernel");
}

template<typename T>
static bool run_minimizer(const DeviceModels<T>& models,
		const DeviceTables<T>& tables, const InterContext<T>& ic,
		const MinParams& params, const double v[3],
		const std::vector<int>& model_of, std::vector<std::vector<double> >& confs,
		std::vector<double>& e, int threads_per_chain, int refine,
		std::vector<std::vector<double> >* coords,
		const std::vector<unsigned>& maxiters,
		const std::vector<FlatGridGeometry>& boxes)
{
	const int n = int(confs.size());
	e.assign(n, 0);
	if (n == 0)
		return true;
	ScratchOffsets so = scratch_layout(models.max_movable, models.max_frames,
			models.max_list, models.max_pairs, Ordered<T>::value);
	ChainOffsets co = chain_layout(models.max_conf, models.max_change,
			models.max_heavy, 0);
	std::vector<T> mem(size_t(n) * co.total, T(0));
	for (int c = 0; c < n; c++)
		for (size_t i = 0; i < confs[c].size(); i++)
			mem[size_t(c) * co.total + co.tmp + i] = T(confs[c][i]);
	DevBuf dmem, dimem, derr, de, dcoords, dmaxiters;
	if (!maxiters.empty())
		dmaxiters.upload(&maxiters[0], sizeof(unsigned) * n);
	const unsigned* pmax = maxiters.empty() ? (const unsigned*) 0
			: dmaxiters.get<unsigned>();
	DevBuf dboxes;
	if (!boxes.empty())
	{
		std::vector<DevBox<T> > b;
		for (int c = 0; c < n; c++)
			b.push_back(make_box<T>(boxes[c]));
		dboxes.upload(&b[0], sizeof(DevBox<T>) * n);
	}
	const DevBox<T>* pbox = boxes.empty() ? (const DevBox<T>*) 0
			: dboxes.get<DevBox<T> >();
	dmem.upload(&mem[0], sizeof(T) * mem.size());
	dimem.alloc(sizeof(int) * size_t(n) * co.int_total);
	de.alloc(sizeof(T) * n);
	if (refine)
		dcoords.alloc(sizeof(T) * size_t(n) * 3 * models.max_heavy);
	int zero = 0;
	derr.upload(&zero, sizeof(int));
	// GPU_ITEM_TIMES=1: report when each item finished (tail diagnostics)
	const bool item_times = getenv("GPU_ITEM_TIMES") != 0;
	DevBuf dend;
	if (item_times)
		dend.alloc(sizeof(unsigned long long) * n);
	unsigned long long* pend = item_times ? dend.get<unsigned long long>()
			: (unsigned long long*) 0;
	unsigned long long t_launch = 0;
	if (item_times)
	{
		// globaltimer of the launch: a trivial kernel's stamp
		unsigned long long* p0 = pend;
		stamp_kernel<<<1, 1>>>(p0);
		cuda_check(cudaMemcpy(&t_launch, p0, sizeof(t_launch),
				cudaMemcpyDeviceToHost), "stamp");
	}
	if (threads_per_chain >= 256)
		launch_bfgs<T, 256>(models, model_of, tables, ic, params, v, so, co, dmem,
				dimem, derr, de, refine, dcoords, pmax, pbox, pend);
	else if (threads_per_chain >= 128)
		launch_bfgs<T, 128>(models, model_of, tables, ic, params, v, so, co, dmem,
				dimem, derr, de, refine, dcoords, pmax, pbox, pend);
	else if (threads_per_chain >= 64)
		launch_bfgs<T, 64>(models, model_of, tables, ic, params, v, so, co, dmem,
				dimem, derr, de, refine, dcoords, pmax, pbox, pend);
	else
		launch_bfgs<T, 32>(models, model_of, tables, ic, params, v, so, co, dmem,
				dimem, derr, de, refine, dcoords, pmax, pbox, pend);
	dmem.download(&mem[0], sizeof(T) * mem.size());
	std::vector<T> he(n);
	de.download(&he[0], sizeof(T) * n);
	std::vector<T> hc;
	if (refine)
	{
		hc.resize(size_t(n) * 3 * models.max_heavy);
		dcoords.download(&hc[0], sizeof(T) * hc.size());
		coords->assign(n, std::vector<double>());
	}
	for (int c = 0; c < n; c++)
	{
		e[c] = he[c];
		for (size_t i = 0; i < confs[c].size(); i++)
			confs[c][i] = mem[size_t(c) * co.total + co.tmp + i];
		if (refine)
		{
			const ModelDesc& d = models.host_desc[model_of[c]];
			const T* src = &hc[size_t(c) * 3 * models.max_heavy];
			(*coords)[c].assign(src, src + 3 * d.num_heavy);
		}
	}
	if (item_times)
		report_item_times(dend, n, t_launch, model_of, models);
	int err = 0;
	derr.download(&err, sizeof(int));
	return err == 0;
}

template<typename T>
bool run_bfgs(const DeviceModels<T>& models, const DeviceTables<T>& tables,
		const InterContext<T>& ic, const MinParams& params, const double v[3],
		const std::vector<int>& model_of, std::vector<std::vector<double> >& confs,
		std::vector<double>& e, int threads_per_chain)
{
	return run_minimizer(models, tables, ic, params, v, model_of, confs, e,
			threads_per_chain, 0, (std::vector<std::vector<double> >*) 0,
			std::vector<unsigned>(), std::vector<FlatGridGeometry>());
}

template<typename T>
bool run_refine(const DeviceModels<T>& models, const DeviceTables<T>& tables,
		const InterContext<T>& ic, const MinParams& params, const double cap[3],
		const std::vector<int>& model_of, std::vector<std::vector<double> >& confs,
		std::vector<double>& e, std::vector<std::vector<double> >& coords,
		int threads_per_chain, const std::vector<unsigned>& maxiters,
		const std::vector<FlatGridGeometry>& boxes)
{
	return run_minimizer(models, tables, ic, params, cap, model_of, confs, e,
			threads_per_chain, 1, &coords, maxiters, boxes);
}

template bool eval_confs<GPU_PRECISION>(const DeviceModels<GPU_PRECISION>&,
		int, const DeviceTables<GPU_PRECISION>&,
		const InterContext<GPU_PRECISION>&, const double[3],
		const std::vector<double>&, std::vector<double>&, std::vector<double>&,
		std::vector<double>*, int);
template void run_monte_carlo<GPU_PRECISION>(
		const DeviceModels<GPU_PRECISION>&, const DeviceTables<GPU_PRECISION>&,
		const InterContext<GPU_PRECISION>&, const MCParams&,
		const std::vector<ChainInput>&, std::vector<ChainResult>&, unsigned*,
		int);
template bool run_bfgs<GPU_PRECISION>(const DeviceModels<GPU_PRECISION>&,
		const DeviceTables<GPU_PRECISION>&, const InterContext<GPU_PRECISION>&,
		const MinParams&, const double[3], const std::vector<int>&,
		std::vector<std::vector<double> >&, std::vector<double>&, int);
template bool run_refine<GPU_PRECISION>(const DeviceModels<GPU_PRECISION>&,
		const DeviceTables<GPU_PRECISION>&, const InterContext<GPU_PRECISION>&,
		const MinParams&, const double[3], const std::vector<int>&,
		std::vector<std::vector<double> >&, std::vector<double>&,
		std::vector<std::vector<double> >&, int, const std::vector<unsigned>&,
		const std::vector<FlatGridGeometry>&);

} // namespace gpusmina
