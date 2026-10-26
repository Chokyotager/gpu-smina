/*
 * Cooperative energy + gradient evaluation: a port of model::eval_deriv
 * (model.cpp) with its two interaction back ends, cache::eval_deriv
 * (grid.cpp interpolation) and non_cache::eval_deriv, plus the
 * precalculate_linear / precalculate_splines pair tables.
 *
 * One thread group (gpu_group.cuh) evaluates one conformation.  When ORDERED
 * is true (the double precision validation build) every floating point
 * reduction is performed by rank 0 in the CPU's order; otherwise parallel,
 * deterministic reductions are used.
 */

#ifndef SMINA_GPU_EVAL_CUH
#define SMINA_GPU_EVAL_CUH

#include "gpu_defs.h"
#include "gpu_dev_structs.h"
#include "gpu_math.cuh"
#include "gpu_group.cuh"

namespace gpusmina
{

template<typename T> __host__ __device__ inline ModelView<T> make_view(
		const DevModels<T>& dm, int m)
{
	ModelView<T> v;
	v.d = dm.desc[m];
	v.atom_type = dm.atom_type + v.d.atom_offset;
	v.atom_charge = dm.atom_charge + v.d.atom_offset;
	v.atom_local = dm.atom_local + 3 * v.d.atom_offset;
	v.atom_frame = dm.atom_frame + v.d.atom_offset;
	v.inflex_coords = dm.inflex_coords + 3 * v.d.inflex_offset;
	v.frames = dm.frames + v.d.frame_offset;
	v.children = dm.children + v.d.child_offset;
	v.level_frames = dm.level_frames + v.d.frame_offset;
	v.level_start = dm.level_start + v.d.level_offset;
	v.pairs = dm.pairs + v.d.pair_offset;
	v.list_start = dm.list_start + v.d.list_offset;
	v.blocks = dm.blocks + v.d.block_offset;
	v.heavy = dm.heavy + v.d.heavy_offset;
	v.nbr_start = dm.nbr_start + v.d.nbr_start_offset;
	v.nbr = dm.nbr + v.d.nbr_offset;
	return v;
}

// Layout of a model's static data staged in shared memory: T-aligned arrays
// first, then 4-byte ones.  Shared by host (sizing) and device (staging).
struct StagedLayout
{
	size_t charge, local, inflex, frames, pairs; // T aligned
	size_t type, frame, children, level_frames, level_start, list_start,
			blocks, heavy, nbr_start, nbr; // 4-byte aligned
	size_t bytes;
};

template<typename T> __host__ __device__ inline StagedLayout staged_layout(
		const ModelDesc& d)
{
	StagedLayout L;
	size_t at = 0;
	const int ninflex = d.num_atoms - d.num_movable;
	L.charge = at;
	at += sizeof(T) * d.num_movable;
	L.local = at;
	at += sizeof(T) * 3 * d.num_movable;
	L.inflex = at;
	at += sizeof(T) * 3 * ninflex;
	L.frames = at;
	at += sizeof(Frame<T>) * d.num_frames;
	L.pairs = at;
	at += sizeof(Pair<T>) * d.num_pairs;
	L.type = at;
	at += 4 * d.num_movable;
	L.frame = at;
	at += 4 * d.num_movable;
	L.children = at;
	at += 4 * d.num_children;
	L.level_frames = at;
	at += 4 * d.num_frames;
	L.level_start = at;
	at += 4 * (d.num_levels + 1);
	L.list_start = at;
	at += 4 * (d.num_lists + 1);
	L.blocks = at;
	at += sizeof(ConfBlock) * d.num_blocks;
	L.heavy = at;
	at += 4 * d.num_heavy;
	L.nbr_start = at;
	at += 4 * (d.num_movable + 1);
	L.nbr = at;
	at += 4 * d.num_nbr;
	L.bytes = (at + 15) & ~size_t(15);
	return L;
}

template<typename X> __device__ __forceinline__ void block_copy(X* dst,
		const X* src, int n)
{
	const int* s = reinterpret_cast<const int*>(src);
	int* d = reinterpret_cast<int*>(dst);
	const int words = int(n * sizeof(X) / 4);
	for (int i = threadIdx.x; i < words; i += blockDim.x)
		d[i] = s[i];
}

// copy a model's static data into shared memory at `smem` and return a view
// of it; must be called by every thread of the block, followed by
// __syncthreads()
template<typename T> __device__ ModelView<T> stage_model(
		const ModelView<T>& g, unsigned char* smem)
{
	const StagedLayout L = staged_layout<T>(g.d);
	ModelView<T> v;
	v.d = g.d;
	T* charge = reinterpret_cast<T*>(smem + L.charge);
	T* local = reinterpret_cast<T*>(smem + L.local);
	T* inflex = reinterpret_cast<T*>(smem + L.inflex);
	Frame<T>* frames = reinterpret_cast<Frame<T>*>(smem + L.frames);
	Pair<T>* pairs = reinterpret_cast<Pair<T>*>(smem + L.pairs);
	int* type = reinterpret_cast<int*>(smem + L.type);
	int* frame = reinterpret_cast<int*>(smem + L.frame);
	int* children = reinterpret_cast<int*>(smem + L.children);
	int* level_frames = reinterpret_cast<int*>(smem + L.level_frames);
	int* level_start = reinterpret_cast<int*>(smem + L.level_start);
	int* list_start = reinterpret_cast<int*>(smem + L.list_start);
	ConfBlock* blocks = reinterpret_cast<ConfBlock*>(smem + L.blocks);
	int* heavy = reinterpret_cast<int*>(smem + L.heavy);
	int* nbr_start = reinterpret_cast<int*>(smem + L.nbr_start);
	int* nbr = reinterpret_cast<int*>(smem + L.nbr);
	const ModelDesc& d = g.d;
	block_copy(charge, g.atom_charge, d.num_movable);
	block_copy(local, g.atom_local, 3 * d.num_movable);
	block_copy(inflex, g.inflex_coords, 3 * (d.num_atoms - d.num_movable));
	block_copy(frames, g.frames, d.num_frames);
	block_copy(pairs, g.pairs, d.num_pairs);
	block_copy(type, g.atom_type, d.num_movable);
	block_copy(frame, g.atom_frame, d.num_movable);
	block_copy(children, g.children, d.num_children);
	block_copy(level_frames, g.level_frames, d.num_frames);
	block_copy(level_start, g.level_start, d.num_levels + 1);
	block_copy(list_start, g.list_start, d.num_lists + 1);
	block_copy(blocks, g.blocks, d.num_blocks);
	block_copy(heavy, g.heavy, d.num_heavy);
	block_copy(nbr_start, g.nbr_start, d.num_movable + 1);
	block_copy(nbr, g.nbr, d.num_nbr);
	v.atom_charge = charge;
	v.atom_local = local;
	v.inflex_coords = inflex;
	v.frames = frames;
	v.pairs = pairs;
	v.atom_type = type;
	v.atom_frame = frame;
	v.children = children;
	v.level_frames = level_frames;
	v.level_start = level_start;
	v.list_start = list_start;
	v.blocks = blocks;
	v.heavy = heavy;
	v.nbr_start = nbr_start;
	v.nbr = nbr;
	return v;
}

template<typename T> __device__ __forceinline__ EvalScratch<T> make_scratch(
		T* base, T* pair_buf, const ScratchOffsets& so, int* error)
{
	EvalScratch<T> ws;
	ws.coords = base + so.coords;
	ws.forces = base + so.forces;
	ws.frame_origin = base + so.frame_origin;
	ws.frame_q = base + so.frame_q;
	ws.frame_m = base + so.frame_m;
	ws.frame_axis = base + so.frame_axis;
	ws.frame_ft = base + so.frame_ft;
	ws.pair_buf = pair_buf;
	ws.error = error;
	return ws;
}

template<typename T> __device__ __forceinline__ vec3<T> atom_coords(
		const ModelView<T>& mv, const EvalScratch<T>& ws, int i)
{
	if (i < mv.d.num_movable)
		return load3(ws.coords + 3 * i);
	return load3(mv.inflex_coords + 3 * (i - mv.d.num_movable));
}

////////////////////////////// pair tables //////////////////////////////

// result_components::eval with c[] already in the orientation used by eval
template<typename T> __device__ __forceinline__ T comb4(const T c[4], T w1,
		T w2, T w12)
{
	return c[0] + w1 * c[1] + w2 * c[2] + w12 * c[3];
}

// precalculate::eval_deriv for a pair of atoms a (first) and b (second)
// whose types index table `tpair`; swapped means type(a) > type(b).
// Returns e and dor (derivative over r).  KIND (0 linear, 1 spline) and NC
// (1 or 4 components) are compile-time for the hot loops; KIND -1 dispatches
// at run time.
template<typename T, int KIND = -1, int NC = -1>
__device__ __forceinline__ void table_eval_deriv(const DevTables<T>& tb,
		int tpair, bool swapped, T abs_qa, T abs_qb, T qab, T r2, T& e,
		T& dor)
{
	const int nc = (NC > 0) ? NC : tb.nc;
	const int kind = (KIND >= 0) ? KIND : tb.kind;
	if (kind == 0)
	{
		// precalculate_linear: data(min,max).eval_deriv(nc, A, B) with
		// (A, B) = swapped ? (b, a) : (a, b)
		T r2_factored = tb.factor * r2;
		int i1 = int(r2_factored);
		int i2 = i1 + 1;
		T rem = r2_factored - T(i1);
		const T* s1 = tb.lin_smooth + (size_t(tpair) * tb.n + i1) * nc * 2;
		const T* s2 = tb.lin_smooth + (size_t(tpair) * tb.n + i2) * nc * 2;
		T e1, e2, d1, d2;
		if (nc == 1)
		{
			e1 = s1[0];
			e2 = s2[0];
			d1 = s1[1];
			d2 = s2[1];
		}
		else
		{
			T wA = swapped ? abs_qb : abs_qa;
			T wB = swapped ? abs_qa : abs_qb;
			T c[4];
			for (int k = 0; k < 4; k++)
				c[k] = s1[2 * k];
			e1 = comb4(c, wA, wB, qab);
			for (int k = 0; k < 4; k++)
				c[k] = s2[2 * k];
			e2 = comb4(c, wA, wB, qab);
			for (int k = 0; k < 4; k++)
				c[k] = s1[2 * k + 1];
			d1 = comb4(c, wA, wB, qab);
			for (int k = 0; k < 4; k++)
				c[k] = s2[2 * k + 1];
			d2 = comb4(c, wA, wB, qab);
		}
		e = e1 + rem * (e2 - e1);
		dor = d1 + rem * (d2 - d1);
	}
	else
	{
		// precalculate_splines: evaldata (swapOrder when swapped), then
		// result_components::eval(a, b), then divide derivative by r
		T r = gsqrt(r2);
		T val[4] = {0, 0, 0, 0}, der[4] = {0, 0, 0, 0};
		for (int c = 0; c < nc; c++)
		{
			T cut = tb.spl_cutoff[tpair * nc + c];
			if (r >= cut)
				continue;
			unsigned index = unsigned(r / tb.fraction);
			const T* sd = tb.spl_data
					+ ((size_t(tpair) * nc + c) * tb.nseg + index) * 5;
			const T lx = r - sd[0];
			val[c] = ((sd[1] * lx + sd[2]) * lx + sd[3]) * lx + sd[4];
			der[c] = (3 * sd[1] * lx + 2 * sd[2]) * lx + sd[3];
		}
		if (nc == 1)
		{
			e = val[0];
			dor = der[0];
		}
		else
		{
			if (swapped)
			{
				T t = val[1];
				val[1] = val[2];
				val[2] = t;
				t = der[1];
				der[1] = der[2];
				der[2] = t;
			}
			e = comb4(val, abs_qa, abs_qb, qab);
			dor = comb4(der, abs_qa, abs_qb, qab);
		}
		dor /= r;
	}
}

// precalculate::eval_fast(t1, t2, r2) -> result components oriented (t1, t2)
// used by the grid builder (cache::populate)
template<typename T> __device__ __forceinline__ void table_eval_fast(
		const DevTables<T>& tb, int t1, int t2, T r2, T out[4])
{
	const int nc = tb.nc;
	int tpair = tri_index_permissive(t1, t2);
	for (int c = 0; c < 4; c++)
		out[c] = 0;
	if (tb.kind == 0)
	{
		int i = int(tb.factor * r2);
		const T* f = tb.lin_fast + (size_t(tpair) * tb.n + i) * nc;
		for (int c = 0; c < nc; c++)
			out[c] = f[c];
	}
	else
	{
		T r = gsqrt(r2);
		for (int c = 0; c < nc; c++)
		{
			T cut = tb.spl_cutoff[tpair * nc + c];
			if (r >= cut)
				continue;
			unsigned index = unsigned(r / tb.fraction);
			const T* sd = tb.spl_data
					+ ((size_t(tpair) * nc + c) * tb.nseg + index) * 5;
			const T lx = r - sd[0];
			out[c] = ((sd[1] * lx + sd[2]) * lx + sd[3]) * lx + sd[4];
		}
	}
	if (t1 > t2) // result_components::swapOrder
	{
		T t = out[1];
		out[1] = out[2];
		out[2] = t;
	}
}

////////////////////////////// grids //////////////////////////////

// grid::evaluate_aux
template<typename T> __device__ __forceinline__ T grid_evaluate_aux(
		const DevGridGeometry<T>& g,
		const T* data, const vec3<T>& location, T slope, T v, vec3<T>* deriv)
{
	vec3<T> s;
	for (int i = 0; i < 3; i++)
		s[i] = (location[i] - g.init[i]) * g.factor[i];
	vec3<T> miss = mk(T(0), T(0), T(0));
	int region[3];
	int a[3];
	for (int i = 0; i < 3; i++)
	{
		if (s[i] < 0)
		{
			miss[i] = -s[i];
			region[i] = -1;
			a[i] = 0;
			s[i] = 0;
		}
		else if (s[i] >= g.dim_fl_minus_1[i])
		{
			miss[i] = s[i] - g.dim_fl_minus_1[i];
			region[i] = 1;
			a[i] = g.dim[i] - 2;
			s[i] = 1;
		}
		else
		{
			region[i] = 0;
			a[i] = int(s[i]);
			s[i] -= T(a[i]);
		}
	}
	const T penalty = slope * dot(miss, mk(g.factor_inv[0], g.factor_inv[1],
			g.factor_inv[2]));

	const int nx = g.dim[0], ny = g.dim[1];
	const int x0 = a[0], y0 = a[1], z0 = a[2];
	const int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
#define GIDX(x, y, z) ((x) + nx * ((y) + ny * (z)))
	const T f000 = data[GIDX(x0, y0, z0)];
	const T f100 = data[GIDX(x1, y0, z0)];
	const T f010 = data[GIDX(x0, y1, z0)];
	const T f110 = data[GIDX(x1, y1, z0)];
	const T f001 = data[GIDX(x0, y0, z1)];
	const T f101 = data[GIDX(x1, y0, z1)];
	const T f011 = data[GIDX(x0, y1, z1)];
	const T f111 = data[GIDX(x1, y1, z1)];
#undef GIDX
	const T x = s[0], y = s[1], z = s[2];
	const T mx = 1 - x, my = 1 - y, mz = 1 - z;

	T f = f000 * mx * my * mz + f100 * x * my * mz + f010 * mx * y * mz
			+ f110 * x * y * mz + f001 * mx * my * z + f101 * x * my * z
			+ f011 * mx * y * z + f111 * x * y * z;

	if (deriv)
	{
		const T x_g = f000 * (-1) * my * mz + f100 * 1 * my * mz
				+ f010 * (-1) * y * mz + f110 * 1 * y * mz
				+ f001 * (-1) * my * z + f101 * 1 * my * z
				+ f011 * (-1) * y * z + f111 * 1 * y * z;
		const T y_g = f000 * mx * (-1) * mz + f100 * x * (-1) * mz
				+ f010 * mx * 1 * mz + f110 * x * 1 * mz
				+ f001 * mx * (-1) * z + f101 * x * (-1) * z
				+ f011 * mx * 1 * z + f111 * x * 1 * z;
		const T z_g = f000 * mx * my * (-1) + f100 * x * my * (-1)
				+ f010 * mx * y * (-1) + f110 * x * y * (-1)
				+ f001 * mx * my * 1 + f101 * x * my * 1
				+ f011 * mx * y * 1 + f111 * x * y * 1;
		vec3<T> gradient = mk(x_g, y_g, z_g);
		curl(f, gradient, v);
		for (int i = 0; i < 3; i++)
		{
			T ge = (region[i] == 0) ? gradient[i] : T(0);
			(*deriv)[i] = g.factor[i] * ge + slope * T(region[i]);
		}
		return f + penalty;
	}
	curl(f, v);
	return f + penalty;
}

////////////////////////////// conformation //////////////////////////////

// model::set: ligands.set_conf then flex.set_conf, frames level by level
template<typename T, int TPC> __device__ void set_conf(const ModelView<T>& mv,
		const T* conf, EvalScratch<T>& ws, const Group<TPC>& G)
{
	for (int lv = 0; lv < mv.d.num_levels; lv++)
	{
		for (int k = mv.level_start[lv] + G.rank; k < mv.level_start[lv + 1];
				k += TPC)
		{
			const int fi = mv.level_frames[k];
			const Frame<T>& f = mv.frames[fi];
			vec3<T> origin, axis;
			quat<T> q;
			if (f.kind == FRAME_RIGID)
			{
				origin = load3(conf + f.conf_idx);
				const T* qp = conf + f.conf_idx + 3;
				q = mkq(qp[0], qp[1], qp[2], qp[3]);
				axis = mk(T(0), T(0), T(0));
			}
			else if (f.kind == FRAME_FIRST_SEGMENT)
			{
				origin = load3(f.origin);
				axis = load3(f.axis);
				q = angle_to_quaternion(axis, conf[f.conf_idx]);
			}
			else
			{
				const int p = f.parent;
				mat3<T> pm;
				for (int j = 0; j < 9; j++)
					pm.d[j] = ws.frame_m[9 * p + j];
				vec3<T> porigin = load3(ws.frame_origin + 3 * p);
				const T* pq = ws.frame_q + 4 * p;
				origin = porigin + mul(pm, load3(f.rel_origin));
				axis = mul(pm, load3(f.rel_axis));
				q = qmul(angle_to_quaternion(axis, conf[f.conf_idx]),
						mkq(pq[0], pq[1], pq[2], pq[3]));
				quaternion_normalize_approx(q);
			}
			mat3<T> m = quaternion_to_r3(q);
			store3(ws.frame_origin + 3 * fi, origin);
			store3(ws.frame_axis + 3 * fi, axis);
			T* qo = ws.frame_q + 4 * fi;
			qo[0] = q.a;
			qo[1] = q.b;
			qo[2] = q.c;
			qo[3] = q.d;
			for (int j = 0; j < 9; j++)
				ws.frame_m[9 * fi + j] = m.d[j];
		}
		G.sync();
	}
	// atom_frame::set_coords: origin + orientation_m * local
	for (int i = G.rank; i < mv.d.num_movable; i += TPC)
	{
		const int fi = mv.atom_frame[i];
		mat3<T> m;
		for (int j = 0; j < 9; j++)
			m.d[j] = ws.frame_m[9 * fi + j];
		vec3<T> c = load3(ws.frame_origin + 3 * fi)
				+ mul(m, load3(mv.atom_local + 3 * i));
		store3(ws.coords + 3 * i, c);
	}
	G.sync();
}

////////////////////////////// interactions //////////////////////////////

// sum of per-atom values: CPU order (rank 0) when ORDERED, else parallel
template<typename T, bool ORDERED, int TPC> __device__ T sum_atoms(
		const T* atom_e, int n, T part, const Group<TPC>& G)
{
	if (ORDERED)
	{
		G.sync();
		T e = 0;
		if (G.rank == 0)
			for (int i = 0; i < n; i++)
				e += atom_e[i];
		return G.bcast(e);
	}
	return G.sum(part);
}

// cache::eval_deriv: sets forces of all movable atoms, returns energy
template<typename T, bool ORDERED, int TPC> __device__ T cache_eval_deriv(
		const ModelView<T>& mv, const DevGrids<T>& grids, T slope, T v,
		EvalScratch<T>& ws, const Group<TPC>& G)
{
	T* atom_e = ws.pair_buf; // ORDERED only
	T part = 0;
	for (int i = G.rank; i < mv.d.num_movable; i += TPC)
	{
		const int t = mv.atom_type[i];
		T ei = 0;
		vec3<T> deriv = mk(T(0), T(0), T(0));
		if (t < NUM_TYPES && !type_is_hydrogen(t))
		{
			const vec3<T> loc = load3(ws.coords + 3 * i);
			ei = grid_evaluate_aux(grids.geo, grids.data[t], loc, slope, v,
					&deriv);
			const T q = mv.atom_charge[i];
			if (q != 0 && grids.chargedata[t])
			{
				vec3<T> cderiv = mk(T(0), T(0), T(0));
				ei += q * grid_evaluate_aux(grids.geo, grids.chargedata[t], loc,
						slope, v, &cderiv);
				add_to(deriv, q * cderiv);
			}
		}
		store3(ws.forces + 3 * i, deriv);
		if (ORDERED)
			atom_e[i] = ei;
		else
			part += ei;
	}
	return sum_atoms<T, ORDERED>(atom_e, mv.d.num_movable, part, G);
}

// grid::evaluate_user on the --user_grid (no deriv when deriv == null)
template<typename T> __device__ T user_grid_eval(const DevUserGrid<T>& ug,
		const vec3<T>& loc, T slope, vec3<T>* deriv)
{
	return grid_evaluate_aux(ug.geo, ug.data, loc, slope, T(1000), deriv);
}

// non_cache::eval_deriv: one thread per ligand atom, receptor atoms in the
// CPU's possibilities order
template<typename T, bool ORDERED, int TPC> __device__ T non_cache_eval_deriv(
		const ModelView<T>& mv, const DevReceptor<T>& rec,
		const DevBox<T>& box, const DevTables<T>& tb, const DevUserGrid<T>& ug,
		T slope, T v, EvalScratch<T>& ws, const Group<TPC>& G)
{
	T* atom_e = ws.pair_buf;
	T part = 0;
	for (int i = G.rank; i < mv.d.num_movable; i += TPC)
	{
		const int t1 = mv.atom_type[i];
		T contrib = 0;
		vec3<T> force = mk(T(0), T(0), T(0));
		if (t1 < NUM_TYPES && !type_is_hydrogen(t1))
		{
			T this_e = 0;
			vec3<T> deriv = mk(T(0), T(0), T(0));
			vec3<T> oob_deriv = mk(T(0), T(0), T(0));
			T oob_penalty = 0;
			const vec3<T> a_coords = load3(ws.coords + 3 * i);
			vec3<T> adj = a_coords;
			for (int j = 0; j < 3; j++)
			{
				if (box.enabled[j])
				{
					if (a_coords[j] < box.begin[j])
					{
						adj[j] = box.begin[j];
						oob_deriv[j] = -1;
						oob_penalty += gabs(a_coords[j] - box.begin[j]);
					}
					else if (a_coords[j] > box.end[j])
					{
						adj[j] = box.end[j];
						oob_deriv[j] = 1;
						oob_penalty += gabs(a_coords[j] - box.end[j]);
					}
				}
			}
			oob_penalty *= slope;
			scale(oob_deriv, slope);

			int cell[3];
			for (int j = 0; j < 3; j++)
			{
				int c = int(gfloor(adj[j] / T(3))) - rec.offset[j];
				cell[j] = min(max(c, 0), rec.dims[j] - 1);
			}
			const int ci = cell[0] + rec.dims[0] * (cell[1] + rec.dims[1] * cell[2]);
			const T qa = mv.atom_charge[i];
			const T abs_qa = gabs(qa);
			for (int k = rec.cell_start[ci]; k < rec.cell_start[ci + 1]; k++)
			{
				const int j = rec.cell_atoms[k];
				const vec3<T> r_ba = adj - load3(rec.coords + 3 * j);
				const T r2 = norm_sqr(r_ba);
				if (r2 < tb.cutoff_sqr)
				{
					if (r2 < c_epsilon_fl<T>())
						*ws.error = 1;
					const int t2 = rec.type[j];
					const T qb = rec.charge[j];
					T e, dor;
					table_eval_deriv(tb, tri_index_permissive(t1, t2), t1 > t2,
							abs_qa, gabs(qb), qa * qb, r2, e, dor);
					this_e += e;
					add_to(deriv, dor * r_ba);
				}
			}
			vec3<T> ug_deriv = mk(T(0), T(0), T(0));
			if (ug.initialized)
				this_e += user_grid_eval(ug, a_coords, slope, &ug_deriv);
			add_to(deriv, ug_deriv);
			curl(this_e, deriv, v);
			force = deriv + oob_deriv;
			contrib = this_e + oob_penalty;
		}
		store3(ws.forces + 3 * i, force);
		if (ORDERED)
			atom_e[i] = contrib;
		else
			part += contrib;
	}
	return sum_atoms<T, ORDERED>(atom_e, mv.d.num_movable, part, G);
}

// model::eval_interacting_pairs_deriv over list l in CPU order (ORDERED):
// pair terms in parallel, accumulation by rank 0; adds to forces
template<typename T, int TPC> __device__ T pairs_ordered(
		const ModelView<T>& mv, const DevTables<T>& tb, int l, T v,
		EvalScratch<T>& ws, const Group<TPC>& G)
{
	const int begin = mv.list_start[l], end = mv.list_start[l + 1];
	const int nmov = mv.d.num_movable;
	for (int k = begin + G.rank; k < end; k += TPC)
	{
		const Pair<T>& p = mv.pairs[k];
		const vec3<T> r = atom_coords(mv, ws, p.b) - atom_coords(mv, ws, p.a);
		const T r2 = norm_sqr(r);
		T e = 0;
		vec3<T> force = mk(T(0), T(0), T(0));
		if (r2 < tb.cutoff_sqr)
		{
			T dor;
			table_eval_deriv(tb, p.tpair, p.swapped != 0, p.abs_qa, p.abs_qb,
					p.qab, r2, e, dor);
			force = dor * r;
			curl(e, force, v);
		}
		T* o = ws.pair_buf + 4 * (k - begin);
		o[0] = e;
		o[1] = force[0];
		o[2] = force[1];
		o[3] = force[2];
	}
	G.sync();
	T e = 0;
	if (G.rank == 0)
	{
		for (int k = begin; k < end; k++)
		{
			const Pair<T>& p = mv.pairs[k];
			const T* o = ws.pair_buf + 4 * (k - begin);
			const vec3<T> r = atom_coords(mv, ws, p.b) - atom_coords(mv, ws, p.a);
			if (norm_sqr(r) < tb.cutoff_sqr)
			{
				e += o[0];
				const vec3<T> force = load3(o + 1);
				if (p.a < nmov)
				{
					vec3<T> fa = load3(ws.forces + 3 * p.a);
					sub_from(fa, force);
					store3(ws.forces + 3 * p.a, fa);
				}
				if (p.b < nmov)
				{
					vec3<T> fb = load3(ws.forces + 3 * p.b);
					add_to(fb, force);
					store3(ws.forces + 3 * p.b, fb);
				}
			}
		}
	}
	return G.bcast(e);
}

// all interacting pairs in two passes, without atomics:
//  1. pair-major: each pair evaluated once, its force stored in pair_buf
//  2. atom-centric: each movable atom gathers +/- the forces of its pairs
// v_list0 applies to list 0 (other_pairs), v_lig to the ligands' pairs
template<typename T, int KIND, int NC, int TPC> __device__ T pairs_two_pass(
		const ModelView<T>& mv, const DevTables<T>& tb, T v_list0, T v_lig,
		EvalScratch<T>& ws, const Group<TPC>& G)
{
	const int npairs = mv.d.num_pairs;
	const int list0_end = mv.list_start[1];
	T* pf = ws.pair_buf;
	T part = 0;
#pragma unroll 2
	for (int k = G.rank; k < npairs; k += TPC)
	{
		const Pair<T>& p = mv.pairs[k];
		const vec3<T> r = atom_coords(mv, ws, p.b) - atom_coords(mv, ws, p.a);
		const T r2 = norm_sqr(r);
		vec3<T> force = mk(T(0), T(0), T(0));
		if (r2 < tb.cutoff_sqr)
		{
			T e, dor;
			table_eval_deriv<T, KIND, NC>(tb, p.tpair, p.swapped != 0, p.abs_qa,
					p.abs_qb, p.qab, r2, e, dor);
			force = dor * r;
			curl(e, force, (k < list0_end) ? v_list0 : v_lig);
			part += e;
		}
		store3(pf + 3 * k, force);
	}
	G.sync();
	for (int i = G.rank; i < mv.d.num_movable; i += TPC)
	{
		vec3<T> f = load3(ws.forces + 3 * i);
		const int e_end = mv.nbr_start[i + 1];
		for (int en = mv.nbr_start[i]; en < e_end; en++)
		{
			const int code = mv.nbr[en];
			const vec3<T> pk = load3(pf + 3 * (code >> 3));
			if (code & 1)
				add_to(f, pk);
			else
				sub_from(f, pk);
		}
		store3(ws.forces + 3 * i, f);
	}
	return G.sum(part);
}

template<typename T, int TPC> __device__ T pairs_eval_deriv_unordered(
		const ModelView<T>& mv, const DevTables<T>& tb, T v_list0, T v_lig,
		EvalScratch<T>& ws, const Group<TPC>& G)
{
	if (tb.kind == 0)
		return (tb.nc == 1) ?
				pairs_two_pass<T, 0, 1>(mv, tb, v_list0, v_lig, ws, G) :
				pairs_two_pass<T, 0, 4>(mv, tb, v_list0, v_lig, ws, G);
	return (tb.nc == 1) ?
			pairs_two_pass<T, 1, 1>(mv, tb, v_list0, v_lig, ws, G) :
			pairs_two_pass<T, 1, 4>(mv, tb, v_list0, v_lig, ws, G);
}

// heterotree / tree ::derivative: forces -> change vector g
template<typename T, int TPC> __device__ void derivative(
		const ModelView<T>& mv, EvalScratch<T>& ws, T* g, const Group<TPC>& G)
{
	// atom_frame::sum_force_and_torque for every frame
	for (int fi = G.rank; fi < mv.d.num_frames; fi += TPC)
	{
		const Frame<T>& f = mv.frames[fi];
		const vec3<T> origin = load3(ws.frame_origin + 3 * fi);
		vec3<T> force = mk(T(0), T(0), T(0)), torque = mk(T(0), T(0), T(0));
		for (int i = f.begin; i < f.end; i++)
		{
			const vec3<T> fo = load3(ws.forces + 3 * i);
			add_to(force, fo);
			add_to(torque, cross(load3(ws.coords + 3 * i) - origin, fo));
		}
		store3(ws.frame_ft + 6 * fi, force);
		store3(ws.frame_ft + 6 * fi + 3, torque);
	}
	G.sync();
	// branches_derivative bottom-up, then set_derivative
	for (int lv = mv.d.num_levels - 1; lv >= 0; lv--)
	{
		for (int k = mv.level_start[lv] + G.rank; k < mv.level_start[lv + 1];
				k += TPC)
		{
			const int fi = mv.level_frames[k];
			const Frame<T>& f = mv.frames[fi];
			const vec3<T> origin = load3(ws.frame_origin + 3 * fi);
			vec3<T> force = load3(ws.frame_ft + 6 * fi);
			vec3<T> torque = load3(ws.frame_ft + 6 * fi + 3);
			for (int c = f.child_begin; c < f.child_end; c++)
			{
				const int ci = mv.children[c];
				const vec3<T> cf = load3(ws.frame_ft + 6 * ci);
				const vec3<T> ct = load3(ws.frame_ft + 6 * ci + 3);
				add_to(force, cf);
				const vec3<T> r = load3(ws.frame_origin + 3 * ci) - origin;
				add_to(torque, cross(r, cf) + ct);
			}
			store3(ws.frame_ft + 6 * fi, force);
			store3(ws.frame_ft + 6 * fi + 3, torque);
			if (f.kind == FRAME_RIGID)
			{
				store3(g + f.change_idx, force);
				store3(g + f.change_idx + 3, torque);
			}
			else
				g[f.change_idx] = dot(torque, load3(ws.frame_axis + 3 * fi));
		}
		G.sync();
	}
}

// optional section timers (build with -DGPU_PROFILE): cycles accumulated by
// lane 0 of the first warp, reported by the kernels' host wrappers
#ifdef GPU_PROFILE
enum ProfileSection
{
	PROF_SET_CONF, PROF_INTER, PROF_PAIRS, PROF_DERIV, PROF_EVAL_TOTAL,
	PROF_BFGS_TOTAL, PROF_MUTATE, PROF_OUTPUT, PROF_MC_TOTAL, PROF_EVALS,
	PROF_COUNT
};
__device__ unsigned long long g_prof[PROF_COUNT];
#define PROF_START(v) long long v = clock64()
#define PROF_ADD(sec, v) \
	if (blockIdx.x == 0 && threadIdx.x == 0) \
		g_prof[sec] += clock64() - (v)
#define PROF_COUNT_ADD(sec, n) \
	if (blockIdx.x == 0 && threadIdx.x == 0) \
		g_prof[sec] += (n)
#else
#define PROF_START(v)
#define PROF_ADD(sec, v)
#define PROF_COUNT_ADD(sec, n)
#endif

// model::eval_deriv; v = {intra-ligand, intermolecular, other pairs}
template<typename T, bool ORDERED, int TPC> __device__ T eval_deriv(
		const ModelView<T>& mv, const DevTables<T>& tb,
		const InterContext<T>& ic, const T v[3], T slope, const DevBox<T>& box,
		const T* conf, T* g, EvalScratch<T>& ws, const Group<TPC>& G)
{
	PROF_START(t0);
	set_conf(mv, conf, ws, G);
	PROF_ADD(PROF_SET_CONF, t0);
	PROF_START(t1);
	T e;
	if (ic.use_cache)
		e = cache_eval_deriv<T, ORDERED>(mv, ic.grids, slope, v[1], ws, G);
	else
		e = non_cache_eval_deriv<T, ORDERED>(mv, ic.rec, box, tb, ic.ug, slope,
				v[1], ws, G);
	G.sync();
	PROF_ADD(PROF_INTER, t1);
	PROF_START(t2);
	if (ORDERED)
	{
		e += pairs_ordered(mv, tb, 0, v[2], ws, G);
		for (int l = 1; l < mv.d.num_lists; l++)
			e += pairs_ordered(mv, tb, l, v[0], ws, G);
	}
	else
		e += pairs_eval_deriv_unordered(mv, tb, v[2], v[0], ws, G);
	G.sync();
	PROF_ADD(PROF_PAIRS, t2);
	PROF_START(t3);
	derivative(mv, ws, g, G);
	PROF_ADD(PROF_DERIV, t3);
	PROF_ADD(PROF_EVAL_TOTAL, t0);
	PROF_COUNT_ADD(PROF_EVALS, 1);
	return e;
}

} // namespace gpusmina

#endif
