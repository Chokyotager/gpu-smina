/*
 * Device port of smina's search: bfgs.h (both line searches), mutate.cpp,
 * monte_carlo.cpp (operator() producing the output container) and
 * coords.cpp (add_to_output_container).  One thread group (gpu_group.cuh)
 * runs one chain; rank 0 owns the random number generator and the scalar
 * control flow is evaluated identically by all threads from group-uniform
 * values.
 */

#ifndef SMINA_GPU_MC_CUH
#define SMINA_GPU_MC_CUH

#include <cfloat>
#include "gpu_eval.cuh"
#include "gpu_rng.cuh"
#include "gpu_mc_structs.h"

namespace gpusmina
{

template<typename T> __device__ __forceinline__ T max_value();
template<> __device__ __forceinline__ float max_value<float>() { return FLT_MAX; }
template<> __device__ __forceinline__ double max_value<double>() { return DBL_MAX; }

// scalar_product over the change vectors
template<typename T, bool ORDERED, int TPC> __device__ T dot_n(const T* a, const T* b,
		int n, const Group<TPC>& G)
{
	if (ORDERED)
	{
		T tmp = 0;
		if (G.rank == 0)
			for (int i = 0; i < n; i++)
				tmp += a[i] * b[i];
		return G.bcast(tmp);
	}
	T part = 0;
	for (int i = G.rank; i < n; i += TPC)
		part += a[i] * b[i];
	return G.sum(part);
}

template<typename T, int TPC> __device__ void copy_n(T* dst, const T* src, int n,
		const Group<TPC>& G)
{
	for (int i = G.rank; i < n; i += TPC)
		dst[i] = src[i];
	G.sync();
}

// minus_mat_vec_product: out(i) = -sum_j h(i, j) * in(j)
template<typename T, int TPC> __device__ void minus_mat_vec(const T* h, const T* in,
		T* out, int n, const Group<TPC>& G)
{
	for (int i = G.rank; i < n; i += TPC)
	{
		T sum = 0;
		for (int j = 0; j < n; j++)
			sum += h[(i < j) ? tri_index(i, j) : tri_index(j, i)] * in[j];
		out[i] = -sum;
	}
	G.sync();
}

// conf::increment(change, factor)
template<typename T, int TPC> __device__ void conf_increment(const ModelView<T>& mv,
		T* x, const T* c, T factor, const Group<TPC>& G)
{
	for (int bi = 0; bi < mv.d.num_blocks; bi++)
	{
		const ConfBlock& b = mv.blocks[bi];
		T* xb = x + b.conf_offset;
		const T* cb = c + b.change_offset;
		int xt = 0, ct = 0;
		if (b.is_ligand)
		{
			if (G.rank == 0)
			{
				for (int k = 0; k < 3; k++)
					xb[k] += factor * cb[k];
				vec3<T> rotation = factor * load3(cb + 3);
				quat<T> q = mkq(xb[3], xb[4], xb[5], xb[6]);
				quaternion_increment(q, rotation);
				xb[3] = q.a;
				xb[4] = q.b;
				xb[5] = q.c;
				xb[6] = q.d;
			}
			xt = 7;
			ct = 6;
		}
		for (int t = G.rank; t < b.ntors; t += TPC)
		{
			T& tor = xb[xt + t];
			tor += normalized_angle(factor * cb[ct + t]);
			normalize_angle(tor);
		}
	}
	G.sync();
}

// quaternion_to_angle (quaternion.cpp)
template<typename T> __device__ vec3<T> quaternion_to_angle(const quat<T>& q)
{
	const T c = q.a;
	if (c > -1 && c < 1)
	{
		T angle = 2 * gacos(c);
		if (angle > c_pi<T>())
			angle -= 2 * c_pi<T>();
		vec3<T> axis = mk(q.b, q.c, q.d);
		T s = gsin(angle / 2);
		if (gabs(s) < c_epsilon_fl<T>())
			return mk(T(0), T(0), T(0));
		scale(axis, angle / s);
		return axis;
	}
	return mk(T(0), T(0), T(0));
}

// conf::operator()(index) in change indexing (orientation as an angle)
template<typename T> __device__ T conf_value(const ModelView<T>& mv,
		const T* x, int index)
{
	for (int bi = 0; bi < mv.d.num_blocks; bi++)
	{
		const ConfBlock& b = mv.blocks[bi];
		const int len = b.ntors + (b.is_ligand ? 6 : 0);
		if (index >= b.change_offset + len)
			continue;
		int k = index - b.change_offset;
		const T* xb = x + b.conf_offset;
		if (!b.is_ligand)
			return xb[k];
		if (k < 3)
			return xb[k];
		if (k < 6)
			return quaternion_to_angle(mkq(xb[3], xb[4], xb[5], xb[6]))[k - 3];
		return xb[7 + (k - 6)];
	}
	return 0;
}

template<typename T>
struct Objective // quasi_newton_aux
{
	const ModelView<T>& mv;
	const DevTables<T>& tb;
	const InterContext<T>& ic;
	const T* v;
	EvalScratch<T>& ws;
	__device__ Objective(const ModelView<T>& mv_, const DevTables<T>& tb_,
			const InterContext<T>& ic_, const T* v_, EvalScratch<T>& ws_) :
			mv(mv_), tb(tb_), ic(ic_), v(v_), ws(ws_)
	{
	}
};

template<typename T, bool ORDERED, int TPC> __device__ __forceinline__ T evaluate(
		const Objective<T>& f, const T* x, T* g, const Group<TPC>& G)
{
	T e = eval_deriv<T, ORDERED>(f.mv, f.tb, f.ic, f.v, x, g, f.ws, G);
#ifdef GPU_TRACE_EVALS
	if (G.rank == 0 && blockIdx.x == 0 && threadIdx.x == 0)
		printf("GPUEVAL %.17g\n", double(e));
#endif
	return e;
}

template<typename T, bool ORDERED, int TPC> __device__ T fast_line_search(
		const Objective<T>& f, int n, const T* x, const T* g, T f0,
		const T* p, T* x_new, T* g_new, T& f1, const Group<TPC>& G)
{
	const T c0 = T(0.0001);
	const int max_trials = 10;
	const T multiplier = T(0.5);
	T alpha = 1;
	const T pg = dot_n<T, ORDERED>(p, g, n, G);
	const int conf_size = f.mv.d.conf_size;
	for (int trial = 0; trial < max_trials; trial++)
	{
		copy_n(x_new, x, conf_size, G);
		conf_increment(f.mv, x_new, p, alpha, G);
		f1 = evaluate<T, ORDERED>(f, x_new, g_new, G);
		if (f1 - f0 < c0 * alpha * pg)
			break;
		alpha *= multiplier;
	}
	return alpha;
}

template<typename T, int TPC> __device__ void zero_n(T* a, int n, const Group<TPC>& G)
{
	for (int i = G.rank; i < n; i += TPC)
		a[i] = 0;
	G.sync();
}

template<typename T, bool ORDERED, int TPC> __device__ T accurate_line_search(
		const Objective<T>& f, int n, const T* x, const T* g, T f0,
		const T* p, T* x_new, T* g_new, T& f1, const Group<TPC>& G)
{
	T a, alpha, alpha2 = 0, alamin, b, disc, f2 = 0;
	T rhs1, rhs2, slope = 0, test, tmplam = 0;
	const T ALF = T(1.0e-4);
	const T FIRST = T(1.0);
	const int conf_size = f.mv.d.conf_size;

	slope = dot_n<T, ORDERED>(g, p, n, G);
	if (slope >= 0)
	{
		copy_n(x_new, x, conf_size, G);
		zero_n(g_new, n, G);
		return 0;
	}
	// test = max_i |p(i)| / max(|x(i)|, 1)
	T part = 0;
	for (int i = G.rank; i < n; i += TPC)
	{
		const T ax = gabs(conf_value(f.mv, x, i));
		T temp = gabs(p[i]) / ((ax < T(1.0)) ? T(1.0) : ax); // std::max
		if (temp > part)
			part = temp;
	}
	test = G.max(part);

	alamin = c_epsilon_fl<T>() / test;
	alpha = FIRST;
	for (;;)
	{
		copy_n(x_new, x, conf_size, G);
		conf_increment(f.mv, x_new, p, alpha, G);
		f1 = evaluate<T, ORDERED>(f, x_new, g_new, G);

		if (alpha < alamin)
		{
			copy_n(x_new, x, conf_size, G);
			zero_n(g_new, n, G);
			return 0;
		}
		else if (f1 <= f0 + ALF * alpha * slope)
		{
			return alpha;
		}
		else
		{
			if (alpha == FIRST)
			{
				tmplam = -slope / (T(2.0) * (f1 - f0 - slope));
			}
			else
			{
				rhs1 = f1 - f0 - alpha * slope;
				rhs2 = f2 - f0 - alpha2 * slope;
				a = (rhs1 / (alpha * alpha) - rhs2 / (alpha2 * alpha2))
						/ (alpha - alpha2);
				b = (-alpha2 * rhs1 / (alpha * alpha)
						+ alpha * rhs2 / (alpha2 * alpha2)) / (alpha - alpha2);
				if (a == T(0.0))
					tmplam = -slope / (T(2.0) * b);
				else
				{
					disc = b * b - T(3.0) * a * slope;
					if (disc < 0)
						tmplam = T(0.5) * alpha;
					else if (b <= 0)
						tmplam = (-b + gsqrt(disc)) / (T(3.0) * a);
					else
						tmplam = -slope / (b + gsqrt(disc));
				}
				if (tmplam > T(.5) * alpha)
					tmplam = T(.5) * alpha;
			}
		}
		alpha2 = alpha;
		f2 = f1;
		alpha = (tmplam < T(0.1) * alpha) ? T(0.1) * alpha : tmplam; // std::max
	}
}

// bfgs_update: returns false when the curvature condition fails
template<typename T, bool ORDERED, int TPC> __device__ bool bfgs_update(T* h,
		const T* p, const T* y, T* minus_hy, const T alpha, int n, const Group<TPC>& G)
{
	const T yp = dot_n<T, ORDERED>(y, p, n, G);
	if (alpha * yp < c_epsilon_fl<T>())
		return false;
	minus_mat_vec(h, y, minus_hy, n, G);
	const T yhy = -dot_n<T, ORDERED>(y, minus_hy, n, G);
	const T r = 1 / (alpha * yp);
	const int total = n * (n + 1) / 2;
	for (int k = G.rank; k < total; k += TPC)
	{
		// k = i + j*(j+1)/2 with i <= j
		int j = int((sqrtf(8.0f * k + 1) - 1) / 2);
		while (j * (j + 1) / 2 > k)
			j--;
		while ((j + 1) * (j + 2) / 2 <= k)
			j++;
		const int i = k - j * (j + 1) / 2;
		h[k] += alpha * r * (minus_hy[i] * p[j] + minus_hy[j] * p[i])
				+ alpha * alpha * (r * r * yhy + r) * p[i] * p[j];
	}
	G.sync();
	return true;
}

template<typename T, int TPC> __device__ void set_diagonal(T* h, T x, int n, const Group<TPC>& G)
{
	for (int i = G.rank; i < n; i += TPC)
		h[tri_index(i, i)] = x;
	G.sync();
}

// bfgs(): x is minimized in place, the final value is returned
template<typename T, bool ORDERED, int TPC> __device__ T bfgs(const Objective<T>& f,
		T* x, const BFGSWork<T>& w, const MinParams& params, const Group<TPC>& G)
{
	PROF_START(tb0);
	const int n = f.mv.d.change_size;
	const int conf_size = f.mv.d.conf_size;
	zero_n(w.h, n * (n + 1) / 2, G);
	set_diagonal(w.h, T(1), n, G);

	T f0 = evaluate<T, ORDERED>(f, x, w.g, G);
	const T f_orig = f0;
	copy_n(w.g_orig, w.g, n, G);
	copy_n(w.x_orig, x, conf_size, G);

	for (unsigned step = 0; step < params.maxiters; step++)
	{
		minus_mat_vec(w.h, w.g, w.p, n, G);
		T f1 = 0;
		T alpha;
		if (params.accurate_line)
			alpha = accurate_line_search<T, ORDERED>(f, n, x, w.g, f0, w.p,
					w.x_new, w.g_new, f1, G);
		else
			alpha = fast_line_search<T, ORDERED>(f, n, x, w.g, f0, w.p,
					w.x_new, w.g_new, f1, G);
		if (alpha == 0)
			break;
		// y = g_new - g
		for (int i = G.rank; i < n; i += TPC)
			w.y[i] = w.g_new[i] - w.g[i];
		G.sync();

		const T prevf0 = f0;
		f0 = f1;
		copy_n(x, w.x_new, conf_size, G);

		if (params.early_term)
		{
			T diff = prevf0 - f0;
			if (gabs(diff) < T(1e-5))
				break;
		}

		copy_n(w.g, w.g_new, n, G);
		const T gradnormsq = dot_n<T, ORDERED>(w.g, w.g, n, G);
		if (!(gradnormsq >= T(1e-4)))
			break;

		if (step == 0)
		{
			const T yy = dot_n<T, ORDERED>(w.y, w.y, n, G);
			if (gabs(yy) > c_epsilon_fl<T>())
				set_diagonal(w.h, alpha * dot_n<T, ORDERED>(w.y, w.p, n, G) / yy,
						n, G);
		}
		bfgs_update<T, ORDERED>(w.h, w.p, w.y, w.minus_hy, alpha, n, G);
	}

	if (!(f0 <= f_orig))
	{
		f0 = f_orig;
		copy_n(x, w.x_orig, conf_size, G);
		copy_n(w.g, w.g_orig, n, G);
	}
	PROF_ADD(PROF_BFGS_TOTAL, tb0);
	return f0;
}

////////////////////////////// mutation //////////////////////////////

// random_inside_sphere (rank 0 only)
__device__ inline void random_inside_sphere(MTState& s, double out[3])
{
	for (;;)
	{
		double r1 = random_fl(-1, 1, s);
		double r2 = random_fl(-1, 1, s);
		double r3 = random_fl(-1, 1, s);
		if (r1 * r1 + r2 * r2 + r3 * r3 < 1)
		{
			out[0] = r1;
			out[1] = r2;
			out[2] = r3;
			return;
		}
	}
}

// model::gyration_radius on the current model state (last set_conf)
template<typename T> __device__ T gyration_radius(const ModelView<T>& mv,
		const ConfBlock& b, const EvalScratch<T>& ws)
{
	const vec3<T> origin = load3(ws.frame_origin + 3 * b.root_frame);
	T acc = 0;
	unsigned counter = 0;
	for (int i = b.atom_begin; i < b.atom_end; i++)
		if (!type_is_hydrogen(mv.atom_type[i]))
		{
			acc += norm_sqr(load3(ws.coords + 3 * i) - origin);
			++counter;
		}
	return (counter > 0) ? gsqrt(acc / T(counter)) : T(0);
}

// mutate_conf (rank 0 only; caller syncs)
template<typename T> __device__ void mutate_conf(const ModelView<T>& mv,
		T* c, T amplitude, const EvalScratch<T>& ws, MTState& rng)
{
	int count = 0;
	for (int bi = 0; bi < mv.d.num_blocks; bi++)
		count += mv.blocks[bi].ntors + (mv.blocks[bi].is_ligand ? 2 : 0);
	if (count == 0)
		return;
	int which = random_int(0, count - 1, rng);
	for (int bi = 0; bi < mv.d.num_blocks; bi++)
	{
		const ConfBlock& b = mv.blocks[bi];
		T* cb = c + b.conf_offset;
		if (b.is_ligand)
		{
			if (which == 0)
			{
				double s[3];
				random_inside_sphere(rng, s);
				for (int k = 0; k < 3; k++)
					cb[k] += amplitude * T(s[k]);
				return;
			}
			--which;
			if (which == 0)
			{
				T gr = gyration_radius(mv, b, ws);
				if (gr > c_epsilon_fl<T>())
				{
					double s[3];
					random_inside_sphere(rng, s);
					vec3<T> rotation = (amplitude / gr)
							* mk(T(s[0]), T(s[1]), T(s[2]));
					quat<T> q = mkq(cb[3], cb[4], cb[5], cb[6]);
					quaternion_increment(q, rotation);
					cb[3] = q.a;
					cb[4] = q.b;
					cb[5] = q.c;
					cb[6] = q.d;
				}
				return;
			}
			--which;
			if (which < b.ntors)
			{
				cb[7 + which] = T(random_fl(-3.1415926535897931,
						3.1415926535897931, rng));
				return;
			}
			which -= b.ntors;
		}
		else
		{
			if (which < b.ntors)
			{
				cb[which] = T(random_fl(-3.1415926535897931, 3.1415926535897931,
						rng));
				return;
			}
			which -= b.ntors;
		}
	}
}

////////////////////////////// output container //////////////////////////////

// add_to_output_container: entries are kept sorted through `order`
template<typename T, bool ORDERED, int TPC> __device__ void add_to_output(
		const ModelView<T>& mv, const ChainView<T>& cv, const T* conf, T e,
		const T* coords, T min_rmsd, int max_size, const Group<TPC>& G)
{
	const int nh = mv.d.num_heavy;
	const int csz = mv.d.conf_size;
	int count = *cv.count;
	// find_closest: smallest rmsd_upper_bound, first index on ties
	T best = max_value<T>();
	int besti = count;
	for (int k = G.rank; k < count; k += TPC)
	{
		const T* oc = cv.out_coords + size_t(cv.order[k]) * 3 * nh;
		T acc = 0;
		for (int a = 0; a < nh; a++)
			acc += norm_sqr(load3(coords + 3 * a) - load3(oc + 3 * a));
		T res = (nh > 0) ? gsqrt(acc / T(nh)) : T(0);
		if (res < best)
		{
			best = res;
			besti = k;
		}
	}
	G.argmin(best, besti);
	int slot = -1; // storage slot to overwrite
	if (besti < count && best < min_rmsd)
	{
		if (e < cv.out_e[cv.order[besti]])
			slot = cv.order[besti];
	}
	else
	{
		if (count < max_size)
		{
			slot = count; // storage slots are allocated in order
			if (G.rank == 0)
				cv.order[count] = count;
			count++;
		}
		else if (count > 0 && e < cv.out_e[cv.order[count - 1]])
			slot = cv.order[count - 1];
	}
	G.sync();
	if (slot >= 0)
	{
		copy_n(cv.out_conf + size_t(slot) * csz, conf, csz, G);
		copy_n(cv.out_coords + size_t(slot) * 3 * nh, coords, 3 * nh, G);
		if (G.rank == 0)
		{
			cv.out_e[slot] = e;
			*cv.count = count;
			// out.sort(): insertion sort of the order by energy
			for (int i = 1; i < count; i++)
			{
				int key = cv.order[i];
				T ke = cv.out_e[key];
				int j = i - 1;
				while (j >= 0 && cv.out_e[cv.order[j]] > ke)
				{
					cv.order[j + 1] = cv.order[j];
					j--;
				}
				cv.order[j + 1] = key;
			}
		}
	}
	G.sync();
}

// model::get_heavy_atom_movable_coords from the current model state
template<typename T, int TPC> __device__ void heavy_coords(const ModelView<T>& mv,
		const EvalScratch<T>& ws, T* out, const Group<TPC>& G)
{
	for (int k = G.rank; k < mv.d.num_heavy; k += TPC)
	{
		const int a = mv.heavy[k];
		out[3 * k] = ws.coords[3 * a];
		out[3 * k + 1] = ws.coords[3 * a + 1];
		out[3 * k + 2] = ws.coords[3 * a + 2];
	}
	G.sync();
}

// metropolis_accept (rank 0 only)
template<typename T> __device__ bool metropolis_accept(T old_f, T new_f,
		double temperature, MTState& rng)
{
	if (new_f < old_f)
		return true;
	const double acceptance_probability = portable_math::exp(
			double(old_f - new_f) / temperature);
	return random_fl(0, 1, rng) < acceptance_probability;
}

// monte_carlo::operator() for one chain; the chain's initial conf (already
// randomized on the host) is in cv.tmp
template<typename T, bool ORDERED, int TPC> __device__ void monte_carlo_chain(
		const ModelView<T>& mv, const DevTables<T>& tb,
		const InterContext<T>& ic, const MCParams& prm, const ChainView<T>& cv,
		EvalScratch<T>& ws, MTState& rng, const Group<TPC>& G, unsigned* progress)
{
	const T hunt_cap[3] = {T(prm.hunt_cap[0]), T(prm.hunt_cap[1]),
			T(prm.hunt_cap[2])};
	const T authentic_v[3] = {T(prm.authentic_v[0]), T(prm.authentic_v[1]),
			T(prm.authentic_v[2])};
	const Objective<T> f_hunt(mv, tb, ic, hunt_cap, ws);
	const Objective<T> f_auth(mv, tb, ic, authentic_v, ws);
	const int csz = mv.d.conf_size;

	PROF_START(tmc);
	T tmp_e = 0;
	T best_e = max_value<T>();
	if (G.rank == 0)
		*cv.count = 0;
	G.sync();

	for (unsigned step = 0; step < prm.num_steps; step++)
	{
		if (progress && G.rank == 0)
			atomicAdd(progress, 1u);
		PROF_START(tm);
		copy_n(cv.cand, cv.tmp, csz, G);
		if (G.rank == 0)
			mutate_conf(mv, cv.cand, T(prm.mutation_amplitude), ws, rng);
		G.sync();
		PROF_ADD(PROF_MUTATE, tm);
		const T cand_e = bfgs<T, ORDERED>(f_hunt, cv.cand, cv.work, prm.min, G);
		int accept = 0;
		if (G.rank == 0)
			accept = (step == 0)
					|| metropolis_accept(tmp_e, cand_e, prm.temperature, rng);
		accept = G.bcast(accept);
		if (accept)
		{
			copy_n(cv.tmp, cv.cand, csz, G);
			tmp_e = cand_e;
			set_conf(mv, cv.tmp, ws, G);
			if (tmp_e < best_e || *cv.count < prm.num_saved_mins)
			{
				tmp_e = bfgs<T, ORDERED>(f_auth, cv.tmp, cv.work, prm.min, G);
				set_conf(mv, cv.tmp, ws, G);
				PROF_START(to);
				heavy_coords(mv, ws, cv.coords, G);
				add_to_output<T, ORDERED>(mv, cv, cv.tmp, tmp_e, cv.coords,
						T(prm.min_rmsd), prm.num_saved_mins, G);
				PROF_ADD(PROF_OUTPUT, to);
				if (tmp_e < best_e)
					best_e = tmp_e;
			}
		}
	}
	PROF_ADD(PROF_MC_TOTAL, tmc);
}

} // namespace gpusmina

#endif
