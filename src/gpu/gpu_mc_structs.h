/*
 * Parameters and per-chain memory layout of the device Monte Carlo search.
 * Plain data, shared by host and device code.
 */

#ifndef SMINA_GPU_MC_STRUCTS_H
#define SMINA_GPU_MC_STRUCTS_H

namespace gpusmina
{

// minimization_params
struct MinParams
{
	unsigned maxiters;
	int accurate_line; // BFGSAccurateLineSearch
	int early_term;
};

// monte_carlo (with its ssd_par.minparm)
struct MCParams
{
	unsigned num_steps;
	double temperature;
	double mutation_amplitude;
	double min_rmsd;
	int num_saved_mins;
	double hunt_cap[3];
	double authentic_v[3];
	MinParams min;
};

// element offsets (in T) of one chain's block of floating point state
struct ChainOffsets
{
	int tmp, cand, coords; // current conf, candidate, heavy atom coords
	int x_new, x_orig, g, g_new, g_orig, p, y, minus_hy, h; // bfgs
	int out_conf, out_e, out_coords; // output container storage
	int total;
	// int block: sorted order of the output storage, then its size
	int order, count, int_total;
};

// everything bfgs needs besides the conf being minimized
template<typename T>
struct BFGSWork
{
	T* x_new;
	T* x_orig;
	T* g;
	T* g_new;
	T* g_orig;
	T* p;
	T* y;
	T* minus_hy;
	T* h; // triangular n x n
};

template<typename T>
struct ChainView
{
	T* tmp;
	T* cand;
	T* coords;
	BFGSWork<T> work;
	T* out_conf;
	T* out_e;
	T* out_coords;
	int* order;
	int* count;
};

inline ChainOffsets chain_layout(int conf_size, int change_size, int num_heavy,
		int num_saved)
{
	ChainOffsets o;
	int at = 0;
	o.tmp = at;
	at += conf_size;
	o.cand = at;
	at += conf_size;
	o.coords = at;
	at += 3 * num_heavy;
	o.x_new = at;
	at += conf_size;
	o.x_orig = at;
	at += conf_size;
	o.g = at;
	at += change_size;
	o.g_new = at;
	at += change_size;
	o.g_orig = at;
	at += change_size;
	o.p = at;
	at += change_size;
	o.y = at;
	at += change_size;
	o.minus_hy = at;
	at += change_size;
	o.h = at;
	at += change_size * (change_size + 1) / 2;
	o.out_conf = at;
	at += num_saved * conf_size;
	o.out_e = at;
	at += num_saved;
	o.out_coords = at;
	at += num_saved * 3 * num_heavy;
	o.total = at;
	o.order = 0;
	o.count = num_saved;
	o.int_total = num_saved + 1;
	return o;
}

} // namespace gpusmina

#endif
