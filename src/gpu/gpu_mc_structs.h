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
	// not smina's search: skip local optimizations QuickVina 2's history
	// check deems redundant (see visited_interesting)
	int visited;
};

// element offsets (in T) of one chain's block of floating point state
struct ChainOffsets
{
	int tmp, cand, coords; // current conf, candidate, heavy atom coords
	int x_new, x_orig, g, g_new, g_orig, p, y, minus_hy, h; // bfgs
	int out_conf, out_e, out_coords; // output container storage
	// a chain stopped between steps: current energy, best energy, and the
	// model state mutation reads (atom coordinates, frame origins)
	int resume_e, resume_coords, resume_origins;
	// QuickVina 2's history (MCParams::visited): points (5n x n), energies,
	// scratch (n + 5n); in the int block, gradient sign bits (4 per point)
	// and the state (size, next slot, full).  vis_n: the n storage is
	// sized for (0: no history)
	int vis_x, vis_f, vis_scratch, vis_n;
	int total;
	// int block: sorted order of the output storage, then its size
	int order, count, vis_bits, vis_state, int_total;
};

// a chain's QuickVina 2 history (null pointers: none)
template<typename T>
struct Visited
{
	T* x;
	T* f;
	T* scratch;
	unsigned* bits;
	int* state;
	int max_n;
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
	Visited<T> vis;
};

inline ChainOffsets chain_layout(int conf_size, int change_size, int num_heavy,
		int num_saved, int num_movable = 0, int num_frames = 0,
		int visited_n = 0)
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
	o.resume_e = at;
	at += 2;
	o.resume_coords = at;
	at += 3 * num_movable;
	o.resume_origins = at;
	at += 3 * num_frames;
	o.vis_n = visited_n;
	o.vis_x = at;
	at += 5 * visited_n * visited_n;
	o.vis_f = at;
	at += 5 * visited_n;
	o.vis_scratch = at;
	at += 6 * visited_n;
	o.total = at;
	o.order = 0;
	o.count = num_saved;
	o.vis_bits = num_saved + 1;
	o.vis_state = o.vis_bits + 4 * 5 * visited_n;
	o.int_total = o.vis_state + (visited_n ? 3 : 0);
	return o;
}

} // namespace gpusmina

#endif
