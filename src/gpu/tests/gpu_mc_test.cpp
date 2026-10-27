/*
 * Phase 2 validation: Monte Carlo chains on the GPU against the CPU.
 *
 *   gpu_mc_test receptor ligand [--steps a,b,...] [--tasks n] [--seed s]
 *       [--scoring name] [--approximation linear|spline] [--flexres spec]
 *
 * GPU_TEST_REFINE=n (0: all) [GPU_TEST_FIRST=k]: --minimize of the poses in
 * `ligand` (use --approximation spline --factor 10 like --minimize).  With
 * GPU_TEST_FULL_CELLS=1 every cell of the CPU's szv_grid_cache is filled
 * completely first: smina fills a cell from the first coordinate that probes
 * it, and an atom exactly on a 3 A plane (e.g. z = -6.0000 in the input)
 * leaves it with a degenerate brick and an incomplete atom list for the rest
 * of the run; the GPU's cell lists are always complete.
 * There, GPU_TEST_TRACE=1 logs every CPU evaluation (energy, gradient); compare
 * with the GPUEVAL lines of a build with -DGPU_TRACE_EVALS.
 *
 * For each step count, runs `tasks` chains seeded like parallel_mc on the CPU
 * (monte_carlo::operator()) and on the GPU, and compares the resulting
 * output containers.  The double precision GPU build must reproduce the CPU
 * chains (same energies and conformations); the float build is reported.
 */

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <boost/timer/timer.hpp>
#include "test_common.h"
#include "cache.h"
#include "parallel_mc.h"
#include "quasi_newton.h"
#include "bfgs.h"
#include "gpu_search.h"
#include "non_cache.h"
#include <cfloat>

using namespace gpusmina;

namespace
{

struct Compare
{
	int chains = 0, identical = 0;
	double max_e = 0, max_conf = 0;
	int first_diverging_chain = -1;
};

Compare compare(const std::vector<output_container>& cpu,
		const std::vector<output_container>& gpu, const FlatModel& fm,
		double tol)
{
	Compare c;
	for (size_t t = 0; t < cpu.size(); t++)
	{
		c.chains++;
		bool same = cpu[t].size() == gpu[t].size();
		for (size_t k = 0; same && k < cpu[t].size(); k++)
		{
			double de = std::abs(cpu[t][k].e - gpu[t][k].e)
					/ std::max(1.0, std::abs(cpu[t][k].e));
			c.max_e = std::max(c.max_e, de);
			std::vector<double> a(fm.conf_size), b(fm.conf_size);
			conf_to_flat(cpu[t][k].c, fm, &a[0]);
			conf_to_flat(gpu[t][k].c, fm, &b[0]);
			double dc = 0;
			for (int i = 0; i < fm.conf_size; i++)
				dc = std::max(dc, std::abs(a[i] - b[i]));
			c.max_conf = std::max(c.max_conf, dc);
			if (de > tol || dc > 1e3 * tol)
				same = false;
		}
		if (same)
			c.identical++;
		else if (c.first_diverging_chain < 0)
			c.first_diverging_chain = int(t);
	}
	return c;
}

template<typename T>
double run_gpu(model& m, const FlatModel& fm, const FlatTables& ft,
		const DeviceGrids<T>& grids, const parallel_mc& par,
		const std::vector<int>& seeds, const vec& c1, const vec& c2,
		std::vector<output_container>& out, int tpc)
{
	std::vector<const FlatModel*> fms(1, &fm);
	DeviceModels<T> dmodels;
	dmodels.upload(fms);
	DeviceTables<T> dtables;
	dtables.upload(ft);
	InterContext<T> ic;
	ic.use_cache = 1;
	ic.grids = grids.dev;
	ic.ug.initialized = 0;
	ic.slope = T(1e6);
	std::vector<ChainInput> chains;
	for (size_t t = 0; t < seeds.size(); t++)
		chains.push_back(make_chain(m, fm, 0, seeds[t], c1, c2));
	MCParams prm = mc_params(par.mc);
	std::vector<ChainResult> res;
	boost::timer::cpu_timer timer;
	run_monte_carlo(dmodels, dtables, ic, prm, chains, res, 0, tpc);
	double secs = timer.elapsed().wall / 1e9;
	out.clear();
	out.resize(seeds.size());
	for (size_t t = 0; t < res.size(); t++)
		chain_to_container(res[t], fm, out[t]);
	return secs;
}

// access to non_cache's box and cell lists
struct probe_non_cache: public non_cache
{
	probe_non_cache(szv_grid_cache& gc, const grid_dims& gd, const precalculate* p,
			fl slope) :
			non_cache(gc, gd, p, slope)
	{
	}
	const grid_dims& get_gd() const
	{
		return gd;
	}
	const szv& possibilities(const vec& c) const
	{
		return sgrid.possibilities(c);
	}
};

// quasi_newton_aux, logging every evaluation (GPU_TEST_TRACE)
struct tracing_aux
{
	model* m;
	const precalculate* p;
	const igrid* ig;
	vec v;
	const grid* ug;
	const FlatModel* fm;
	fl operator()(const conf& c, change& g)
	{
		const fl e = m->eval_deriv(*p, *ig, v, c, g, *ug);
		std::vector<double> fg(fm->change_size);
		change_to_flat(g, *fm, &fg[0]);
		std::cout << "CPUEVAL " << std::setprecision(17) << e;
		for (size_t i = 0; i < fg.size(); i++)
			std::cout << " " << fg[i];
		std::cout << "\n";
		return e;
	}
};

// main.cpp refine_structure
void cpu_refine_structure(model& m, const precalculate& prec, non_cache& nc,
		output_type& out, const vec& cap, const minimization_params& minparm,
		grid& user_grid, const FlatModel* fm = 0)
{
	change g(m.get_size());
	quasi_newton quasi_newton_par(minparm);
	const fl slope_orig = nc.getSlope();
	fl slope = 10;
	VINA_FOR(p, 5)
	{
		nc.setSlope(slope);
		if (getenv("GPU_TEST_TRACE"))
		{
			tracing_aux aux = {&m, &prec, &nc, cap, &user_grid, fm};
			out.e = bfgs(aux, out.c, g, 0.0, minparm);
		}
		else
			quasi_newton_par(m, prec, nc, out, g, cap, user_grid);
		m.set(out.c);
		if (nc.within(m))
			break;
		slope *= 10;
	}
	out.coords = m.get_heavy_atom_movable_coords();
	if (!nc.within(m))
		out.e = max_fl;
	nc.setSlope(slope_orig);
}

// --minimize on every pose of a file: CPU (main.cpp's local_only path) vs
// the GPU's batched refine in double precision.  Run with the portable-math
// build the two must agree exactly.
int refine_test(const model& initm, const std::string& ligname,
		const precalculate& prec, int tpc)
{
	MolGetter mols(initm, true);
	mols.setInputFile(ligname);
	std::vector<model*> ms;
	model m;
	while (mols.readMoleculeIntoModel(m))
		ms.push_back(new model(m));
	const int first = getenv("GPU_TEST_FIRST") ? atoi(getenv("GPU_TEST_FIRST")) : 0;
	const int count = atoi(getenv("GPU_TEST_REFINE"));
	const int last = std::min<int>(ms.size(), count > 0 ? first + count : ms.size());
	minimization_params minparm; // --minimize defaults
	minparm.maxiters = 10000;
	minparm.type = minimization_params::BFGSAccurateLineSearch;
	const vec cap(10, 10, 10);
	grid user_grid;

	std::vector<FlatModel> fms(last - first);
	std::vector<const FlatModel*> fmp;
	std::vector<std::vector<double> > confs;
	std::vector<FlatGridGeometry> boxes;
	std::vector<double> cpu_e;
	std::vector<std::vector<double> > cpu_conf;
	double cpu_s = 0;
	for (int i = first; i < last; i++)
	{
		model& mi = *ms[i];
		FlatModel& fm = fms[i - first];
		fm = gpu_model_flattener::flatten_model(mi);
		fmp.push_back(&fm);
		const grid_dims gd = mi.movable_atoms_box(4, 0.375);
		boxes.push_back(gpu_model_flattener::grid_geometry(gd));
		confs.push_back(std::vector<double>(fm.conf_size));
		conf_to_flat(mi.get_initial_conf(), fm, &confs.back()[0]);

		boost::timer::cpu_timer timer;
		szv_grid_cache gridcache(mi, prec.cutoff_sqr());
		probe_non_cache nc(gridcache, gd, &prec, 1e6);
		if (getenv("GPU_TEST_FULL_CELLS"))
		{
			// probe every 3 A cell of the box at its centre first, so that
			// szv_grid_cache fills it with the whole cell's neighbours
			int lo[3], hi[3];
			for (int k = 0; k < 3; k++)
			{
				lo[k] = int(std::floor(gd[k].begin / 3.0));
				hi[k] = int(std::floor(gd[k].end / 3.0));
			}
			for (int x = lo[0]; x <= hi[0]; x++)
				for (int y = lo[1]; y <= hi[1]; y++)
					for (int z = lo[2]; z <= hi[2]; z++)
						nc.possibilities(vec(3 * x + 1.5, 3 * y + 1.5, 3 * z + 1.5));
		}
		output_type out(mi.get_initial_conf(), max_fl);
		cpu_refine_structure(mi, prec, nc, out, cap, minparm, user_grid, &fm);
		cpu_s += timer.elapsed().wall / 1e9;
		cpu_e.push_back(out.e);
		cpu_conf.push_back(std::vector<double>(fm.conf_size));
		conf_to_flat(out.c, fm, &cpu_conf.back()[0]);
	}
	const int n = last - first;
	DeviceModels<double> dmodels;
	dmodels.upload(fmp);
	DeviceTables<double> dtables;
	dtables.upload(gpu_model_flattener::flatten_tables(prec));
	DeviceReceptor<double> drec;
	drec.upload(gpu_model_flattener::flatten_receptor(*ms[0], prec.cutoff_sqr()));
	InterContext<double> ic;
	std::memset(&ic, 0, sizeof(ic));
	ic.use_cache = 0;
	ic.rec = drec.dev;
	ic.ug.initialized = 0;
	ic.slope = 1e6;
	MinParams prm = {minparm.maxiters, 1, 0};
	std::vector<int> model_of(n);
	for (int i = 0; i < n; i++)
		model_of[i] = i;
	std::vector<double> e;
	std::vector<std::vector<double> > coords;
	const double v[3] = {cap[0], cap[1], cap[2]};
	boost::timer::cpu_timer timer;
	run_refine(dmodels, dtables, ic, prm, v, model_of, confs, e, coords, tpc,
			std::vector<unsigned>(n, minparm.maxiters), boxes);
	const double gpu_s = timer.elapsed().wall / 1e9;
	int same = 0, close = 0;
	for (int i = 0; i < n; i++)
	{
		double dc = 0;
		for (size_t k = 0; k < confs[i].size(); k++)
			dc = std::max(dc, std::abs(confs[i][k] - cpu_conf[i][k]));
		const double ge = e[i] >= DBL_MAX ? max_fl : e[i];
		const double de = std::abs(ge - cpu_e[i]);
		const bool s = de <= 1e-9 * std::max(1.0, std::abs(cpu_e[i])) && dc < 1e-6;
		same += s;
		close += de < 0.01;
		if (!s)
			std::cout << "pose " << first + i << ": cpu e " << std::setprecision(10)
					<< cpu_e[i] << " gpu e " << ge << " dconf " << dc << "\n";
	}
	std::cout << "refine fp64: " << same << "/" << n << " identical, " << close
			<< "/" << n << " within 0.01; cpu " << cpu_s << " s, gpu " << gpu_s
			<< " s\n";
	for (size_t i = 0; i < ms.size(); i++)
		delete ms[i];
	return same == n ? 0 : 1;
}

std::vector<int> parse_list(const std::string& s)
{
	std::vector<int> v;
	std::stringstream ss(s);
	std::string tok;
	while (std::getline(ss, tok, ','))
		v.push_back(atoi(tok.c_str()));
	return v;
}

} // namespace

int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		std::cerr << "usage: gpu_mc_test receptor ligand [options]\n";
		return 2;
	}
	std::string recname = argv[1], ligname = argv[2];
	std::string scoring, approx = "linear", flexres, autobox;
	std::vector<int> steps = parse_list("1,10,100");
	int tasks = 4, seed = 42, tpc = 32;
	double factor = 32;
	for (int i = 3; i + 1 < argc; i += 2)
	{
		std::string o = argv[i];
		if (o == "--steps")
			steps = parse_list(argv[i + 1]);
		else if (o == "--tasks")
			tasks = atoi(argv[i + 1]);
		else if (o == "--tpc")
			tpc = atoi(argv[i + 1]);
		else if (o == "--seed")
			seed = atoi(argv[i + 1]);
		else if (o == "--scoring")
			scoring = argv[i + 1];
		else if (o == "--approximation")
			approx = argv[i + 1];
		else if (o == "--factor")
			factor = atof(argv[i + 1]);
		else if (o == "--flexres")
			flexres = argv[i + 1];
		else if (o == "--autobox_ligand")
			autobox = argv[i + 1];
		else
		{
			std::cerr << "unknown option " << o << "\n";
			return 2;
		}
	}

	set_fixed_rotable_hydrogens(true);
	tee log(true);
	FlexInfo finfo(flexres, -1, "", log);
	model initm;
	gputest::load_receptor(recname, finfo, initm);
	MolGetter mols(initm, true);
	mols.setInputFile(ligname);
	model m;
	if (!mols.readMoleculeIntoModel(m))
		return 2;

	custom_terms t;
	gputest::setup_terms(t, scoring, "");
	weighted_terms wt(&t, t.weights());
	boost::shared_ptr<precalculate> prec = gputest::make_prec(wt, approx, factor);
	if (getenv("GPU_TEST_REFINE"))
		return refine_test(initm, ligname, *prec, tpc);
	grid_dims gd = gputest::ligand_box(m, 4);
	if (autobox.size())
	{
		// main.cpp setup_autobox (+4 A) and grid_dims construction
		using namespace OpenBabel;
		obmol_opener opener;
		OBConversion conv;
		opener.openForInput(conv, autobox);
		OBMol mol;
		Box b;
		while (conv.Read(&mol))
			b.add_ligand_box(mol);
		vec center((b.max_x + b.min_x) / 2.0, (b.max_y + b.min_y) / 2.0,
				(b.max_z + b.min_z) / 2.0);
		b.expand(4);
		vec span(b.max_x - b.min_x, b.max_y - b.min_y, b.max_z - b.min_z);
		VINA_FOR_IN(i, gd)
		{
			gd[i].n = sz(std::ceil(span[i] / 0.375));
			fl real_span = 0.375 * gd[i].n;
			gd[i].begin = center[i] - real_span / 2;
			gd[i].end = gd[i].begin + real_span;
		}
	}
	vec c1(gd[0].begin, gd[1].begin, gd[2].begin);
	vec c2(gd[0].end, gd[1].end, gd[2].end);
	grid user_grid;

	// main_procedure's search parameters
	parallel_mc par;
	sz heuristic = m.num_movable_atoms()
			+ 10 * m.get_size().num_degrees_of_freedom();
	const unsigned full_steps = unsigned(70 * 3 * (50 + heuristic) / 2);
	par.mc.ssd_par.evals = unsigned((25 + m.num_movable_atoms()) / 3);
	minimization_params minparm;
	minparm.maxiters = par.mc.ssd_par.evals;
	par.mc.ssd_par.minparm = minparm;
	par.mc.min_rmsd = 1.0;
	par.mc.num_saved_mins = 20;
	par.mc.hunt_cap = vec(10, 10, 10);

	cache cpu_cache("scoring_function_version001", gd, 1e6);
	std::vector<smt> types;
	m.get_movable_atom_types(types);
	cpu_cache.populate(m, *prec, types, user_grid);

	FlatModel fm = gpu_model_flattener::flatten_model(m);
	FlatTables ft = gpu_model_flattener::flatten_tables(*prec);
	FlatReceptor frg = gpu_model_flattener::flatten_receptor_for_grid(m, gd,
			prec->cutoff_sqr());
	FlatGridGeometry geo = gpu_model_flattener::grid_geometry(gd);
	FlatUserGrid fug = gpu_model_flattener::flatten_user_grid(user_grid);
	std::vector<int> itypes(types.begin(), types.end());
	DeviceGrids<double> g64;
	g64.build(frg, geo, ft, itypes, fug, 1e6);
	DeviceGrids<float> g32;
	g32.build(frg, geo, ft, itypes, fug, 1e6);

	rng master(static_cast<rng::result_type>(seed));
	std::vector<int> seeds;
	for (int i = 0; i < tasks; i++)
		seeds.push_back(random_int(0, 1000000, master));

	{
		int maxdeg = 0;
		for (int i = 0; i < fm.num_movable; i++)
			maxdeg = std::max(maxdeg, fm.nbr_start[i + 1] - fm.nbr_start[i]);
		std::cout << "pairs " << fm.pairs.size() << "  nbr entries "
				<< fm.nbr.size() << "  max per atom " << maxdeg << "  frames "
				<< fm.frames.size() << "  levels " << fm.num_levels() << "\n";
	}
	{
		// fingerprint of the parsed model (tree, atom order, pairs)
		unsigned long long h = 1469598103934665603ULL;
		auto mix = [&](long long v) { h = (h ^ (unsigned long long) v) * 1099511628211ULL; };
		for (size_t i = 0; i < fm.atom_type.size(); i++)
			mix(fm.atom_type[i]), mix(fm.atom_frame[i]), mix(llround(fm.atom_local[3 * i] * 1e4));
		for (size_t i = 0; i < fm.pairs.size(); i++)
			mix(fm.pairs[i].a), mix(fm.pairs[i].b);
		for (size_t i = 0; i < fm.frames.size(); i++)
			mix(fm.frames[i].parent), mix(fm.frames[i].begin), mix(fm.frames[i].end);
		std::cout << "model fingerprint " << std::hex << h << std::dec << "\n";
	}
	std::cout << "box " << gd[0].begin << " " << gd[1].begin << " "
			<< gd[2].begin << " / " << gd[0].end << " " << gd[1].end << " "
			<< gd[2].end << "\n";
	std::cout << "atoms " << m.num_movable_atoms() << "  dof "
			<< m.get_size().num_degrees_of_freedom() << "  full run "
			<< full_steps << " steps, bfgs maxiters " << minparm.maxiters << "\n";

	if (getenv("GPU_TEST_BFGS"))
	{
		// isolate bfgs: CPU quasi_newton vs GPU bfgs from the same confs
		rng gen(1);
		std::vector<conf> cs;
		std::vector<std::vector<double> > flat;
		for (int i = 0; i < 200; i++)
		{
			conf c(m.get_size());
			c.randomize(c1, c2, gen);
			cs.push_back(c);
			flat.push_back(std::vector<double>(fm.conf_size));
			conf_to_flat(c, fm, &flat.back()[0]);
		}
		if (getenv("GPU_TEST_TRACE"))
		{
			int idx = atoi(getenv("GPU_TEST_TRACE"));
			std::swap(cs[0], cs[idx]);
			std::swap(flat[0], flat[idx]);
		}
		std::vector<const FlatModel*> fms(1, &fm);
		DeviceModels<double> dmodels;
		dmodels.upload(fms);
		DeviceTables<double> dtables;
		dtables.upload(ft);
		InterContext<double> ic;
		ic.use_cache = 1;
		ic.grids = g64.dev;
		ic.ug.initialized = 0;
		ic.slope = 1e6;
		MCParams prm = mc_params(par.mc);
		const double v[3] = {10, 10, 10};
		std::vector<int> mo(cs.size(), 0);
		std::vector<double> ge;
		run_bfgs(dmodels, dtables, ic, prm.min, v, mo, flat, ge);
		if (getenv("GPU_TEST_TRACE"))
		{
			// log the CPU's evaluation sequence for conf 0 (compare with
			// a GPU build defining GPU_TRACE_EVALS)
			struct logging_aux
			{
				model* m;
				const precalculate* p;
				const igrid* ig;
				grid* ug;
				fl operator()(const conf& c, change& g)
				{
					fl e = m->eval_deriv(*p, *ig, vec(10, 10, 10), c, g, *ug);
					std::cout << "CPUEVAL " << std::setprecision(17) << e << "\n";
					return e;
				}
			} aux = {&m, prec.get(), &cpu_cache, &user_grid};
			conf c = cs[0];
			change g(m.get_size());
			bfgs(aux, c, g, 0.0, par.mc.ssd_par.minparm);
		}
		quasi_newton qn(par.mc.ssd_par.minparm);
		int same = 0;
		for (size_t i = 0; i < cs.size(); i++)
		{
			output_type o(cs[i], 0);
			change g(m.get_size());
			qn(m, *prec, cpu_cache, o, g, vec(10, 10, 10), user_grid);
			std::vector<double> a(fm.conf_size);
			conf_to_flat(o.c, fm, &a[0]);
			double dc = 0;
			for (int k = 0; k < fm.conf_size; k++)
				dc = std::max(dc, std::abs(a[k] - flat[i][k]));
			bool s = std::abs(o.e - ge[i]) < 1e-9 * std::max(1.0, std::abs(o.e))
					&& dc < 1e-6;
			same += s;
			if (!s && same + 5 > int(i))
				std::cout << "bfgs conf " << i << " cpu e " << o.e << " gpu e "
						<< ge[i] << " dconf " << dc << "\n";
		}
		std::cout << "bfgs fp64: " << same << "/" << cs.size() << " identical\n";
	}

	if (getenv("GPU_TEST_DETERMINISM"))
	{
		// float build twice with the same inputs must agree exactly
		par.mc.num_steps = steps[0];
		std::vector<output_container> a, b;
		run_gpu(m, fm, ft, g32, par, seeds, c1, c2, a, tpc);
		int diff = 0;
		for (int rep = 0; rep < 5; rep++)
		{
			run_gpu(m, fm, ft, g32, par, seeds, c1, c2, b, tpc);
			for (int t = 0; t < tasks; t++)
			{
				bool same = a[t].size() == b[t].size();
				for (size_t k = 0; same && k < a[t].size(); k++)
					same = a[t][k].e == b[t][k].e;
				diff += !same;
			}
		}
		std::cout << "fp32 determinism: " << diff << " differing chain results in "
				<< 5 * tasks << "\n";
		return diff != 0;
	}

	if (getenv("GPU_TEST_REFINE_DET"))
	{
		// refine_structure on the GPU twice from the same poses
		int tpcs[2] = {32, 128};
		std::vector<const FlatModel*> fms(1, &fm);
		DeviceModels<float> dmodels;
		dmodels.upload(fms);
		DeviceTables<float> dtables;
		dtables.upload(ft);
		FlatReceptor fr = gpu_model_flattener::flatten_receptor(m,
				prec->cutoff_sqr());
		DeviceReceptor<float> drec;
		drec.upload(fr);
		InterContext<float> ic;
		std::memset(&ic, 0, sizeof(ic));
		ic.use_cache = 0;
		ic.rec = drec.dev;
		ic.box = make_box<float>(geo);
		ic.ug.initialized = 0;
		rng gen(5);
		std::vector<std::vector<double> > start;
		for (int i = 0; i < 64; i++)
		{
			conf c(m.get_size());
			c.randomize(c1, c2, gen);
			start.push_back(std::vector<double>(fm.conf_size));
			conf_to_flat(c, fm, &start.back()[0]);
		}
		MCParams prm = mc_params(par.mc);
		const double v[3] = {1000, 1000, 1000};
		for (int ti = 0; ti < 2; ti++)
		{
			int diff = 0;
			std::vector<double> e0;
			for (int rep = 0; rep < 4; rep++)
			{
				std::vector<std::vector<double> > confs = start, coords;
				std::vector<double> e;
				run_refine(dmodels, dtables, ic, prm.min, v,
						std::vector<int>(start.size(), 0), confs, e, coords, tpcs[ti]);
				if (rep == 0)
					e0 = e;
				else
					for (size_t i = 0; i < e.size(); i++)
						diff += e[i] != e0[i];
			}
			std::cout << "refine tpc " << tpcs[ti] << ": " << diff
					<< " differing energies in " << 3 * start.size() << "\n";
		}
		return 0;
	}

	if (getenv("GPU_TEST_THROUGHPUT"))
	{
		// GPU only: many chains, report chain-steps per second
		int n = atoi(getenv("GPU_TEST_THROUGHPUT"));
		par.mc.num_steps = steps[0];
		rng m2(static_cast<rng::result_type>(seed));
		std::vector<int> many;
		for (int i = 0; i < n; i++)
			many.push_back(random_int(0, 1000000, m2));
		std::vector<output_container> o;
		double t = run_gpu(m, fm, ft, g32, par, many, c1, c2, o, tpc);
		std::cout << n << " chains x " << steps[0] << " steps, tpc " << tpc
				<< ": " << t << " s, " << n * double(steps[0]) / t
				<< " chain-steps/s\n";
		return 0;
	}

	bool ok = true;
	for (size_t si = 0; si < steps.size(); si++)
	{
		par.mc.num_steps = steps[si];
		std::vector<output_container> cpu(tasks);
		boost::timer::cpu_timer timer;
		for (int i = 0; i < tasks; i++)
		{
			model mt(m);
			rng gen(static_cast<rng::result_type>(seeds[i]));
			par.mc(mt, cpu[i], *prec, cpu_cache, c1, c2, NULL, gen, user_grid);
		}
		double cpu_s = timer.elapsed().wall / 1e9;
		std::vector<output_container> o64, o32;
		double t64 = run_gpu(m, fm, ft, g64, par, seeds, c1, c2, o64, tpc);
		double t32 = run_gpu(m, fm, ft, g32, par, seeds, c1, c2, o32, tpc);
		Compare c = compare(cpu, o64, fm, 1e-9);
		Compare d = compare(cpu, o32, fm, 1e-4);
		double cpu_best = max_fl, g32_best = max_fl;
		for (int i = 0; i < tasks; i++)
		{
			if (!cpu[i].empty())
				cpu_best = std::min(cpu_best, cpu[i][0].e);
			if (!o32[i].empty())
				g32_best = std::min(g32_best, o32[i][0].e);
		}
		if (getenv("GPU_TEST_DIST"))
		{
			// distribution of per-chain best energies
			std::vector<double> bc, bg;
			for (int i = 0; i < tasks; i++)
			{
				if (!cpu[i].empty())
					bc.push_back(cpu[i][0].e);
				if (!o32[i].empty())
					bg.push_back(o32[i][0].e);
			}
			std::sort(bc.begin(), bc.end());
			std::sort(bg.begin(), bg.end());
			std::cout << std::fixed << std::setprecision(2)
					<< "  best-E quartiles cpu " << bc[0] << " " << bc[bc.size() / 4]
					<< " " << bc[bc.size() / 2] << " " << bc[3 * bc.size() / 4]
					<< " " << bc.back() << " | gpu32 " << bg[0] << " "
					<< bg[bg.size() / 4] << " " << bg[bg.size() / 2] << " "
					<< bg[3 * bg.size() / 4] << " " << bg.back() << "\n";
			int pc = 0, pg = 0;
			for (size_t i = 0; i < bc.size(); i++)
				pc += bc[i] > -5;
			for (size_t i = 0; i < bg.size(); i++)
				pg += bg[i] > -5;
			std::cout << "  chains with best E > -5: cpu " << pc << "/" << bc.size()
					<< "  gpu32 " << pg << "/" << bg.size() << "\n";
		}
		bool pass = c.identical == c.chains;
		ok &= pass;
		std::cout << std::setw(6) << steps[si] << " steps  fp64: "
				<< c.identical << "/" << c.chains << " chains identical (max dE "
				<< std::scientific << std::setprecision(1) << c.max_e
				<< ", dconf " << c.max_conf << ")" << (pass ? "" : "  FAIL")
				<< "  | fp32: " << d.identical << "/" << d.chains
				<< " identical, best E " << std::fixed << std::setprecision(3)
				<< g32_best << " vs cpu " << cpu_best << "  | time cpu "
				<< std::setprecision(2) << cpu_s << "s gpu64 " << t64
				<< "s gpu32 " << t32 << "s\n";
	}
	std::cout << (ok ? "PASS" : "FAIL") << "\n";
	return ok ? 0 : 1;
}
