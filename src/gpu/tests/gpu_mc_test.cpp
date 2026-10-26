/*
 * Phase 2 validation: Monte Carlo chains on the GPU against the CPU.
 *
 *   gpu_mc_test receptor ligand [--steps a,b,...] [--tasks n] [--seed s]
 *       [--scoring name] [--approximation linear|spline] [--flexres spec]
 *
 * For each step count, runs `tasks` chains seeded like parallel_mc on the CPU
 * (monte_carlo::operator()) and on the GPU, and compares the resulting
 * output containers.  The double precision GPU build must reproduce the CPU
 * chains (same energies and conformations); the float build is reported.
 */

#include <cstdlib>
#include <iomanip>
#include <boost/timer/timer.hpp>
#include "test_common.h"
#include "cache.h"
#include "parallel_mc.h"
#include "quasi_newton.h"
#include "bfgs.h"
#include "gpu_search.h"

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
		std::vector<output_container>& out)
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
	run_monte_carlo(dmodels, dtables, ic, prm, chains, res);
	double secs = timer.elapsed().wall / 1e9;
	out.clear();
	out.resize(seeds.size());
	for (size_t t = 0; t < res.size(); t++)
		chain_to_container(res[t], fm, out[t]);
	return secs;
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
	std::string scoring, approx = "linear", flexres;
	std::vector<int> steps = parse_list("1,10,100");
	int tasks = 4, seed = 42;
	double factor = 32;
	for (int i = 3; i + 1 < argc; i += 2)
	{
		std::string o = argv[i];
		if (o == "--steps")
			steps = parse_list(argv[i + 1]);
		else if (o == "--tasks")
			tasks = atoi(argv[i + 1]);
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
	grid_dims gd = gputest::ligand_box(m, 4);
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
		double t64 = run_gpu(m, fm, ft, g64, par, seeds, c1, c2, o64);
		double t32 = run_gpu(m, fm, ft, g32, par, seeds, c1, c2, o32);
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
