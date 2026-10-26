/*
 * Phase 1 validation: GPU grid maps and model::eval_deriv against the CPU.
 *
 *   gpu_eval_test receptor ligand [--scoring name] [--custom_scoring file]
 *       [--approximation linear|spline] [--factor f] [--flexres spec]
 *       [--nconf n] [--seed s]
 *
 * Prints maximum relative errors per check and exits non-zero if any check
 * exceeds the acceptance tolerance (FP32 1e-4, FP64 1e-9).
 */

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <type_traits>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include "test_common.h"
#include "cache.h"
#include "non_cache.h"
#include "szv_grid.h"
#include "gpu_flatten.h"
#include "gpu_kernels.h"

using namespace gpusmina;

namespace
{

struct Stat
{
	// per conformation: relative energy error, relative gradient error
	// (normalized by the largest CPU gradient component), and whether the
	// conformation sits within float precision of a discontinuity of the CPU
	// scoring function (grid cell face, hard cutoff)
	std::vector<double> de, dg;
	std::vector<bool> flagged;
	int worst_e = -1, worst_g = -1;
	void add(int conf, double ecpu, double egpu,
			const std::vector<double>& gcpu, const double* ggpu,
			bool flag = false)
	{
		double e = std::abs(ecpu - egpu) / std::max(1.0, std::abs(ecpu));
		double gscale = 1.0, g = 0;
		for (size_t k = 0; k < gcpu.size(); k++)
			gscale = std::max(gscale, std::abs(gcpu[k]));
		for (size_t k = 0; k < gcpu.size(); k++)
			g = std::max(g, std::abs(gcpu[k] - ggpu[k]) / gscale);
		de.push_back(e);
		dg.push_back(g);
		flagged.push_back(flag);
	}
};

// FP64 must match to `tol` everywhere.  FP32 is limited by float coordinate
// resolution (~5e-6 A) amplified by the local steepness of the function:
// at least 99% of evaluations must be within `tol` and every evaluation not
// at a discontinuity within 10 * tol.
bool report(const std::string& what, const Stat& s, double tol, bool strict)
{
	size_t n = s.de.size(), within = 0, nflag = 0;
	double max_e = 0, max_g = 0, flag_e = 0, flag_g = 0;
	for (size_t i = 0; i < n; i++)
	{
		if (s.de[i] <= tol && s.dg[i] <= tol)
			within++;
		if (s.flagged[i] && !strict)
		{
			nflag++;
			flag_e = std::max(flag_e, s.de[i]);
			flag_g = std::max(flag_g, s.dg[i]);
		}
		else
		{
			max_e = std::max(max_e, s.de[i]);
			max_g = std::max(max_g, s.dg[i]);
		}
	}
	double frac = n ? double(within) / n : 1;
	bool ok = strict ? (max_e <= tol && max_g <= tol)
			: (frac >= 0.99 && max_e <= 10 * tol && max_g <= 10 * tol);
	std::cout << std::left << std::setw(40) << what << std::fixed
			<< std::setprecision(1) << std::right << std::setw(6)
			<< 100 * frac << "% within " << std::scientific
			<< std::setprecision(0) << tol << "  max E " << std::setprecision(2)
			<< max_e << " grad " << max_g;
	if (nflag)
		std::cout << "  | " << nflag << " at discontinuity: E " << flag_e
				<< " grad " << flag_g;
	std::cout << (ok ? "  ok" : "  FAIL") << "\n";
	return ok;
}

// conformations: half fully random in the box, half perturbed around the
// input pose (the regime minimization and refinement actually see)
std::vector<conf> make_confs(const model& m, const grid_dims& gd, int n,
		rng& gen)
{
	std::vector<conf> out;
	vec c1(gd[0].begin, gd[1].begin, gd[2].begin);
	vec c2(gd[0].end, gd[1].end, gd[2].end);
	conf init = m.get_initial_conf();
	for (int i = 0; i < n; i++)
	{
		conf c = init;
		if (i % 2 == 0)
			c.randomize(c1, c2, gen);
		else
		{
			VINA_FOR_IN(l, c.ligands)
			{
				c.ligands[l].rigid.position += random_inside_sphere(gen);
				vec rot = 0.3 * random_inside_sphere(gen);
				quaternion_increment(c.ligands[l].rigid.orientation, rot);
				VINA_FOR_IN(t, c.ligands[l].torsions)
					c.ligands[l].torsions[t] = random_fl(-0.5, 0.5, gen);
			}
			VINA_FOR_IN(f, c.flex)
				VINA_FOR_IN(t, c.flex[f].torsions)
					c.flex[f].torsions[t] = random_fl(-pi, pi, gen);
		}
		out.push_back(c);
	}
	return out;
}

template<typename T>
bool run_precision(const char* label, double tol, model& m,
		const precalculate& prec, const FlatModel& fm, const FlatTables& ft,
		const grid_dims& gd, cache& cpu_cache, szv_grid_cache& gcache,
		const std::vector<conf>& confs, grid& user_grid)
{
	bool ok = true;
	const double cache_slope = 1e6;

	// grids
	std::vector<smt> types;
	m.get_movable_atom_types(types);
	std::vector<int> itypes(types.begin(), types.end());
	FlatReceptor frg = gpu_model_flattener::flatten_receptor_for_grid(m, gd,
			prec.cutoff_sqr());
	FlatGridGeometry geo = gpu_model_flattener::grid_geometry(gd);
	FlatUserGrid fug = gpu_model_flattener::flatten_user_grid(user_grid);
	DeviceGrids<T> dgrids;
	dgrids.build(frg, geo, ft, itypes, fug, cache_slope);
	double gmax = 0;
	for (size_t i = 0; i < itypes.size(); i++)
	{
		int t = itypes[i];
		std::vector<T> d, cd;
		dgrids.download(t, d, cd);
		const grid& cg = gpu_model_flattener::cache_grid(cpu_cache, t);
		const array3d<fl>& a = gpu_model_flattener::grid_data(cg);
		const array3d<fl>& ca = gpu_model_flattener::grid_chargedata(cg);
		VINA_FOR(z, a.dim2())
			VINA_FOR(y, a.dim1())
				VINA_FOR(x, a.dim0())
				{
					size_t idx = x + a.dim0() * (y + a.dim1() * z);
					double ref = a(x, y, z);
					gmax = std::max(gmax, std::abs(ref - d[idx])
							/ std::max(1.0, std::abs(ref)));
					if (!cd.empty())
					{
						double cref = ca(x, y, z);
						gmax = std::max(gmax, std::abs(cref - cd[idx])
								/ std::max(1.0, std::abs(cref)));
					}
				}
	}
	std::cout << std::left << std::setw(46)
			<< (std::string(label) + " grid maps") << " max rel err    "
			<< std::scientific << std::setprecision(2) << gmax
			<< (gmax <= tol ? "  ok" : "  FAIL") << "\n";
	ok &= gmax <= tol;

	// models, tables, receptor
	std::vector<const FlatModel*> fms(1, &fm);
	DeviceModels<T> dmodels;
	dmodels.upload(fms);
	DeviceTables<T> dtables;
	dtables.upload(ft);
	FlatReceptor fr = gpu_model_flattener::flatten_receptor(m,
			prec.cutoff_sqr());
	DeviceReceptor<T> drec;
	drec.upload(fr);
	DeviceUserGrid<T> dug;
	dug.upload(fug);

	// Float coordinates (~1e-6 A off) can legitimately land on the other
	// side of a discontinuity of the CPU scoring function: a grid cell face
	// (trilinear gradients jump) or the hard distance cutoff (terms are not
	// zero at r = cutoff).  Flag confs that sit within float precision of one.
	std::vector<bool> on_face(confs.size(), false), at_cutoff(confs.size(),
			false);
	if (!std::is_same<T, double>::value)
	{
		const double c2 = prec.cutoff_sqr(), tol2 = 2e-4;
		FlatReceptor fr0 = gpu_model_flattener::flatten_receptor(m, c2);
		for (size_t i = 0; i < confs.size(); i++)
		{
			m.set(confs[i]);
			const vecv& cs = m.coordinates();
			for (int a = 0; a < fm.num_movable; a++)
			{
				int t = fm.atom_type[a];
				if (t >= NUM_TYPES || type_is_hydrogen(t))
					continue;
				vec adj = cs[a];
				for (int k = 0; k < 3; k++)
				{
					double sd = (cs[a][k] - geo.init[k]) * geo.factor[k];
					double frac = sd - std::floor(sd);
					if (sd > 0 && sd < geo.dim_fl_minus_1[k]
							&& std::min(frac, 1 - frac) < 1e-4)
						on_face[i] = true;
					adj[k] = std::min(std::max(adj[k], fl(gd[k].begin)),
							fl(gd[k].end));
				}
				for (size_t r = 0; r < fr0.type.size(); r++)
				{
					vec rc(fr0.coords[3 * r], fr0.coords[3 * r + 1],
							fr0.coords[3 * r + 2]);
					if (std::abs(vec_distance_sqr(adj, rc) - c2) < tol2)
						at_cutoff[i] = true;
				}
			}
			for (size_t p = 0; p < fm.pairs.size(); p++)
				if (std::abs(vec_distance_sqr(cs[fm.pairs[p].a],
						cs[fm.pairs[p].b]) - c2) < tol2)
					at_cutoff[i] = true;
		}
	}

	std::vector<double> flat(confs.size() * fm.conf_size);
	for (size_t i = 0; i < confs.size(); i++)
		conf_to_flat(confs[i], fm, &flat[i * fm.conf_size]);

	const double vs[3] = {10, 1000, 1e3};
	for (int backend = 0; backend < 2; backend++)
	{
		for (int vi = 0; vi < 2; vi++)
		{
			const double vv = vs[vi];
			const double v[3] = {vv, vv, vv};
			const double slopes[2] = {cache_slope, 10};
			for (int si = 0; si < (backend == 0 ? 1 : 2); si++)
			{
				double slope = slopes[si];
				InterContext<T> ic;
				ic.use_cache = backend == 0;
				ic.grids = dgrids.dev;
				ic.rec = drec.dev;
				ic.box = make_box<T>(geo);
				ic.ug = dug.dev;
				ic.slope = T(slope);
				std::vector<double> ge, gg;
				std::vector<double> gc;
				bool noerr = eval_confs(dmodels, 0, dtables, ic, v, flat, ge, gg,
						&gc);
				if (backend == 0 && vi == 0)
				{
					double cmax = 0;
					for (size_t i = 0; i < confs.size(); i++)
					{
						m.set(confs[i]);
						const vecv& cs = m.coordinates();
						for (int a = 0; a < fm.num_movable; a++)
							for (int k = 0; k < 3; k++)
								cmax = std::max(cmax, std::abs(cs[a][k]
										- gc[(i * fm.num_movable + a) * 3 + k]));
					}
					std::cout << std::left << std::setw(46)
							<< (std::string(label) + " atom coordinates")
							<< " max abs err    " << std::scientific
							<< std::setprecision(2) << cmax << " A\n";
				}

				non_cache nc(gcache, gd, &prec, slope);
				Stat st;
				bool cpu_err = false;
				for (size_t i = 0; i < confs.size(); i++)
				{
					change g(m.get_size());
					fl e;
					try
					{
						if (backend == 0)
							e = m.eval_deriv(prec, cpu_cache, vec(vv, vv, vv),
									confs[i], g, user_grid);
						else
							e = m.eval_deriv(prec, nc, vec(vv, vv, vv), confs[i],
									g, user_grid);
					} catch (std::runtime_error&)
					{
						cpu_err = true;
						continue;
					}
					std::vector<double> gf(fm.change_size);
					change_to_flat(g, fm, &gf[0]);
					st.add(int(i), e, ge[i], gf, &gg[i * fm.change_size],
							(backend == 0 && on_face[i]) || at_cutoff[i]);
				}
				std::stringstream name;
				name << label << (backend == 0 ? " cache" : " non_cache")
						<< " v=" << vv << " slope=" << slope;
				ok &= report(name.str(), st, tol, std::is_same<T, double>::value);
				if (cpu_err != !noerr)
				{
					std::cout << "  overlap error mismatch\n";
					ok = false;
				}
			}
		}
	}
	return ok;
}

} // namespace

int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		std::cerr << "usage: gpu_eval_test receptor ligand [options]\n";
		return 2;
	}
	std::string recname = argv[1], ligname = argv[2];
	std::string scoring, custom, approx = "linear", flexres, user_grid_file;
	double factor = 32;
	int nconf = 400;
	int seed = 7;
	for (int i = 3; i + 1 < argc; i += 2)
	{
		std::string o = argv[i];
		if (o == "--scoring")
			scoring = argv[i + 1];
		else if (o == "--custom_scoring")
			custom = argv[i + 1];
		else if (o == "--approximation")
			approx = argv[i + 1];
		else if (o == "--factor")
			factor = atof(argv[i + 1]);
		else if (o == "--flexres")
			flexres = argv[i + 1];
		else if (o == "--user_grid")
			user_grid_file = argv[i + 1];
		else if (o == "--nconf")
			nconf = atoi(argv[i + 1]);
		else if (o == "--seed")
			seed = atoi(argv[i + 1]);
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
	{
		std::cerr << "could not read ligand\n";
		return 2;
	}

	custom_terms t;
	gputest::setup_terms(t, scoring, custom);
	weighted_terms wt(&t, t.weights());
	boost::shared_ptr<precalculate> prec = gputest::make_prec(wt, approx, factor);

	grid_dims gd = gputest::ligand_box(m, 4);
	grid user_grid;
	if (user_grid_file.size())
	{
		// main.cpp setup_user_gd + grid::init
		std::ifstream uin(user_grid_file.c_str());
		std::string line;
		std::vector<std::string> tok;
		for (int k = 0; k < 3; k++)
			std::getline(uin, line);
		std::getline(uin, line);
		boost::algorithm::split(tok, line, boost::algorithm::is_space());
		const fl granularity = ::atof(tok[1].c_str());
		std::getline(uin, line);
		boost::algorithm::split(tok, line, boost::algorithm::is_space());
		vec span((::atof(tok[1].c_str()) + 1) * granularity,
				(::atof(tok[2].c_str()) + 1) * granularity,
				(::atof(tok[3].c_str()) + 1) * granularity);
		std::getline(uin, line);
		boost::algorithm::split(tok, line, boost::algorithm::is_space());
		vec center(::atof(tok[1].c_str()) + 0.5 * granularity,
				::atof(tok[2].c_str()) + 0.5 * granularity,
				::atof(tok[3].c_str()) + 0.5 * granularity);
		grid_dims ugd;
		VINA_FOR_IN(i, ugd)
		{
			ugd[i].n = sz(std::ceil(span[i] / granularity));
			fl real_span = granularity * ugd[i].n;
			ugd[i].begin = center[i] - real_span / 2;
			ugd[i].end = ugd[i].begin + real_span;
		}
		user_grid.init(ugd, uin, 1.0);
	}

	std::cout << "atoms " << m.num_movable_atoms() << "  dof "
			<< m.get_size().num_degrees_of_freedom() << "  flex " << m.num_flex()
			<< "  scoring " << (custom.size() ? custom : (scoring.size() ? scoring : "vina"))
			<< "  approx " << approx << "\n";

	cache cpu_cache("scoring_function_version001", gd, 1e6);
	std::vector<smt> types;
	m.get_movable_atom_types(types);
	cpu_cache.populate(m, *prec, types, user_grid);
	szv_grid_cache gcache(m, prec->cutoff_sqr());

	FlatModel fm = gpu_model_flattener::flatten_model(m);
	FlatTables ft = gpu_model_flattener::flatten_tables(*prec);

	rng gen(seed);
	std::vector<conf> confs = make_confs(m, gd, nconf, gen);

	bool ok = true;
	ok &= run_precision<double>("fp64", 1e-9, m, *prec, fm, ft, gd, cpu_cache,
			gcache, confs, user_grid);
	ok &= run_precision<float>("fp32", 1e-4, m, *prec, fm, ft, gd, cpu_cache,
			gcache, confs, user_grid);
	std::cout << (ok ? "PASS" : "FAIL") << "\n";
	return ok ? 0 : 1;
}
