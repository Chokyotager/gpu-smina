#include "gpu_search.h"
#include <sstream>

namespace gpusmina
{

MTState export_rng(const rng& g)
{
	std::stringstream ss;
	ss << g;
	MTState s;
	for (int j = 0; j < MT_N; j++)
		ss >> s.x[j];
	VINA_CHECK(!ss.fail());
	s.i = MT_N; // boost's operator>> sets i = n after reading the state
	return s;
}

ChainInput make_chain(const model& m, const FlatModel& fm, int model_index,
		int seed, const vec& corner1, const vec& corner2)
{
	ChainInput in;
	in.model = model_index;
	rng generator(static_cast<rng::result_type>(seed));
	output_type tmp(m.get_size(), 0);
	tmp.c.randomize(corner1, corner2, generator);
	in.conf.assign(fm.conf_size, 0);
	conf_to_flat(tmp.c, fm, &in.conf[0]);
	in.rng = export_rng(generator);
	return in;
}

MCParams mc_params(const monte_carlo& mc)
{
	MCParams p;
	p.num_steps = mc.num_steps;
	p.temperature = mc.temperature;
	p.mutation_amplitude = mc.mutation_amplitude;
	p.min_rmsd = mc.min_rmsd;
	p.num_saved_mins = int(mc.num_saved_mins);
	for (int i = 0; i < 3; i++)
	{
		p.hunt_cap[i] = mc.hunt_cap[i];
		p.authentic_v[i] = 1000; // monte_carlo::operator() authentic_v
	}
	minimization_params mp = mc.ssd_par.minparm;
	if (mp.maxiters == 0)
		mp.maxiters = mc.ssd_par.evals;
	p.min.maxiters = mp.maxiters;
	p.min.accurate_line = mp.type == minimization_params::BFGSAccurateLineSearch;
	p.min.early_term = mp.early_term;
	p.visited = 0;
	return p;
}

void chain_to_container(const ChainResult& r, const FlatModel& fm,
		output_container& out)
{
	for (size_t k = 0; k < r.e.size(); k++)
	{
		conf c;
		flat_to_conf(&r.conf[k][0], fm, c);
		output_type* o = new output_type(c, r.e[k]);
		for (size_t a = 0; a + 2 < r.coords[k].size(); a += 3)
			o->coords.push_back(vec(r.coords[k][a], r.coords[k][a + 1],
					r.coords[k][a + 2]));
		out.push_back(o);
	}
}

} // namespace gpusmina
