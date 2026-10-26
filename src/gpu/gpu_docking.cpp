#include "gpu_docking.h"
#include "gpu_search.h"
#include "coords.h"
#include <cuda_runtime.h>
#include <boost/progress.hpp>
#include <boost/thread/thread.hpp>
#include <sstream>
#include <cfloat>
#include <cstring>
#include <type_traits>

namespace gpusmina
{

std::vector<int> parse_devices(const std::string& s)
{
	std::vector<int> d;
	std::stringstream ss(s);
	std::string tok;
	while (std::getline(ss, tok, ','))
		if (!tok.empty())
			d.push_back(atoi(tok.c_str()));
	if (d.empty())
		d.push_back(0);
	return d;
}

static bool same_box(const grid_dims& a, const grid_dims& b)
{
	for (int i = 0; i < 3; i++)
		if (a[i].begin != b[i].begin || a[i].end != b[i].end || a[i].n != b[i].n)
			return false;
	return true;
}

static MinParams min_params(const minimization_params& mp)
{
	MinParams p;
	p.maxiters = mp.maxiters;
	p.accurate_line = mp.type == minimization_params::BFGSAccurateLineSearch;
	p.early_term = mp.early_term;
	return p;
}

// parallel_mc's progress bar, fed from a host-mapped step counter
class ProgressPoller
{
	volatile unsigned* counter;
	unsigned long total;
	volatile bool done;
	boost::thread thread;

	void run()
	{
		boost::progress_display bar(total);
		unsigned long shown = 0;
		for (;;)
		{
			bool finished = done;
			unsigned long now = *counter;
			if (now > total)
				now = total;
			if (now > shown)
			{
				bar += now - shown;
				shown = now;
			}
			if (finished)
				break;
			boost::this_thread::sleep(boost::posix_time::milliseconds(50));
		}
	}
public:
	ProgressPoller(unsigned* c, unsigned long t) :
			counter(c), total(t), done(false)
	{
		thread = boost::thread(&ProgressPoller::run, this);
	}
	~ProgressPoller()
	{
		done = true;
		thread.join();
	}
};

struct EngineBase
{
	int device;
	int sm_count;
	explicit EngineBase(int d) :
			device(d), sm_count(1)
	{
		cudaDeviceProp prop;
		cuda_check(cudaSetDevice(device), "cudaSetDevice");
		cuda_check(cudaGetDeviceProperties(&prop, device),
				"cudaGetDeviceProperties");
		if (prop.computeMode == cudaComputeModeProhibited)
			throw std::runtime_error("GPU device is in compute mode prohibited");
		sm_count = prop.multiProcessorCount;
		cudaSetDeviceFlags(cudaDeviceMapHost);
	}
	virtual ~EngineBase()
	{
	}
	virtual void parallel_mc(const ::parallel_mc& par, const model& m,
			output_container& out, const precalculate& p, const grid_dims& gd,
			const vec& corner1, const vec& corner2, rng& generator,
			const grid& user_grid, double cache_slope, int tpc) = 0;
	virtual void refine(const model& m, const precalculate& p,
			const grid_dims& gd, std::vector<output_type*>& poses,
			const vec& cap, const minimization_params& minparm,
			const grid& user_grid, int tpc) = 0;
	virtual void dock_batch(GpuDocking::DockJob* const* jobs, size_t njobs,
			const precalculate& p, const grid_dims& gd, const vec& corner1,
			const vec& corner2, int seed, const grid& user_grid,
			double cache_slope, const vec& cap, unsigned* progress, int tpc) = 0;
	// concurrently resident chains (32 threads each)
	size_t capacity() const
	{
		return size_t(sm_count) * 3 * 8;
	}

	// few chains: use more threads per chain to lower latency
	int choose_tpc(int requested, size_t items) const
	{
		if (requested > 0)
			return requested;
		return (items < size_t(4 * sm_count)) ? 128 : 32;
	}
};

template<typename T>
struct Engine: public EngineBase
{
	const precalculate* tables_key;
	FlatTables flat_tables;
	DeviceTables<T> tables;

	bool have_box;
	grid_dims box;
	DeviceReceptor<T> receptor; // non_cache
	FlatGridGeometry geometry;
	DeviceGrids<T>* grids; // cache, for `box`
	double grids_slope;

	const grid* user_grid_key;
	FlatUserGrid flat_user_grid;
	DeviceUserGrid<T> user_grid;

	explicit Engine(int d) :
			EngineBase(d), tables_key(0), have_box(false), grids(0),
					grids_slope(0), user_grid_key(0)
	{
	}
	~Engine()
	{
		delete grids;
	}

	void ensure_tables(const precalculate& p)
	{
		if (tables_key == &p)
			return;
		flat_tables = gpu_model_flattener::flatten_tables(p);
		tables.upload(flat_tables);
		tables_key = &p;
		delete grids; // grids are built from the tables
		grids = 0;
	}

	void ensure_user_grid(const grid& ug)
	{
		if (user_grid_key == &ug)
			return;
		flat_user_grid = gpu_model_flattener::flatten_user_grid(ug);
		user_grid.upload(flat_user_grid);
		user_grid_key = &ug;
		delete grids;
		grids = 0;
	}

	// receptor atoms and cell lists for the box (the receptor is the same
	// for every ligand of a run)
	void ensure_box(const model& m, const grid_dims& gd, double cutoff_sqr)
	{
		if (have_box && same_box(box, gd))
			return;
		FlatReceptor r = gpu_model_flattener::flatten_receptor(m, gd, cutoff_sqr);
		geometry = gpu_model_flattener::grid_geometry(gd);
		receptor.upload(r, geometry);
		box = gd;
		have_box = true;
		delete grids;
		grids = 0;
	}

	void ensure_grids(const model& m, const std::vector<int>& types,
			double slope)
	{
		if (!grids || grids_slope != slope)
		{
			delete grids;
			grids = new DeviceGrids<T>();
			grids_slope = slope;
		}
		bool missing = false;
		for (size_t i = 0; i < types.size(); i++)
			missing |= !grids->has(types[i]);
		if (!missing)
			return;
		FlatReceptor frg = gpu_model_flattener::flatten_receptor_for_grid(m, box,
				flat_tables.cutoff_sqr);
		grids->build(frg, geometry, flat_tables, types, flat_user_grid, slope);
	}

	InterContext<T> context(bool use_cache, double slope)
	{
		InterContext<T> ic;
		ic.use_cache = use_cache;
		if (use_cache)
			ic.grids = grids->dev;
		else
			std::memset(&ic.grids, 0, sizeof(ic.grids));
		ic.rec = receptor.dev;
		ic.ug = user_grid.dev;
		ic.slope = T(slope);
		return ic;
	}

	void parallel_mc(const ::parallel_mc& par, const model& m,
			output_container& out, const precalculate& p, const grid_dims& gd,
			const vec& corner1, const vec& corner2, rng& generator,
			const grid& ug, double cache_slope, int tpc)
	{
		cuda_check(cudaSetDevice(device), "cudaSetDevice");
		ensure_tables(p);
		ensure_user_grid(ug);
		ensure_box(m, gd, p.cutoff_sqr());
		std::vector<smt> types;
		m.get_movable_atom_types(types);
		ensure_grids(m, std::vector<int>(types.begin(), types.end()),
				cache_slope);

		FlatModel fm = gpu_model_flattener::flatten_model(m);
		std::vector<const FlatModel*> fms(1, &fm);
		DeviceModels<T> models;
		models.upload(fms);

		// parallel_mc::operator(): one task per chain, seeded in order
		std::vector<ChainInput> chains;
		VINA_FOR(i, par.num_tasks)
			chains.push_back(make_chain(m, fm, 0,
					::random_int(0, 1000000, generator), corner1, corner2));
		MCParams prm = mc_params(par.mc);

		unsigned* counter = 0;
		ProgressPoller* poller = 0;
		if (par.display_progress)
		{
			cuda_check(cudaHostAlloc((void**) &counter, sizeof(unsigned),
					cudaHostAllocMapped), "cudaHostAlloc");
			*counter = 0;
			poller = new ProgressPoller(counter,
					(unsigned long) par.num_tasks * par.mc.num_steps);
		}
		std::vector<ChainResult> results;
		try
		{
			run_monte_carlo(models, tables, context(true, cache_slope), prm,
					chains, results, counter, choose_tpc(tpc, chains.size()));
		} catch (...)
		{
			delete poller;
			if (counter)
				cudaFreeHost(counter);
			throw;
		}
		delete poller;
		if (counter)
			cudaFreeHost(counter);

		// merge_output_containers (parallel_mc.cpp)
		const fl min_rmsd = 2;
		VINA_FOR_IN(t, results)
		{
			output_container task_out;
			chain_to_container(results[t], fm, task_out);
			VINA_FOR_IN(i, task_out)
				add_to_output_container(out, task_out[i], min_rmsd,
						par.mc.num_saved_mins);
		}
		out.sort();
	}

	void dock_batch(GpuDocking::DockJob* const* jobs, size_t njobs,
			const precalculate& p, const grid_dims& gd, const vec& corner1,
			const vec& corner2, int seed, const grid& ug, double cache_slope,
			const vec& cap, unsigned* progress, int tpc);

	void refine(const model& m, const precalculate& p, const grid_dims& gd,
			std::vector<output_type*>& poses, const vec& cap,
			const minimization_params& minparm, const grid& ug, int tpc)
	{
		if (poses.empty())
			return;
		cuda_check(cudaSetDevice(device), "cudaSetDevice");
		ensure_tables(p);
		ensure_user_grid(ug);
		ensure_box(m, gd, p.cutoff_sqr());

		FlatModel fm = gpu_model_flattener::flatten_model(m);
		std::vector<const FlatModel*> fms(1, &fm);
		DeviceModels<T> models;
		models.upload(fms);

		std::vector<std::vector<double> > confs(poses.size(),
				std::vector<double>(fm.conf_size));
		VINA_FOR_IN(i, poses)
			conf_to_flat(poses[i]->c, fm, &confs[i][0]);
		std::vector<int> model_of(poses.size(), 0);
		std::vector<double> e;
		std::vector<std::vector<double> > coords;
		const double v[3] = {cap[0], cap[1], cap[2]};
		// refine_structure starts from the conf's slope schedule, the
		// context slope is unused
		bool ok = run_refine(models, tables, context(false, 0),
				min_params(minparm), v, model_of, confs, e, coords,
				choose_tpc(tpc, poses.size()));
		if (!ok)
			throw std::runtime_error(
					"Ligand atom exactly overlaps receptor atom.  I can't deal with this.");
		const double maxT = std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;
		VINA_FOR_IN(i, poses)
		{
			flat_to_conf(&confs[i][0], fm, poses[i]->c);
			poses[i]->e = (e[i] >= maxT) ? max_fl : e[i];
			poses[i]->coords.clear();
			for (size_t a = 0; a + 2 < coords[i].size(); a += 3)
				poses[i]->coords.push_back(vec(coords[i][a], coords[i][a + 1],
						coords[i][a + 2]));
		}
	}
};

template<typename T>
void Engine<T>::dock_batch(GpuDocking::DockJob* const* jobs, size_t njobs,
		const precalculate& p, const grid_dims& gd, const vec& corner1,
		const vec& corner2, int seed, const grid& ug, double cache_slope,
		const vec& cap, unsigned* progress, int tpc)
{
	if (njobs == 0)
		return;
	cuda_check(cudaSetDevice(device), "cudaSetDevice");
	ensure_tables(p);
	ensure_user_grid(ug);
	ensure_box(*jobs[0]->m, gd, p.cutoff_sqr());
	std::vector<int> types;
	for (size_t j = 0; j < njobs; j++)
	{
		std::vector<smt> t;
		jobs[j]->m->get_movable_atom_types(t);
		types.insert(types.end(), t.begin(), t.end());
	}
	ensure_grids(*jobs[0]->m, types, cache_slope);

	std::vector<FlatModel> fms(njobs);
	std::vector<const FlatModel*> fmp(njobs);
	for (size_t j = 0; j < njobs; j++)
	{
		fms[j] = gpu_model_flattener::flatten_model(*jobs[j]->m);
		fmp[j] = &fms[j];
	}
	DeviceModels<T> models;
	models.upload(fmp);

	// each ligand: a fresh generator, tasks seeded in order (do_search,
	// parallel_mc::operator())
	std::vector<ChainInput> chains;
	std::vector<size_t> first_chain(njobs + 1, 0);
	for (size_t j = 0; j < njobs; j++)
	{
		const ::parallel_mc& par = jobs[j]->par;
		rng generator(static_cast<rng::result_type>(seed));
		const MCParams mp = mc_params(par.mc);
		first_chain[j] = chains.size();
		VINA_FOR(i, par.num_tasks)
		{
			ChainInput c = make_chain(*jobs[j]->m, fms[j], int(j),
					::random_int(0, 1000000, generator), corner1, corner2);
			c.num_steps = mp.num_steps;
			c.maxiters = mp.min.maxiters;
			chains.push_back(c);
		}
	}
	first_chain[njobs] = chains.size();
	const MCParams prm = mc_params(jobs[0]->par.mc);
	std::vector<ChainResult> results;
	run_monte_carlo(models, tables, context(true, cache_slope), prm, chains,
			results, progress, choose_tpc(tpc, chains.size()));

	// merge_output_containers per ligand
	std::vector<output_type*> poses;
	std::vector<int> pose_model;
	std::vector<unsigned> pose_maxiters;
	for (size_t j = 0; j < njobs; j++)
	{
		output_container& out = jobs[j]->out;
		for (size_t c = first_chain[j]; c < first_chain[j + 1]; c++)
		{
			output_container task_out;
			chain_to_container(results[c], fms[j], task_out);
			VINA_FOR_IN(i, task_out)
				add_to_output_container(out, task_out[i], 2,
						jobs[j]->par.mc.num_saved_mins);
		}
		out.sort();
		VINA_FOR_IN(i, out)
		{
			poses.push_back(&out[i]);
			pose_model.push_back(int(j));
			pose_maxiters.push_back(jobs[j]->par.mc.ssd_par.minparm.maxiters);
		}
	}

	// refine_structure for every pose of every ligand
	std::vector<std::vector<double> > confs(poses.size());
	VINA_FOR_IN(i, poses)
	{
		confs[i].resize(fms[pose_model[i]].conf_size);
		conf_to_flat(poses[i]->c, fms[pose_model[i]], &confs[i][0]);
	}
	std::vector<double> e;
	std::vector<std::vector<double> > coords;
	const double v[3] = {cap[0], cap[1], cap[2]};
	bool ok = run_refine(models, tables, context(false, 0),
			min_params(jobs[0]->par.mc.ssd_par.minparm), v, pose_model, confs, e,
			coords, choose_tpc(tpc, poses.size()), pose_maxiters);
	if (!ok)
		throw std::runtime_error(
				"Ligand atom exactly overlaps receptor atom.  I can't deal with this.");
	const double maxT = std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;
	VINA_FOR_IN(i, poses)
	{
		flat_to_conf(&confs[i][0], fms[pose_model[i]], poses[i]->c);
		poses[i]->e = (e[i] >= maxT) ? max_fl : e[i];
		poses[i]->coords.clear();
		for (size_t a = 0; a + 2 < coords[i].size(); a += 3)
			poses[i]->coords.push_back(vec(coords[i][a], coords[i][a + 1],
					coords[i][a + 2]));
	}
}

GpuDocking::GpuDocking(const GpuSettings& s) :
		settings(s)
{
	if (settings.devices.empty())
		settings.devices.push_back(0);
	int count = 0;
	cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
	for (size_t i = 0; i < settings.devices.size(); i++)
	{
		int d = settings.devices[i];
		if (d < 0 || d >= count)
		{
			std::stringstream ss;
			ss << "GPU device " << d << " not available (" << count
					<< " device(s) found)";
			throw std::runtime_error(ss.str());
		}
		if (settings.fp64)
			engines.push_back(new Engine<double>(d));
		else
			engines.push_back(new Engine<float>(d));
	}
}

GpuDocking::~GpuDocking()
{
	for (size_t i = 0; i < engines.size(); i++)
		delete engines[i];
}

void GpuDocking::parallel_mc(const ::parallel_mc& par, const model& m,
		output_container& out, const precalculate& p, const grid_dims& gd,
		const vec& corner1, const vec& corner2, rng& generator,
		const grid& user_grid, double cache_slope)
{
	engines[0]->parallel_mc(par, m, out, p, gd, corner1, corner2, generator,
			user_grid, cache_slope, settings.threads_per_chain);
}

void GpuDocking::refine(const model& m, const precalculate& p,
		const grid_dims& gd, std::vector<output_type*>& poses, const vec& cap,
		const minimization_params& minparm, const grid& user_grid)
{
	engines[0]->refine(m, p, gd, poses, cap, minparm, user_grid,
			settings.threads_per_chain);
}

size_t GpuDocking::chains_per_batch() const
{
	size_t c = 0;
	for (size_t i = 0; i < engines.size(); i++)
		c += engines[i]->capacity();
	return 4 * c;
}

void GpuDocking::dock_batch(std::vector<DockJob*>& jobs, const precalculate& p,
		const grid_dims& gd, const vec& corner1, const vec& corner2, int seed,
		const grid& user_grid, double cache_slope, const vec& cap,
		bool display_progress)
{
	if (jobs.empty())
		return;
	// split the jobs over the devices in proportion to their capacity
	const size_t nd = engines.size();
	size_t total_cap = 0;
	for (size_t d = 0; d < nd; d++)
		total_cap += engines[d]->capacity();
	std::vector<size_t> start(nd + 1, 0);
	size_t acc = 0;
	for (size_t d = 0; d < nd; d++)
	{
		start[d] = jobs.size() * acc / total_cap;
		acc += engines[d]->capacity();
	}
	start[nd] = jobs.size();

	unsigned long total_steps = 0;
	for (size_t j = 0; j < jobs.size(); j++)
		total_steps += (unsigned long) jobs[j]->par.num_tasks
				* jobs[j]->par.mc.num_steps;
	unsigned* counter = 0;
	ProgressPoller* poller = 0;
	if (display_progress)
	{
		cuda_check(cudaSetDevice(settings.devices[0]), "cudaSetDevice");
		cuda_check(cudaHostAlloc((void**) &counter, sizeof(unsigned),
				cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc");
		*counter = 0;
		poller = new ProgressPoller(counter, total_steps);
	}

	std::vector<std::string> errors(nd);
	boost::thread_group threads;
	for (size_t d = 0; d < nd; d++)
	{
		if (start[d] == start[d + 1])
			continue;
		EngineBase* e = engines[d];
		DockJob* const* first = &jobs[start[d]];
		const size_t n = start[d + 1] - start[d];
		std::string* err = &errors[d];
		const int tpc = settings.threads_per_chain;
		threads.create_thread([=, &p, &gd, &corner1, &corner2, &user_grid, &cap]()
		{
			try
			{
				e->dock_batch(first, n, p, gd, corner1, corner2, seed, user_grid,
						cache_slope, cap, counter, tpc);
			} catch (std::exception& ex)
			{
				*err = ex.what();
			}
		});
	}
	threads.join_all();
	delete poller;
	if (counter)
		cudaFreeHost(counter);
	for (size_t d = 0; d < nd; d++)
		if (!errors[d].empty())
			throw std::runtime_error(errors[d]);
}

std::string GpuDocking::description() const
{
	std::stringstream ss;
	for (size_t i = 0; i < settings.devices.size(); i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, settings.devices[i]);
		ss << (i ? ", " : "") << "GPU " << settings.devices[i] << " (" << prop.name
				<< ")";
	}
	if (settings.fp64)
		ss << ", double precision";
	return ss.str();
}

} // namespace gpusmina
