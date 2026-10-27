#include "gpu_docking.h"
#include "gpu_search.h"
#include "coords.h"
#include <cuda_runtime.h>
#include <boost/progress.hpp>
#include <boost/thread/thread.hpp>
#include <sstream>
#include <cfloat>
#include <cmath>
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

// --fast (GpuSettings::chain_split 0): chains per search task when a device
// docks `tasks` search tasks (exhaustiveness x ligands) at once: the largest
// power of two from 4 to 32 that keeps the chains within 4 waves of what the
// device holds at 32 threads per chain.  A few ligands leave most of the GPU
// idle and get 32; a screen already fills the GPU and only gains at its
// tail, while shorter chains need more evaluations for the same number of
// steps.  At least 4: below 32, --fast also halves the steps, which longer
// chains do not tolerate (screen top scores drift).  Measured:
// README.GPU.md.
static int auto_split(size_t tasks, size_t capacity)
{
	int split = 32;
	while (split > 4 && tasks * split > 4 * capacity)
		split /= 2;
	return split;
}

// --fast (GpuSettings::step_fraction 0): half the steps once the GPU is
// full (split below 32), where that halves the time; with fewer chains a
// run is mostly fixed overhead, and the full search keeps its scores
static double auto_fraction(int split)
{
	return split >= 32 ? 1.0 : 0.5;
}

struct EngineBase
{
	int device;
	int sm_count;
	Residency mc_res, bfgs_res; // of this engine's kernels
	int chain_split; // GpuSettings::chain_split (0: automatic)
	double step_fraction; // GpuSettings::step_fraction (0: automatic)
	bool skip_visited; // GpuSettings::skip_visited
	explicit EngineBase(int d) :
			device(d), sm_count(1), chain_split(1), step_fraction(1),
					skip_visited(false)
	{
		cudaDeviceProp prop;
		cuda_check(cudaSetDevice(device), "cudaSetDevice");
		// block (rather than spin) while waiting for the GPU so the host
		// threads that parse and finish ligands keep their CPU; harmless if
		// another engine already set the flags on this device
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync | cudaDeviceMapHost);
		cudaGetLastError();
		cuda_check(cudaGetDeviceProperties(&prop, device),
				"cudaGetDeviceProperties");
		if (prop.computeMode == cudaComputeModeProhibited)
			throw std::runtime_error("GPU device is in compute mode prohibited");
		sm_count = prop.multiProcessorCount;
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
	virtual void minimize_batch(GpuDocking::MinimizeJob* jobs, size_t njobs,
			const precalculate& p, const vec& cap, const grid& user_grid,
			int tpc) = 0;
	// the receptor's data and grid maps for every atom type of `jobs`
	virtual void prepare_docking(GpuDocking::DockJob* const* jobs, size_t njobs,
			const precalculate& p, const grid_dims& gd, const grid& user_grid,
			double cache_slope) = 0;
	// search and refinement of `jobs` (after prepare_docking)
	virtual void dock_sub(GpuDocking::DockJob* const* jobs, size_t njobs,
			const precalculate& p, const vec& corner1, const vec& corner2,
			int seed, double cache_slope, const vec& cap, unsigned* progress,
			int tpc) = 0;
	double fraction_for(int split) const
	{
		return step_fraction > 0 ? step_fraction : auto_fraction(split);
	}

	// chains per search task when this device docks `tasks` tasks at once
	int split_for(size_t tasks) const
	{
		return chain_split > 0 ? chain_split : auto_split(tasks, capacity());
	}

	// parallel_mc's tasks for a ligand, appended to `chains`: one chain per
	// task, seeded in order from `generator` (parallel_mc::operator()); with
	// split n, n chains per task with 1/n of the steps each; with
	// step_fraction f, f of the steps
	void add_chains(std::vector<ChainInput>& chains, const model& m,
			const FlatModel& fm, int model_index, const ::parallel_mc& par,
			rng& generator, const vec& corner1, const vec& corner2,
			int split) const
	{
		const MCParams mp = mc_params(par.mc);
		VINA_FOR(i, par.num_tasks * split)
		{
			ChainInput c = make_chain(m, fm, model_index,
					::random_int(0, 1000000, generator), corner1, corner2);
			c.num_steps = chain_steps(mp.num_steps, split, fraction_for(split));
			c.maxiters = mp.min.maxiters;
			chains.push_back(c);
		}
	}
	static unsigned chain_steps(unsigned steps, int split, double fraction)
	{
		return std::max(1u, unsigned(std::ceil(steps * fraction / split)));
	}

	// chains resident at 32 threads each (the throughput configuration)
	size_t capacity() const
	{
		return mc_res.items[0];
	}

	// threads per item for a bfgs launch: a launch takes as long as its
	// slowest item, so use the most threads per item that keeps all items
	// resident; beyond that, 32 threads per item give the best throughput
	int choose_tpc(int requested, size_t items) const
	{
		return requested > 0 ? requested : bfgs_res.tpc(items);
	}
};

template<typename T>
struct Engine: public EngineBase
{
	const precalculate* tables_key;
	FlatTables flat_tables;
	DeviceTables<T> tables;

	bool have_receptor;
	DeviceReceptor<T> receptor; // non_cache, any box
	bool have_box;
	grid_dims box;
	FlatGridGeometry geometry;
	DeviceGrids<T>* grids; // cache, for `box`
	double grids_slope;

	const grid* user_grid_key;
	FlatUserGrid flat_user_grid;
	DeviceUserGrid<T> user_grid;

	explicit Engine(int d) :
			EngineBase(d), tables_key(0), have_receptor(false), have_box(false),
					grids(0), grids_slope(0), user_grid_key(0)
	{
		cuda_check(cudaSetDevice(device), "cudaSetDevice");
		preload_kernels<T>();
		mc_res = mc_residency<T>();
		bfgs_res = bfgs_residency<T>();
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

	// receptor atoms and cell lists (the receptor is the same for every
	// ligand of a run)
	void ensure_receptor(const model& m, double cutoff_sqr)
	{
		if (have_receptor)
			return;
		receptor.upload(gpu_model_flattener::flatten_receptor(m, cutoff_sqr));
		have_receptor = true;
	}

	// the docking box (grids are built for it)
	void ensure_box(const model& m, const grid_dims& gd, double cutoff_sqr)
	{
		ensure_receptor(m, cutoff_sqr);
		if (have_box && same_box(box, gd))
			return;
		geometry = gpu_model_flattener::grid_geometry(gd);
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
		ic.box = make_box<T>(geometry);
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

		std::vector<ChainInput> chains;
		const int split = split_for(par.num_tasks);
		add_chains(chains, m, fm, 0, par, generator, corner1, corner2, split);
		MCParams prm = mc_params(par.mc);
		prm.visited = skip_visited;

		unsigned* counter = 0;
		ProgressPoller* poller = 0;
		if (par.display_progress)
		{
			cuda_check(cudaHostAlloc((void**) &counter, sizeof(unsigned),
					cudaHostAllocMapped), "cudaHostAlloc");
			*counter = 0;
			poller = new ProgressPoller(counter, (unsigned long) chains.size()
					* chain_steps(par.mc.num_steps, split, fraction_for(split)));
		}
		std::vector<ChainResult> results;
		try
		{
			run_monte_carlo(models, tables, context(true, cache_slope), prm,
					chains, results, counter, tpc, &mc_res);
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

	void prepare_docking(GpuDocking::DockJob* const* jobs, size_t njobs,
			const precalculate& p, const grid_dims& gd, const grid& ug,
			double cache_slope)
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
	}
	void dock_sub(GpuDocking::DockJob* const* jobs, size_t njobs,
			const precalculate& p, const vec& corner1, const vec& corner2,
			int seed, double cache_slope, const vec& cap, unsigned* progress,
			int tpc);

	void minimize_batch(GpuDocking::MinimizeJob* jobs, size_t njobs,
			const precalculate& p, const vec& cap, const grid& ug, int tpc)
	{
		if (njobs == 0)
			return;
		cuda_check(cudaSetDevice(device), "cudaSetDevice");
		ensure_tables(p);
		ensure_user_grid(ug);
		ensure_receptor(*jobs[0].m, p.cutoff_sqr());
		std::vector<FlatModel> fms(njobs);
		std::vector<const FlatModel*> fmp(njobs);
		std::vector<std::vector<double> > confs(njobs);
		std::vector<int> model_of(njobs);
		std::vector<unsigned> maxiters(njobs);
		std::vector<FlatGridGeometry> boxes(njobs);
		for (size_t j = 0; j < njobs; j++)
		{
			fms[j] = gpu_model_flattener::flatten_model(*jobs[j].m);
			fmp[j] = &fms[j];
			confs[j].resize(fms[j].conf_size);
			conf_to_flat(jobs[j].out->c, fms[j], &confs[j][0]);
			model_of[j] = int(j);
			maxiters[j] = jobs[j].minparm.maxiters;
			boxes[j] = gpu_model_flattener::grid_geometry(jobs[j].gd);
		}
		DeviceModels<T> models;
		models.upload(fmp);
		std::vector<double> e;
		std::vector<std::vector<double> > coords;
		const double v[3] = {cap[0], cap[1], cap[2]};
		bool ok = run_refine(models, tables, context(false, 0),
				min_params(jobs[0].minparm), v, model_of, confs, e, coords,
				choose_tpc(tpc, njobs), maxiters, boxes);
		if (!ok)
			throw std::runtime_error(
					"Ligand atom exactly overlaps receptor atom.  I can't deal with this.");
		const double maxT = std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;
		for (size_t j = 0; j < njobs; j++)
		{
			output_type& o = *jobs[j].out;
			flat_to_conf(&confs[j][0], fms[j], o.c);
			o.e = (e[j] >= maxT) ? max_fl : e[j];
			o.coords.clear();
			for (size_t a = 0; a + 2 < coords[j].size(); a += 3)
				o.coords.push_back(vec(coords[j][a], coords[j][a + 1],
						coords[j][a + 2]));
		}
	}

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
void Engine<T>::dock_sub(GpuDocking::DockJob* const* jobs, size_t njobs,
		const precalculate& p, const vec& corner1, const vec& corner2, int seed,
		double cache_slope, const vec& cap, unsigned* progress, int tpc)
{
	if (njobs == 0)
		return;
	cuda_check(cudaSetDevice(device), "cudaSetDevice");

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
	size_t tasks = 0;
	for (size_t j = 0; j < njobs; j++)
		tasks += jobs[j]->par.num_tasks;
	const int split = split_for(tasks);
	for (size_t j = 0; j < njobs; j++)
	{
		rng generator(static_cast<rng::result_type>(seed));
		first_chain[j] = chains.size();
		add_chains(chains, *jobs[j]->m, fms[j], int(j), jobs[j]->par, generator,
				corner1, corner2, split);
	}
	first_chain[njobs] = chains.size();
	MCParams prm = mc_params(jobs[0]->par.mc);
	prm.visited = skip_visited;
	std::vector<ChainResult> results;
	run_monte_carlo(models, tables, context(true, cache_slope), prm, chains,
			results, progress, tpc, &mc_res);

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
		engines.back()->chain_split = std::max(0, settings.chain_split);
		engines.back()->step_fraction = settings.step_fraction;
		engines.back()->skip_visited = settings.skip_visited;
	}
}

GpuDocking::~GpuDocking()
{
	for (size_t i = 0; i < engines.size(); i++)
		delete engines[i];
	for (size_t i = 0; i < local_engines.size(); i++)
		if (!settings.fp64) // shared with `engines` in double precision runs
			delete local_engines[i];
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

void GpuDocking::minimize_batch(std::vector<MinimizeJob>& jobs,
		const precalculate& p, const vec& cap, const grid& user_grid)
{
	if (jobs.empty())
		return;
	if (local_engines.empty())
	{
		if (settings.fp64)
			local_engines = engines;
		else
			for (size_t i = 0; i < settings.devices.size(); i++)
				local_engines.push_back(new Engine<double>(settings.devices[i]));
	}
	const std::vector<EngineBase*>& engines = local_engines;
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
	std::vector<std::string> errors(nd);
	boost::thread_group threads;
	for (size_t d = 0; d < nd; d++)
	{
		if (start[d] == start[d + 1])
			continue;
		EngineBase* e = engines[d];
		MinimizeJob* first = &jobs[start[d]];
		const size_t n = start[d + 1] - start[d];
		std::string* err = &errors[d];
		const int tpc = settings.threads_per_chain;
		threads.create_thread([=, &p, &cap, &user_grid]()
		{
			try
			{
				e->minimize_batch(first, n, p, cap, user_grid, tpc);
			} catch (std::exception& ex)
			{
				*err = ex.what();
			}
		});
	}
	threads.join_all();
	for (size_t d = 0; d < nd; d++)
		if (!errors[d].empty())
			throw std::runtime_error(errors[d]);
}

size_t GpuDocking::chains_per_batch() const
{
	size_t c = 0;
	for (size_t i = 0; i < engines.size(); i++)
		c += engines[i]->capacity();
	// with chain_split, the same ligands per batch (more, shorter chains
	// pack the GPU better), up to 32 times the resident chains (memory)
	if (settings.chain_split <= 0) // automatic: adapts to the batch
		return 4 * c;
	return std::max<size_t>(1, std::min(4 * c, 32 * c / settings.chain_split));
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
	for (size_t d = 0; d < nd; d++)
	{
		size_t tasks = 0;
		for (size_t j = start[d]; j < start[d + 1]; j++)
			tasks += jobs[j]->par.num_tasks;
		const int split = engines[d]->split_for(tasks);
		for (size_t j = start[d]; j < start[d + 1]; j++)
			total_steps += (unsigned long) jobs[j]->par.num_tasks * split
					* EngineBase::chain_steps(jobs[j]->par.mc.num_steps, split,
							engines[d]->fraction_for(split));
	}
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
				e->prepare_docking(first, n, p, gd, user_grid, cache_slope);
				e->dock_sub(first, n, p, corner1, corner2, seed, cache_slope, cap,
						counter, tpc);
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
