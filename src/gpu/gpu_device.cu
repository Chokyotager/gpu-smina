#include "gpu_device.h"
#include "gpu_flatten_data.h"
#include "gpu_eval.cuh"
#include <cuda_runtime.h>
#include <sstream>
#include <algorithm>

namespace gpusmina
{

void cuda_check(int err, const char* what)
{
	if (err != cudaSuccess)
	{
		std::stringstream ss;
		ss << "CUDA error in " << what << ": "
				<< cudaGetErrorString(cudaError_t(err));
		throw std::runtime_error(ss.str());
	}
}

void DevBuf::release()
{
	if (ptr)
		cudaFree(ptr);
	ptr = 0;
	bytes = 0;
}

void DevBuf::alloc(size_t n)
{
	release();
	if (n == 0)
		n = 1;
	cuda_check(cudaMalloc(&ptr, n), "cudaMalloc");
	bytes = n;
}

void DevBuf::upload(const void* src, size_t n)
{
	alloc(n);
	cuda_check(cudaMemcpy(ptr, src, n, cudaMemcpyHostToDevice), "upload");
}

void DevBuf::download(void* dst, size_t n) const
{
	cuda_check(cudaMemcpy(dst, ptr, n, cudaMemcpyDeviceToHost), "download");
}

void DevBuf::zero()
{
	cuda_check(cudaMemset(ptr, 0, bytes), "memset");
}

template<typename T>
static void set_geometry(DevGridGeometry<T>& g, const FlatGridGeometry& f)
{
	for (int i = 0; i < 3; i++)
	{
		g.dim[i] = f.dim[i];
		g.init[i] = T(f.init[i]);
		g.factor[i] = T(f.factor[i]);
		g.dim_fl_minus_1[i] = T(f.dim_fl_minus_1[i]);
		g.factor_inv[i] = T(f.factor_inv[i]);
	}
}

template<typename T>
void DeviceTables<T>::upload(const FlatTables& t)
{
	dev.kind = t.kind;
	dev.nc = t.num_components;
	dev.cutoff_sqr = T(t.cutoff_sqr);
	dev.factor = T(t.factor);
	dev.n = t.n;
	dev.fraction = T(t.fraction);
	dev.nseg = t.nseg;
	upload_converted<T>(fast, t.lin_fast);
	upload_converted<T>(smooth, t.lin_smooth);
	upload_converted<T>(spl_cutoff, t.spl_cutoff);
	upload_converted<T>(spl_data, t.spl_data);
	dev.lin_fast = fast.get<T>();
	dev.lin_smooth = smooth.get<T>();
	dev.spl_cutoff = spl_cutoff.get<T>();
	dev.spl_data = spl_data.get<T>();
}

template<typename T>
void DeviceReceptor<T>::upload(const FlatReceptor& r,
		const FlatGridGeometry& box)
{
	upload_converted<T>(coords, r.coords);
	upload_converted<int>(type, r.type);
	upload_converted<T>(charge, r.charge);
	upload_converted<int>(cell_start, r.cell_start);
	upload_converted<int>(cell_atoms, r.cell_atoms);
	dev.coords = coords.get<T>();
	dev.type = type.get<int>();
	dev.charge = charge.get<T>();
	dev.cell_start = cell_start.get<int>();
	dev.cell_atoms = cell_atoms.get<int>();
	for (int i = 0; i < 3; i++)
	{
		dev.offset[i] = r.offset[i];
		dev.dims[i] = r.dims[i];
		dev.gd_begin[i] = T(box.gd_begin[i]);
		dev.gd_end[i] = T(box.gd_end[i]);
		dev.gd_enabled[i] = box.gd_enabled[i];
	}
}

template<typename T>
void DeviceUserGrid<T>::upload(const FlatUserGrid& u)
{
	dev.initialized = u.initialized;
	if (!u.initialized)
		return;
	set_geometry(dev.geo, u.geom);
	upload_converted<T>(data, u.data);
	dev.data = data.get<T>();
}

/////////////////////////////// grid build ///////////////////////////////

template<typename T>
struct GridOutputs
{
	T* data[NUM_TYPES];
	T* chargedata[NUM_TYPES];
};

// cache::populate for one grid point per thread
template<typename T>
__global__ void build_grids_kernel(DevGridGeometry<double> geo,
		DevReceptor<double> rec, DevTables<double> tb, int ntypes,
		GridOutputs<T> out, int haschargeterms, DevUserGrid<double> ug,
		double slope, int* type_list)
{
	const long total = long(geo.dim[0]) * geo.dim[1] * geo.dim[2];
	const long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
	if (idx >= total)
		return;
	const int x = int(idx % geo.dim[0]);
	const int y = int((idx / geo.dim[0]) % geo.dim[1]);
	const int z = int(idx / (long(geo.dim[0]) * geo.dim[1]));
	const int xyz[3] = {x, y, z};
	vec3<double> probe;
	int cell[3];
	for (int k = 0; k < 3; k++)
	{
		probe[k] = geo.init[k] + geo.factor_inv[k] * double(xyz[k]);
		cell[k] = int(floor(probe[k] / 3.0)) - rec.offset[k];
	}
	const int ci = cell[0] + rec.dims[0] * (cell[1] + rec.dims[1] * cell[2]);
	const int b = rec.cell_start[ci], e = rec.cell_start[ci + 1];
	for (int j = 0; j < ntypes; j++)
	{
		const int t2 = type_list[j];
		double aff = 0, chg = 0;
		for (int k = b; k < e; k++)
		{
			const int ai = rec.cell_atoms[k];
			const vec3<double> ac = load3(rec.coords + 3 * ai);
			const double r2 = norm_sqr(ac - probe);
			if (r2 <= tb.cutoff_sqr)
			{
				double val[4];
				table_eval_fast(tb, rec.type[ai], t2, r2, val);
				const double q = rec.charge[ai];
				if (haschargeterms)
				{
					aff += val[0] + val[1] * fabs(q);
					chg += val[2] + val[3] * q;
				}
				else
					aff += val[0];
			}
		}
		if (ug.initialized)
			aff += grid_evaluate_aux(ug.geo, ug.data,
					mk(double(x), double(y), double(z)), slope, 1000.0,
					(vec3<double>*) 0);
		out.data[t2][idx] = T(aff);
		if (haschargeterms)
			out.chargedata[t2][idx] = T(chg);
	}
}

template<typename T>
DeviceGrids<T>::DeviceGrids()
{
	for (int t = 0; t < NUM_TYPES; t++)
	{
		dev.data[t] = 0;
		dev.chargedata[t] = 0;
	}
}

template<typename T>
void DeviceGrids<T>::build(const FlatReceptor& rec_for_grid,
		const FlatGridGeometry& geo, const FlatTables& tables,
		const std::vector<int>& types, const FlatUserGrid& ug, double slope)
{
	set_geometry(dev.geo, geo);
	std::vector<int> needed;
	for (size_t i = 0; i < types.size(); i++)
		if (!dev.data[types[i]] && std::find(needed.begin(), needed.end(),
				types[i]) == needed.end())
			needed.push_back(types[i]);
	if (needed.empty())
		return;

	const bool hascharge = tables.num_components > 1;
	const size_t npts = size_t(geo.dim[0]) * geo.dim[1] * geo.dim[2];
	GridOutputs<T> out;
	for (int t = 0; t < NUM_TYPES; t++)
	{
		out.data[t] = out.chargedata[t] = 0;
	}
	for (size_t i = 0; i < needed.size(); i++)
	{
		int t = needed[i];
		data[t].alloc(npts * sizeof(T));
		out.data[t] = data[t].get<T>();
		dev.data[t] = out.data[t];
		if (hascharge)
		{
			chargedata[t].alloc(npts * sizeof(T));
			out.chargedata[t] = chargedata[t].get<T>();
			dev.chargedata[t] = out.chargedata[t];
		}
	}

	DeviceTables<double> dtab;
	dtab.upload(tables);
	DeviceReceptor<double> drec;
	drec.upload(rec_for_grid, geo);
	DeviceUserGrid<double> dug;
	dug.upload(ug);
	DevGridGeometry<double> dgeo;
	set_geometry(dgeo, geo);
	DevBuf tl;
	tl.upload(&needed[0], needed.size() * sizeof(int));

	const int threads = 128;
	const int blocks = int((npts + threads - 1) / threads);
	build_grids_kernel<T><<<blocks, threads>>>(dgeo, drec.dev, dtab.dev,
			int(needed.size()), out, hascharge, dug.dev, slope, tl.get<int>());
	cuda_check(cudaGetLastError(), "build_grids_kernel launch");
	cuda_check(cudaDeviceSynchronize(), "build_grids_kernel");
}

template<typename T>
void DeviceGrids<T>::download(int t, std::vector<T>& d,
		std::vector<T>& cd) const
{
	const size_t npts = size_t(dev.geo.dim[0]) * dev.geo.dim[1] * dev.geo.dim[2];
	d.assign(npts, 0);
	cd.clear();
	if (dev.data[t])
		data[t].download(&d[0], npts * sizeof(T));
	if (dev.chargedata[t])
	{
		cd.assign(npts, 0);
		chargedata[t].download(&cd[0], npts * sizeof(T));
	}
}

/////////////////////////////// models ///////////////////////////////

template<typename T>
static Frame<T> convert(const Frame<double>& f)
{
	Frame<T> o;
	o.parent = f.parent;
	o.kind = f.kind;
	o.begin = f.begin;
	o.end = f.end;
	o.child_begin = f.child_begin;
	o.child_end = f.child_end;
	o.conf_idx = f.conf_idx;
	o.change_idx = f.change_idx;
	for (int i = 0; i < 3; i++)
	{
		o.rel_origin[i] = T(f.rel_origin[i]);
		o.rel_axis[i] = T(f.rel_axis[i]);
		o.origin[i] = T(f.origin[i]);
		o.axis[i] = T(f.axis[i]);
	}
	return o;
}

template<typename T>
static Pair<T> convert(const Pair<double>& p)
{
	Pair<T> o;
	o.a = p.a;
	o.b = p.b;
	o.tpair = p.tpair;
	o.swapped = p.swapped;
	o.abs_qa = T(p.abs_qa);
	o.abs_qb = T(p.abs_qb);
	o.qab = T(p.qab);
	return o;
}

template<typename T>
void DeviceModels<T>::upload(const std::vector<const FlatModel*>& models)
{
	host_desc.clear();
	std::vector<int> v_type, v_frame, v_children, v_level_frames,
			v_level_start, v_list_start, v_heavy;
	std::vector<double> v_charge, v_local, v_init, v_inflex, v_forigin;
	std::vector<Frame<T> > v_frames;
	std::vector<Pair<T> > v_pairs;
	std::vector<ConfBlock> v_blocks;
	max_movable = max_frames = max_list = max_change = max_conf = max_heavy = 0;

	for (size_t mi = 0; mi < models.size(); mi++)
	{
		const FlatModel& m = *models[mi];
		ModelDesc d;
		d.num_movable = m.num_movable;
		d.num_atoms = m.num_atoms;
		d.atom_offset = int(v_type.size());
		d.inflex_offset = int(v_inflex.size() / 3);
		d.frame_offset = int(v_frames.size());
		d.num_frames = int(m.frames.size());
		d.child_offset = int(v_children.size());
		d.level_offset = int(v_level_start.size());
		d.num_levels = m.num_levels();
		d.pair_offset = int(v_pairs.size());
		d.list_offset = int(v_list_start.size());
		d.num_lists = m.num_lists();
		d.block_offset = int(v_blocks.size());
		d.num_blocks = int(m.blocks.size());
		d.conf_size = m.conf_size;
		d.change_size = m.change_size;
		d.heavy_offset = int(v_heavy.size());
		d.num_heavy = int(m.heavy.size());
		d.num_ligands = m.num_ligands;
		d.num_flex = m.num_flex;
		host_desc.push_back(d);

		v_type.insert(v_type.end(), m.atom_type.begin(), m.atom_type.end());
		v_charge.insert(v_charge.end(), m.atom_charge.begin(), m.atom_charge.end());
		v_local.insert(v_local.end(), m.atom_local.begin(), m.atom_local.end());
		v_frame.insert(v_frame.end(), m.atom_frame.begin(), m.atom_frame.end());
		v_init.insert(v_init.end(), m.init_coords.begin(), m.init_coords.end());
		v_inflex.insert(v_inflex.end(), m.inflex_coords.begin(), m.inflex_coords.end());
		for (size_t i = 0; i < m.frames.size(); i++)
			v_frames.push_back(convert<T>(m.frames[i]));
		v_forigin.insert(v_forigin.end(), m.init_frame_origin.begin(),
				m.init_frame_origin.end());
		v_children.insert(v_children.end(), m.children.begin(), m.children.end());
		v_level_frames.insert(v_level_frames.end(), m.level_frames.begin(),
				m.level_frames.end());
		v_level_start.insert(v_level_start.end(), m.level_start.begin(),
				m.level_start.end());
		for (size_t i = 0; i < m.pairs.size(); i++)
			v_pairs.push_back(convert<T>(m.pairs[i]));
		v_list_start.insert(v_list_start.end(), m.list_start.begin(),
				m.list_start.end());
		v_blocks.insert(v_blocks.end(), m.blocks.begin(), m.blocks.end());
		v_heavy.insert(v_heavy.end(), m.heavy.begin(), m.heavy.end());

		max_movable = std::max(max_movable, m.num_movable);
		max_frames = std::max(max_frames, int(m.frames.size()));
		for (int l = 0; l < m.num_lists(); l++)
			max_list = std::max(max_list, m.list_start[l + 1] - m.list_start[l]);
		max_change = std::max(max_change, m.change_size);
		max_conf = std::max(max_conf, m.conf_size);
		max_heavy = std::max(max_heavy, int(m.heavy.size()));
	}

	desc.upload(&host_desc[0], host_desc.size() * sizeof(ModelDesc));
	upload_converted<int>(atom_type, v_type);
	upload_converted<T>(atom_charge, v_charge);
	upload_converted<T>(atom_local, v_local);
	upload_converted<int>(atom_frame, v_frame);
	upload_converted<T>(init_coords, v_init);
	upload_converted<T>(inflex_coords, v_inflex);
	upload_converted<Frame<T> >(frames, v_frames);
	upload_converted<T>(init_frame_origin, v_forigin);
	upload_converted<int>(children, v_children);
	upload_converted<int>(level_frames, v_level_frames);
	upload_converted<int>(level_start, v_level_start);
	upload_converted<Pair<T> >(pairs, v_pairs);
	upload_converted<int>(list_start, v_list_start);
	upload_converted<ConfBlock>(blocks, v_blocks);
	upload_converted<int>(heavy, v_heavy);

	dev.num_models = int(host_desc.size());
	dev.desc = desc.get<ModelDesc>();
	dev.atom_type = atom_type.get<int>();
	dev.atom_charge = atom_charge.get<T>();
	dev.atom_local = atom_local.get<T>();
	dev.atom_frame = atom_frame.get<int>();
	dev.init_coords = init_coords.get<T>();
	dev.inflex_coords = inflex_coords.get<T>();
	dev.frames = frames.get<Frame<T> >();
	dev.init_frame_origin = init_frame_origin.get<T>();
	dev.children = children.get<int>();
	dev.level_frames = level_frames.get<int>();
	dev.level_start = level_start.get<int>();
	dev.pairs = pairs.get<Pair<T> >();
	dev.list_start = list_start.get<int>();
	dev.blocks = blocks.get<ConfBlock>();
	dev.heavy = heavy.get<int>();
}

ScratchOffsets scratch_layout(int max_movable, int max_frames, int max_list,
		bool ordered)
{
	ScratchOffsets s;
	s.coords = 0;
	s.forces = s.coords + 3 * max_movable;
	s.frame_origin = s.forces + 3 * max_movable;
	s.frame_q = s.frame_origin + 3 * max_frames;
	s.frame_m = s.frame_q + 4 * max_frames;
	s.frame_axis = s.frame_m + 9 * max_frames;
	s.frame_ft = s.frame_axis + 3 * max_frames;
	s.pair_buf = s.frame_ft + 6 * max_frames;
	int pb = ordered ? std::max(max_movable, 4 * max_list) : 0;
	s.total = s.pair_buf + pb;
	return s;
}

template class DeviceTables<float>;
template class DeviceTables<double>;
template class DeviceReceptor<float>;
template class DeviceReceptor<double>;
template class DeviceUserGrid<float>;
template class DeviceUserGrid<double>;
template class DeviceGrids<float>;
template class DeviceGrids<double>;
template class DeviceModels<float>;
template class DeviceModels<double>;

} // namespace gpusmina
