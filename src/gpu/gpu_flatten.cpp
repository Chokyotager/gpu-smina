#include "gpu_flatten.h"
#include "brick.h"
#include <stdexcept>

using namespace gpusmina;

namespace
{

void put3(std::vector<double>& out, const vec& v)
{
	out.push_back(v[0]);
	out.push_back(v[1]);
	out.push_back(v[2]);
}

void copy3(double* dst, const vec& v)
{
	dst[0] = v[0];
	dst[1] = v[1];
	dst[2] = v[2];
}

Frame<double> make_frame(int parent, int kind, const atom_range& r)
{
	Frame<double> f;
	f.parent = parent;
	f.kind = kind;
	f.begin = int(r.begin);
	f.end = int(r.end);
	f.child_begin = f.child_end = 0;
	f.conf_idx = f.change_idx = -1;
	for (int i = 0; i < 3; i++)
		f.rel_origin[i] = f.rel_axis[i] = f.origin[i] = f.axis[i] = 0;
	return f;
}

struct tree_builder
{
	FlatModel& fm;
	std::vector<std::vector<int> > kids;
	std::vector<int> depth;
	explicit tree_builder(FlatModel& fm_) :
			fm(fm_)
	{
	}

	int add(const Frame<double>& f, const vec& current_origin, int d)
	{
		int idx = int(fm.frames.size());
		fm.frames.push_back(f);
		put3(fm.init_frame_origin, current_origin);
		kids.push_back(std::vector<int>());
		depth.push_back(d);
		if (f.parent >= 0)
			kids[f.parent].push_back(idx);
		return idx;
	}

	// tree<segment> in set_conf (preorder) order; torsions are consumed in
	// the same order
	void add_branches(const branches& bs, int parent, int d, int& torsion,
			const ConfBlock& b)
	{
		VINA_FOR_IN(i, bs)
		{
			const branch& br = bs[i];
			Frame<double> f = make_frame(parent, FRAME_SEGMENT, br.node);
			copy3(f.rel_origin, br.node.get_relative_origin());
			copy3(f.rel_axis, br.node.get_relative_axis());
			f.conf_idx = b.conf_offset + (b.is_ligand ? 7 : 0) + torsion;
			f.change_idx = b.change_offset + (b.is_ligand ? 6 : 0) + torsion;
			torsion++;
			int idx = add(f, br.node.get_origin(), d);
			add_branches(br.children, idx, d + 1, torsion, b);
		}
	}
};

} // namespace

FlatModel gpu_model_flattener::flatten_model(const model& m)
{
	FlatModel fm;
	const sz nmov = m.m_num_movable_atoms;
	fm.num_movable = int(nmov);
	fm.num_atoms = int(m.atoms.size());
	VINA_CHECK(m.coords.size() == m.atoms.size());

	VINA_FOR(i, nmov)
	{
		const atom& a = m.atoms[i];
		fm.atom_type.push_back(int(a.get()));
		fm.atom_charge.push_back(a.charge);
		put3(fm.atom_local, a.coords);
		put3(fm.init_coords, m.coords[i]);
		if (!a.is_hydrogen())
			fm.heavy.push_back(int(i));
	}
	VINA_RANGE(i, nmov, m.atoms.size())
		put3(fm.inflex_coords, m.coords[i]);

	// torsion trees: ligands then flex, each in preorder (model::set order)
	tree_builder tb(fm);
	int conf_offset = 0, change_offset = 0;
	VINA_FOR_IN(li, m.ligands)
	{
		const ligand& lig = m.ligands[li];
		sz ntors = 0;
		count_torsions(lig, ntors);
		ConfBlock b;
		b.conf_offset = conf_offset;
		b.change_offset = change_offset;
		b.ntors = int(ntors);
		b.is_ligand = 1;
		b.atom_begin = int(lig.begin);
		b.atom_end = int(lig.end);
		Frame<double> root = make_frame(-1, FRAME_RIGID, lig.node);
		root.conf_idx = conf_offset;
		root.change_idx = change_offset;
		b.root_frame = tb.add(root, lig.node.get_origin(), 0);
		int torsion = 0;
		tb.add_branches(lig.children, b.root_frame, 1, torsion, b);
		VINA_CHECK(torsion == int(ntors));
		fm.blocks.push_back(b);
		conf_offset += 7 + int(ntors);
		change_offset += 6 + int(ntors);
	}
	VINA_FOR_IN(ri, m.flex)
	{
		const residue& res = m.flex[ri];
		sz ntors = 0;
		count_torsions(res, ntors);
		ConfBlock b;
		b.conf_offset = conf_offset;
		b.change_offset = change_offset;
		b.ntors = int(ntors);
		b.is_ligand = 0;
		b.atom_begin = b.atom_end = 0;
		Frame<double> root = make_frame(-1, FRAME_FIRST_SEGMENT, res.node);
		copy3(root.origin, res.node.get_origin());
		copy3(root.axis, res.node.get_axis());
		root.conf_idx = conf_offset;
		root.change_idx = change_offset;
		b.root_frame = tb.add(root, res.node.get_origin(), 0);
		int torsion = 1;
		tb.add_branches(res.children, b.root_frame, 1, torsion, b);
		VINA_CHECK(torsion == int(ntors));
		fm.blocks.push_back(b);
		conf_offset += int(ntors);
		change_offset += int(ntors);
	}
	fm.conf_size = conf_offset;
	fm.change_size = change_offset;
	fm.num_ligands = int(m.ligands.size());
	fm.num_flex = int(m.flex.size());

	// children (CSR, in branch order) and depth levels
	VINA_FOR_IN(f, fm.frames)
	{
		fm.frames[f].child_begin = int(fm.children.size());
		fm.children.insert(fm.children.end(), tb.kids[f].begin(),
				tb.kids[f].end());
		fm.frames[f].child_end = int(fm.children.size());
	}
	int maxdepth = 0;
	VINA_FOR_IN(f, tb.depth)
		maxdepth = std::max(maxdepth, tb.depth[f]);
	fm.level_start.push_back(0);
	for (int d = 0; d <= maxdepth && !fm.frames.empty(); d++)
	{
		VINA_FOR_IN(f, tb.depth)
			if (tb.depth[f] == d)
				fm.level_frames.push_back(int(f));
		fm.level_start.push_back(int(fm.level_frames.size()));
	}

	// every movable atom belongs to exactly one frame
	fm.atom_frame.assign(nmov, -1);
	VINA_FOR_IN(f, fm.frames)
		for (int i = fm.frames[f].begin; i < fm.frames[f].end; i++)
		{
			VINA_CHECK(i < int(nmov) && fm.atom_frame[i] == -1);
			fm.atom_frame[i] = int(f);
		}
	VINA_FOR(i, nmov)
		VINA_CHECK(fm.atom_frame[i] >= 0);

	// pair lists in model::eval_deriv order: other_pairs, then ligands
	std::vector<const interacting_pairs*> lists;
	lists.push_back(&m.other_pairs);
	VINA_FOR_IN(li, m.ligands)
		lists.push_back(&m.ligands[li].pairs);
	fm.list_start.push_back(0);
	VINA_FOR_IN(l, lists)
	{
		const interacting_pairs& ps = *lists[l];
		VINA_FOR_IN(i, ps)
		{
			const interacting_pair& ip = ps[i];
			const atom& a = m.atoms[ip.a];
			const atom& b = m.atoms[ip.b];
			int ta = int(a.get()), tb_ = int(b.get());
			Pair<double> p;
			p.a = int(ip.a);
			p.b = int(ip.b);
			p.tpair = tri_index_permissive(ta, tb_);
			p.swapped = ta > tb_;
			p.abs_qa = std::abs(a.charge);
			p.abs_qb = std::abs(b.charge);
			p.qab = a.charge * b.charge;
			fm.pairs.push_back(p);
		}
		fm.list_start.push_back(int(fm.pairs.size()));
	}
	return fm;
}

FlatTables gpu_model_flattener::flatten_tables(const precalculate& p)
{
	FlatTables t;
	t.cutoff = p.m_cutoff;
	t.cutoff_sqr = p.m_cutoff_sqr;
	if (p.scoring.has_slow())
		throw std::runtime_error(
				"GPU acceleration does not support non-precalculable scoring terms");

	if (const precalculate_linear* lin =
			dynamic_cast<const precalculate_linear*>(&p))
	{
		t.kind = FlatTables::Linear;
		t.num_components = int(lin->num_components);
		t.factor = lin->factor;
		t.n = int(lin->n);
		const sz nc = lin->num_components;
		t.lin_fast.resize(sz(NUM_TYPE_PAIRS) * t.n * nc);
		t.lin_smooth.resize(sz(NUM_TYPE_PAIRS) * t.n * nc * 2);
		VINA_CHECK(lin->data.dim() == sz(NUM_TYPES));
		VINA_FOR(t2, sz(NUM_TYPES))
			VINA_RANGE(t1, 0, t2 + 1)
			{
				const precalculate_linear_element& el = lin->data(t1, t2);
				const sz pi = sz(tri_index(int(t1), int(t2)));
				VINA_CHECK(el.fast.size() == sz(t.n));
				VINA_FOR(i, sz(t.n))
					VINA_FOR(c, nc)
					{
						sz idx = (pi * t.n + i) * nc + c;
						t.lin_fast[idx] = el.fast[i][c];
						t.lin_smooth[2 * idx] = el.smooth[c][i].first;
						t.lin_smooth[2 * idx + 1] = el.smooth[c][i].second;
					}
			}
	}
	else if (const precalculate_splines* spl =
			dynamic_cast<const precalculate_splines*>(&p))
	{
		t.kind = FlatTables::Spline;
		const sz nc = p.scoring.num_used_components();
		t.num_components = int(nc);
		t.spl_cutoff.assign(sz(NUM_TYPE_PAIRS) * nc, 0);
		// first pass: force the lazily built splines and find the layout
		VINA_FOR(t2, sz(NUM_TYPES))
			VINA_RANGE(t1, 0, t2 + 1)
			{
				const spline_cache& sc = spl->data(t1, t2);
				sc.eval(0); // builds the splines if needed
				VINA_CHECK(sc.num_splines == nc);
				VINA_FOR(c, nc)
				{
					const Spline& s = sc.splines[c];
					if (s.getCutoff() == 0)
						continue; // never initialized: evaluates to zero
					if (t.nseg == 0)
					{
						t.nseg = int(s.getData().size());
						t.fraction = s.getFraction();
					}
					VINA_CHECK(int(s.getData().size()) == t.nseg);
					VINA_CHECK(s.getFraction() == t.fraction);
				}
			}
		t.spl_data.assign(sz(NUM_TYPE_PAIRS) * nc * t.nseg * 5, 0);
		VINA_FOR(t2, sz(NUM_TYPES))
			VINA_RANGE(t1, 0, t2 + 1)
			{
				const spline_cache& sc = spl->data(t1, t2);
				const sz pi = sz(tri_index(int(t1), int(t2)));
				VINA_FOR(c, nc)
				{
					const Spline& s = sc.splines[c];
					t.spl_cutoff[pi * nc + c] = s.getCutoff();
					const std::vector<SplineData>& d = s.getData();
					VINA_FOR_IN(k, d)
					{
						double* o = &t.spl_data[((pi * nc + c) * t.nseg + k) * 5];
						o[0] = d[k].x;
						o[1] = d[k].a;
						o[2] = d[k].b;
						o[3] = d[k].c;
						o[4] = d[k].d;
					}
				}
			}
	}
	else
	{
		throw std::runtime_error(
				"GPU acceleration requires the linear or spline approximation");
	}
	return t;
}

namespace
{

void local_dims(const grid_dims& gd, int offset[3], int dims[3])
{
	// szv_grid_cache::get_local_dims, granularity 3
	const fl granularity = 3.0;
	for (int i = 0; i < 3; i++)
	{
		offset[i] = int(std::floor(gd[i].begin / granularity));
		dims[i] = int(std::ceil(gd[i].end / granularity)) - offset[i] + 1;
	}
}

// szv_grid_cache::compute_relevant
void relevant_atoms(const model& m, const atomv& grid_atoms,
		const grid_dims& gd, double cutoff_sqr, szv& out)
{
	vec start, end;
	for (sz i = 0; i < 3; i++)
	{
		start[i] = gd[i].begin;
		end[i] = gd[i].end;
	}
	VINA_FOR_IN(i, grid_atoms)
	{
		const atom& a = grid_atoms[i];
		if (a.acceptable_type() && !a.is_hydrogen()
				&& brick_distance_sqr(start, end, a.coords) < cutoff_sqr)
			out.push_back(i);
	}
}

// fill cell lists given the brick of every cell, per dimension
void build_cells(const atomv& grid_atoms, const szv& relevant,
		double cutoff_sqr, const std::vector<double> lower[3],
		const std::vector<double> upper[3], FlatReceptor& r)
{
	r.cell_start.clear();
	r.cell_atoms.clear();
	r.cell_start.push_back(0);
	for (int z = 0; z < r.dims[2]; z++)
		for (int y = 0; y < r.dims[1]; y++)
			for (int x = 0; x < r.dims[0]; x++)
			{
				vec lo(lower[0][x], lower[1][y], lower[2][z]);
				vec hi(upper[0][x], upper[1][y], upper[2][z]);
				VINA_FOR_IN(ri, relevant)
				{
					const atom& a = grid_atoms[relevant[ri]];
					if (brick_distance_sqr(lo, hi, a.coords) < cutoff_sqr)
						r.cell_atoms.push_back(int(ri));
				}
				r.cell_start.push_back(int(r.cell_atoms.size()));
			}
}

void add_relevant(const atomv& grid_atoms, const szv& relevant,
		FlatReceptor& r)
{
	VINA_FOR_IN(ri, relevant)
	{
		const atom& a = grid_atoms[relevant[ri]];
		put3(r.coords, a.coords);
		r.type.push_back(int(a.get()));
		r.charge.push_back(a.charge);
	}
}

} // namespace

FlatReceptor gpu_model_flattener::flatten_receptor(const model& m,
		const grid_dims& gd, double cutoff_sqr)
{
	FlatReceptor r;
	szv relevant;
	relevant_atoms(m, m.grid_atoms, gd, cutoff_sqr, relevant);
	add_relevant(m.grid_atoms, relevant, r);
	local_dims(gd, r.offset, r.dims);
	// szv_grid_cache::get bricks: [floor(c/3)*3, ceil(c/3)*3]; for any
	// coordinate not exactly on a cell boundary this is the full cell
	std::vector<double> lower[3], upper[3];
	for (int d = 0; d < 3; d++)
		for (int i = 0; i < r.dims[d]; i++)
		{
			lower[d].push_back((r.offset[d] + i) * 3.0);
			upper[d].push_back((r.offset[d] + i + 1) * 3.0);
		}
	build_cells(m.grid_atoms, relevant, cutoff_sqr, lower, upper, r);
	return r;
}

FlatReceptor gpu_model_flattener::flatten_receptor_for_grid(const model& m,
		const grid_dims& gd, double cutoff_sqr)
{
	FlatReceptor r;
	szv relevant;
	relevant_atoms(m, m.grid_atoms, gd, cutoff_sqr, relevant);
	add_relevant(m.grid_atoms, relevant, r);
	local_dims(gd, r.offset, r.dims);

	// cache::populate fills each szv cell from the first probe (x, y, z
	// loop order) that falls into it; the brick is [floor, ceil] of that
	// probe's coordinate, which is degenerate for probes exactly on a
	// 3 Angstrom boundary.  Reproduce that per dimension.
	grid g(gd, false);
	const fl granularity = 3.0;
	std::vector<double> lower[3], upper[3];
	for (int d = 0; d < 3; d++)
	{
		lower[d].assign(r.dims[d], 0);
		upper[d].assign(r.dims[d], 0);
		std::vector<bool> seen(r.dims[d], false);
		for (sz i = 0; i <= gd[d].n; i++)
		{
			sz idx[3] = {0, 0, 0};
			idx[d] = i;
			vec probe = g.index_to_argument(idx[0], idx[1], idx[2]);
			fl c = probe[d];
			int cell = int(std::floor(c / granularity)) - r.offset[d];
			VINA_CHECK(cell >= 0 && cell < r.dims[d]);
			if (!seen[cell])
			{
				seen[cell] = true;
				lower[d][cell] = std::floor(c / granularity) * granularity;
				upper[d][cell] = std::ceil(c / granularity) * granularity;
			}
		}
	}
	build_cells(m.grid_atoms, relevant, cutoff_sqr, lower, upper, r);
	return r;
}

FlatGridGeometry gpu_model_flattener::grid_geometry(const grid_dims& gd)
{
	FlatGridGeometry geo;
	grid g(gd, false); // grid::init computes the exact values used by cache
	for (int i = 0; i < 3; i++)
	{
		geo.dim[i] = int(gd[i].n + 1);
		geo.init[i] = g.m_init[i];
		geo.factor[i] = g.m_factor[i];
		geo.dim_fl_minus_1[i] = g.m_dim_fl_minus_1[i];
		geo.factor_inv[i] = g.m_factor_inv[i];
		geo.gd_begin[i] = gd[i].begin;
		geo.gd_end[i] = gd[i].end;
		geo.gd_enabled[i] = gd[i].n > 0;
	}
	return geo;
}

FlatUserGrid gpu_model_flattener::flatten_user_grid(const grid& g)
{
	FlatUserGrid u;
	u.initialized = g.initialized();
	if (!u.initialized)
		return u;
	u.geom.dim[0] = int(g.data.dim0());
	u.geom.dim[1] = int(g.data.dim1());
	u.geom.dim[2] = int(g.data.dim2());
	for (int i = 0; i < 3; i++)
	{
		u.geom.init[i] = g.m_init[i];
		u.geom.factor[i] = g.m_factor[i];
		u.geom.dim_fl_minus_1[i] = g.m_dim_fl_minus_1[i];
		u.geom.factor_inv[i] = g.m_factor_inv[i];
		u.geom.gd_begin[i] = u.geom.gd_end[i] = 0;
		u.geom.gd_enabled[i] = 0;
	}
	u.data.resize(g.data.dim0() * g.data.dim1() * g.data.dim2());
	VINA_FOR(z, g.data.dim2())
		VINA_FOR(y, g.data.dim1())
			VINA_FOR(x, g.data.dim0())
				u.data[x + g.data.dim0() * (y + g.data.dim1() * z)] = g.data(x, y, z);
	return u;
}

namespace gpusmina
{

void conf_to_flat(const conf& c, const FlatModel& fm, double* out)
{
	VINA_CHECK(c.ligands.size() == sz(fm.num_ligands));
	VINA_CHECK(c.flex.size() == sz(fm.num_flex));
	for (int bi = 0; bi < int(fm.blocks.size()); bi++)
	{
		const ConfBlock& b = fm.blocks[bi];
		double* o = out + b.conf_offset;
		const flv* tors;
		if (b.is_ligand)
		{
			const ligand_conf& lc = c.ligands[bi];
			copy3(o, lc.rigid.position);
			o[3] = lc.rigid.orientation.R_component_1();
			o[4] = lc.rigid.orientation.R_component_2();
			o[5] = lc.rigid.orientation.R_component_3();
			o[6] = lc.rigid.orientation.R_component_4();
			o += 7;
			tors = &lc.torsions;
		}
		else
			tors = &c.flex[bi - fm.num_ligands].torsions;
		VINA_CHECK(tors->size() == sz(b.ntors));
		VINA_FOR_IN(i, *tors)
			o[i] = (*tors)[i];
	}
}

void flat_to_conf(const double* in, const FlatModel& fm, conf& c)
{
	c.ligands.resize(fm.num_ligands);
	c.flex.resize(fm.num_flex);
	for (int bi = 0; bi < int(fm.blocks.size()); bi++)
	{
		const ConfBlock& b = fm.blocks[bi];
		const double* p = in + b.conf_offset;
		flv* tors;
		if (b.is_ligand)
		{
			ligand_conf& lc = c.ligands[bi];
			lc.rigid.position = vec(p[0], p[1], p[2]);
			lc.rigid.orientation = qt(p[3], p[4], p[5], p[6]);
			p += 7;
			tors = &lc.torsions;
		}
		else
			tors = &c.flex[bi - fm.num_ligands].torsions;
		tors->resize(b.ntors);
		VINA_FOR(i, sz(b.ntors))
			(*tors)[i] = p[i];
	}
}

void change_to_flat(const change& g, const FlatModel& fm, double* out)
{
	for (int bi = 0; bi < int(fm.blocks.size()); bi++)
	{
		const ConfBlock& b = fm.blocks[bi];
		double* o = out + b.change_offset;
		const flv* tors;
		if (b.is_ligand)
		{
			const ligand_change& lc = g.ligands[bi];
			copy3(o, lc.rigid.position);
			copy3(o + 3, lc.rigid.orientation);
			o += 6;
			tors = &lc.torsions;
		}
		else
			tors = &g.flex[bi - fm.num_ligands].torsions;
		VINA_FOR_IN(i, *tors)
			o[i] = (*tors)[i];
	}
}

void flat_to_change(const double* in, const FlatModel& fm, change& g)
{
	for (int bi = 0; bi < int(fm.blocks.size()); bi++)
	{
		const ConfBlock& b = fm.blocks[bi];
		const double* p = in + b.change_offset;
		flv* tors;
		if (b.is_ligand)
		{
			ligand_change& lc = g.ligands[bi];
			lc.rigid.position = vec(p[0], p[1], p[2]);
			lc.rigid.orientation = vec(p[3], p[4], p[5]);
			p += 6;
			tors = &lc.torsions;
		}
		else
			tors = &g.flex[bi - fm.num_ligands].torsions;
		VINA_FOR_IN(i, *tors)
			(*tors)[i] = p[i];
	}
}

} // namespace gpusmina
