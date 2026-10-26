/*
 * Shared setup for the GPU validation programs: load a receptor/ligand the
 * way smina's main() does and build the scoring function and approximation.
 */

#ifndef SMINA_GPU_TEST_COMMON_H
#define SMINA_GPU_TEST_COMMON_H

#include <string>
#include <vector>
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>
#include <openbabel/mol.h>
#include <openbabel/atom.h>
#include <openbabel/obiter.h>
#include <openbabel/obconversion.h>
#include "common.h"
#include "model.h"
#include "parse_pdbqt.h"
#include "file.h"
#include "molgetter.h"
#include "flexinfo.h"
#include "obmolopener.h"
#include "custom_terms.h"
#include "weighted_terms.h"
#include "builtinscoring.h"
#include "precalculate.h"
#include "box.h"
#include "tee.h"

namespace gputest
{

// main.cpp: create_init_model (receptor via OpenBabel unless plain pdbqt)
inline void load_receptor(const std::string& rigid_name, FlexInfo& finfo,
		model& initm)
{
	if (!finfo.hasContent()
			&& boost::filesystem::extension(rigid_name) == ".pdbqt")
	{
		ifile rigidin(rigid_name);
		initm = parse_receptor_pdbqt(rigid_name, rigidin);
		return;
	}
	using namespace OpenBabel;
	obmol_opener fileopener;
	OBConversion conv;
	conv.SetOutFormat("PDBQT");
	conv.AddOption("r", OBConversion::OUTOPTIONS);
	conv.AddOption("c", OBConversion::OUTOPTIONS);
	fileopener.openForInput(conv, rigid_name);
	OBMol rec;
	if (!conv.Read(&rec))
		throw file_error(rigid_name, true);
	rec.AddHydrogens(true);
	FOR_ATOMS_OF_MOL(a, rec)
	{
		a->GetPartialCharge();
	}
	OBMol rigid;
	std::string flexstr;
	finfo.extractFlex(rec, rigid, flexstr);
	std::string recstr = conv.WriteString(&rigid);
	std::stringstream recstream(recstr);
	if (flexstr.size() > 0)
	{
		std::stringstream flexstream(flexstr);
		initm = parse_receptor_pdbqt(rigid_name, recstream, "", flexstream);
	}
	else
		initm = parse_receptor_pdbqt(rigid_name, recstream);
}

// the scoring function selection of main()
inline void setup_terms(custom_terms& t, const std::string& builtin,
		const std::string& custom_file)
{
	if (custom_file.size() > 0)
	{
		ifile custom(custom_file);
		t.add_terms_from_file(custom);
	}
	else if (builtin.size() > 0)
	{
		if (!builtin_scoring_functions.set(t, builtin))
			throw std::runtime_error("unknown builtin scoring " + builtin);
	}
	else
	{
		t.add("gauss(o=0,_w=0.5,_c=8)", -0.035579);
		t.add("gauss(o=3,_w=2,_c=8)", -0.005156);
		t.add("repulsion(o=0,_c=8)", 0.840245);
		t.add("hydrophobic(g=0.5,_b=1.5,_c=8)", -0.035069);
		t.add("non_dir_h_bond(g=-0.7,_b=0,_c=8)", -0.587439);
		t.add("num_tors_div", 5 * 0.05846 / 0.1 - 1);
	}
}

inline boost::shared_ptr<precalculate> make_prec(const weighted_terms& wt,
		const std::string& approx, fl factor)
{
	if (approx == "spline")
		return boost::shared_ptr<precalculate>(
				new precalculate_splines(wt, factor));
	return boost::shared_ptr<precalculate>(new precalculate_linear(wt, factor));
}

// box around the ligand (autobox_ligand semantics on the model's movable
// atoms, +add on every side) with smina's 0.375 granularity
inline grid_dims ligand_box(const model& m, fl add)
{
	return m.movable_atoms_box(2 * add, 0.375);
}

} // namespace gputest

#endif
