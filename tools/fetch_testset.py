#!/usr/bin/env python3
"""Download protein-ligand complexes from RCSB for redocking validation.

For each PDB id: rec.pdb = protein ATOM records + metal ions, lig.sdf = the
largest drug-like HETATM group (bond orders from the RCSB ModelServer).
"""
import json
import sys
import urllib.request
from pathlib import Path

EXCLUDE = set("""HOH WAT DOD SO4 PO4 GOL EDO PEG PG4 PGE 1PE ACT ACE CL NA K MG CA ZN MN
FE FE2 CU CO NI CD HG IOD BR DMS MPD MES TRS EPE BME FMT IMD NH4 NO3 SCN CIT TAR MLI
BOG LDA HEZ P6G 2PE NAG MAN BMA FUC GAL GLC SIN AZI CSO MSE""".split())
METALS = {"ZN", "MG", "CA", "MN", "FE", "FE2", "CO", "NI", "CU"}


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode()


def heavy(line):
    el = line[76:78].strip() or line[12:16].strip()[0]
    return el.upper() not in ("H", "D")


def main(ids, outroot, want):
    ok = []
    for pdb in ids:
        if len(ok) >= want:
            break
        d = Path(outroot) / pdb
        try:
            text = fetch(f"https://files.rcsb.org/download/{pdb.upper()}.pdb")
        except Exception as e:
            print(pdb, "download failed", e)
            continue
        lines = text.splitlines()
        # first model only
        if any(l.startswith("ENDMDL") for l in lines):
            lines = lines[: next(i for i, l in enumerate(lines) if l.startswith("ENDMDL"))]
        groups = {}
        for l in lines:
            if l.startswith("HETATM"):
                key = (l[17:20].strip(), l[21], l[22:26].strip())
                if key[0] in EXCLUDE:
                    continue
                if l[16] not in (" ", "A"):
                    continue
                groups.setdefault(key, []).append(l)
        cands = [(sum(heavy(x) for x in v), k) for k, v in groups.items()]
        cands = [c for c in cands if 10 <= c[0] <= 70]
        if not cands:
            print(pdb, "no ligand candidate")
            continue
        cands.sort(key=lambda c: (-c[0], c[1][1]))
        n, (resn, chain, resi) = cands[0]
        url = (f"https://models.rcsb.org/v1/{pdb}/ligand?auth_comp_id={resn}"
               f"&auth_asym_id={chain}&auth_seq_id={resi}&encoding=sdf")
        try:
            sdf = fetch(url)
        except Exception as e:
            print(pdb, "ligand fetch failed", e)
            continue
        if "$$$$" not in sdf:
            print(pdb, "bad sdf")
            continue
        sdf = sdf[: sdf.index("$$$$") + 4] + "\n"
        d.mkdir(parents=True, exist_ok=True)
        rec = [l for l in lines if l.startswith("ATOM") and l[16] in (" ", "A")]
        rec += [l for l in lines if l.startswith("HETATM") and l[17:20].strip() in METALS]
        (d / "rec.pdb").write_text("\n".join(rec) + "\nEND\n")
        (d / "lig.sdf").write_text(sdf)
        (d / "info.json").write_text(json.dumps(
            {"pdb": pdb, "ligand": resn, "chain": chain, "resi": resi, "heavy_atoms": n}))
        ok.append(pdb)
        print(pdb, resn, chain, resi, n, "ok")
    print("collected", len(ok), " ".join(ok))


if __name__ == "__main__":
    main(sys.argv[3:], sys.argv[1], int(sys.argv[2]))
