#!/usr/bin/env python3
"""Write a synthetic AutoDock-style .map for testing smina's --user_grid.

usage: make_user_grid.py out.map cx cy cz [npts] [spacing]

The map holds a few smooth Gaussian wells and bumps so that interpolated
values and gradients are non-trivial everywhere.
"""
import math
import sys


def main():
    out = sys.argv[1]
    cx, cy, cz = (float(v) for v in sys.argv[2:5])
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 60
    spacing = float(sys.argv[6]) if len(sys.argv) > 6 else 0.375
    half = n * spacing / 2
    centers = [(cx + dx, cy + dy, cz + dz, a, w) for dx, dy, dz, a, w in [
        (0.0, 0.0, 0.0, -1.5, 4.0), (3.0, -2.0, 1.0, 0.8, 2.5),
        (-4.0, 2.5, -1.5, -0.9, 3.0), (1.5, 4.0, 3.0, 0.5, 2.0)]]
    with open(out, "w") as f:
        f.write("GRID_PARAMETER_FILE test.gpf\n")
        f.write("GRID_DATA_FILE test.maps.fld\n")
        f.write("MACROMOLECULE test.pdbqt\n")
        f.write("SPACING %.3f\n" % spacing)
        f.write("NELEMENTS %d %d %d\n" % (n, n, n))
        f.write("CENTER %.3f %.3f %.3f\n" % (cx, cy, cz))
        for k in range(n + 1):
            z = cz - half + k * spacing
            for j in range(n + 1):
                y = cy - half + j * spacing
                for i in range(n + 1):
                    x = cx - half + i * spacing
                    v = 0.0
                    for px, py, pz, a, w in centers:
                        d2 = (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2
                        v += a * math.exp(-d2 / (2 * w * w))
                    f.write("%.4f\n" % v)


if __name__ == "__main__":
    main()
