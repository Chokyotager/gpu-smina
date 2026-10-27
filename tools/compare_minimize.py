#!/usr/bin/env python3
"""Compare --minimize outputs pose by pose.

usage: compare_minimize.py ref.sdf other.sdf [label]

Prints how many minimizedAffinity values agree within 1e-5 / 0.01 / 0.1
kcal/mol, the largest difference, and how many poses have identical
coordinates (to the 4 decimals of the SDF).
"""
import sys


def poses(path):
    out = []
    for block in open(path).read().split("$$$$\n"):
        if not block.strip():
            continue
        lines = block.split("\n")
        e = float(block.split("<minimizedAffinity>\n")[1].split("\n")[0])
        natoms = int(lines[3][:3])
        xyz = [l[:30] for l in lines[4:4 + natoms]]
        out.append((e, xyz))
    return out


def main():
    a, b = poses(sys.argv[1]), poses(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else sys.argv[2]
    assert len(a) == len(b), (len(a), len(b))
    d = [abs(x[0] - y[0]) for x, y in zip(a, b)]
    same_xyz = sum(1 for x, y in zip(a, b) if x[1] == y[1])
    n = len(d)
    print("%s: %d poses; |dE| <= 1e-5: %d, <= 0.01: %d (%.1f%%), <= 0.1: %d;"
          " max %.3f; identical coords %d" % (
              label, n, sum(x <= 1e-5 for x in d), sum(x <= 0.01 for x in d),
              100.0 * sum(x <= 0.01 for x in d) / n, sum(x <= 0.1 for x in d),
              max(d), same_xyz))


if __name__ == "__main__":
    main()
