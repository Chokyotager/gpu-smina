#!/usr/bin/env python3
"""Compare the top scores of two docking runs of the same ligands.

usage: compare_screen.py reference.sdf|dir other.sdf|dir [label]

Each input is an SDF file or a directory of SDF files (e.g. the outputs of
tools/cpu_vs.sh); ligands are matched by name, and a ligand's top score is
the minimizedAffinity of its first pose.  Prints the median, mean and
standard deviation of other - reference, and how many ligands differ by more
than 0.5 and 1 kcal/mol either way.
"""
import os
import statistics
import sys


def top_scores(path):
    files = ([os.path.join(path, f) for f in sorted(os.listdir(path))
              if f.endswith(".sdf")] if os.path.isdir(path) else [path])
    top = {}
    for f in files:
        for block in open(f).read().split("$$$$\n"):
            if not block.strip():
                continue
            name = block.split("\n")[0]
            if name not in top:
                top[name] = float(
                    block.split("<minimizedAffinity>\n")[1].split("\n")[0])
    return top


def main():
    a, b = top_scores(sys.argv[1]), top_scores(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else sys.argv[2]
    common = sorted(set(a) & set(b))
    d = [b[n] - a[n] for n in common]
    print("%s: %d ligands; top score difference median %+.3f, mean %+.3f, "
          "s.d. %.3f kcal/mol; better by > 0.5: %d, > 1: %d; worse by > 0.5: "
          "%d, > 1: %d" % (label, len(d), statistics.median(d),
                           statistics.mean(d), statistics.pstdev(d),
                           sum(x < -0.5 for x in d), sum(x < -1 for x in d),
                           sum(x > 0.5 for x in d), sum(x > 1 for x in d)))


if __name__ == "__main__":
    main()
