#!/usr/bin/env python3
"""Compare two kinds of runs in a redock_benchmark.py results.json.

usage: compare_redock.py results.json kind_a kind_b

Success (top pose < 2 A), mean top score and mean time of each; the
difference b - a in successes and in mean top score with a permutation test
that shuffles the a/b labels within each complex (so differences between
complexes do not count), 20000 permutations.
"""
import json
import random
import sys


def main():
    res = json.load(open(sys.argv[1]))
    a, b = sys.argv[2], sys.argv[3]
    pairs = {}  # complex -> list of (label, success, score)
    for c, seeds in res.items():
        for s, kinds in seeds.items():
            for k in (a, b):
                r = kinds.get(k)
                if r and r["rmsd"] is not None and r["score"] is not None:
                    pairs.setdefault(c, []).append(
                        (k, 1 if r["rmsd"] < 2.0 else 0, r["score"], r["time"]))
    for k in (a, b):
        rs = [x for v in pairs.values() for x in v if x[0] == k]
        print("%-12s success %d/%d = %.1f%%, mean top score %.3f, mean time %.1f s"
              % (k, sum(x[1] for x in rs), len(rs), 100.0 * sum(x[1] for x in rs)
                 / len(rs), sum(x[2] for x in rs) / len(rs),
                 sum(x[3] for x in rs) / len(rs)))

    def stats(groups):
        ds = de = 0.0
        for v in groups:
            na = sum(1 for x in v if x[0] == a)
            nb = len(v) - na
            if not na or not nb:
                continue
            ds += sum(x[1] for x in v if x[0] == b) - sum(x[1] for x in v if x[0] == a) * nb / na
            de += (sum(x[2] for x in v if x[0] == b) / nb - sum(x[2] for x in v if x[0] == a) / na) * nb
        return ds, de / max(1, sum(1 for v in groups for x in v if x[0] == b))

    groups = list(pairs.values())
    obs_s, obs_e = stats(groups)
    rnd = random.Random(1)
    n, cs, ce = 20000, 0, 0
    for _ in range(n):
        shuffled = []
        for v in groups:
            labels = [x[0] for x in v]
            rnd.shuffle(labels)
            shuffled.append([(l,) + x[1:] for l, x in zip(labels, v)])
        s, e = stats(shuffled)
        cs += abs(s) >= abs(obs_s) - 1e-9
        ce += abs(e) >= abs(obs_e) - 1e-9
    print("%s - %s: successes %+.0f (permutation p = %.2f), mean top score "
          "%+.3f kcal/mol (p = %.2f)" % (b, a, obs_s, cs / n, obs_e, ce / n))


if __name__ == "__main__":
    main()
