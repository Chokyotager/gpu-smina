#!/usr/bin/env python3
"""Redocking validation: CPU smina vs GPU smina on the Astex test set.

usage: redock_benchmark.py [--gpu-only] [--gpu-args "..."] [--label name]
                           cpu_smina gpu_smina testdata_dir out_dir seeds...

For every complex (dir with rec.pdb, lig.sdf) and seed, docks lig.sdf back
into rec.pdb (autobox around the crystal ligand) with both binaries and
records the top-pose RMSD to the crystal pose (obrms, symmetry corrected),
the top score and the wall time.  Results go to out_dir/results.json and a
summary is printed.  CPU and GPU runs of different complexes are run
concurrently (CPU: 4 jobs x 8 threads; GPU: one job per device).
--gpu-only skips the CPU runs, --gpu-args adds options to the GPU runs
(e.g. "--gpu_split 16") and --label names the GPU runs' files and results
(default "gpu"); results.json keeps the entries of earlier runs.
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def top_rmsd(ref, out):
    try:
        r = subprocess.run(["obrms", "-f", ref, out], capture_output=True,
                text=True, timeout=600)
        vals = [float(l.split()[-1]) for l in r.stdout.splitlines() if l.strip()]
        return vals[0] if vals else None
    except Exception:
        return None


def top_score(out):
    try:
        txt = open(out).read()
        return float(txt.split("<minimizedAffinity>\n")[1].split("\n")[0])
    except Exception:
        return None


def run(binary, cdir, seed, out, extra):
    cmd = [binary, "-r", os.path.join(cdir, "rec.pdb"), "-l",
           os.path.join(cdir, "lig.sdf"), "--autobox_ligand",
           os.path.join(cdir, "lig.sdf"), "--seed", str(seed), "-o", out,
           "-q"] + extra
    t = time.time()
    subprocess.run(cmd, capture_output=True, timeout=7200)
    return time.time() - t


def main():
    args = sys.argv[1:]
    gpu_only, gpu_args, label = False, [], "gpu"
    while args and args[0].startswith("--"):
        opt = args.pop(0)
        if opt == "--gpu-only":
            gpu_only = True
        elif opt == "--gpu-args":
            gpu_args = args.pop(0).split()
        elif opt == "--label":
            label = args.pop(0)
    cpu, gpu, data, outdir = args[:4]
    seeds = [int(s) for s in args[4:]] or [1, 2, 3, 4, 5]
    os.makedirs(outdir, exist_ok=True)
    complexes = sorted(d for d in os.listdir(data)
                       if os.path.exists(os.path.join(data, d, "lig.sdf")))
    jobs = [(c, s) for c in complexes for s in seeds]
    path = os.path.join(outdir, "results.json")
    results = json.load(open(path)) if os.path.exists(path) else {}

    def cpu_job(cs):
        c, s = cs
        out = os.path.join(outdir, "%s_s%d_cpu.sdf" % (c, s))
        t = run(cpu, os.path.join(data, c), s, out, ["--cpu", "8"])
        ref = os.path.join(data, c, "lig.sdf")
        return ("cpu", c, s, t, top_rmsd(ref, out), top_score(out))

    def gpu_job(cs, device):
        c, s = cs
        out = os.path.join(outdir, "%s_s%d_%s.sdf" % (c, s, label))
        t = run(gpu, os.path.join(data, c), s, out,
                ["--gpu", "--device", str(device)] + gpu_args)
        ref = os.path.join(data, c, "lig.sdf")
        return (label, c, s, t, top_rmsd(ref, out), top_score(out))

    with ThreadPoolExecutor(4) as cpool, ThreadPoolExecutor(2) as gpool:
        futures = [] if gpu_only else [cpool.submit(cpu_job, j) for j in jobs]
        futures += [gpool.submit(gpu_job, j, i % 2) for i, j in enumerate(jobs)]
        for f in futures:
            kind, c, s, t, rmsd, score = f.result()
            results.setdefault(c, {}).setdefault(str(s), {})[kind] = {
                "time": t, "rmsd": rmsd, "score": score}
            print(kind, c, s, "%.1fs" % t, rmsd, score, flush=True)
    json.dump(results, open(path, "w"), indent=1)

    for kind in (["gpu"] if gpu_only else ["cpu"]) + ([label] if label != "gpu" else ["gpu"]):
        rs = [results[c][str(s)][kind] for c, s in jobs if kind in results[c][str(s)]]
        ok = [r for r in rs if r["rmsd"] is not None]
        succ = sum(1 for r in ok if r["rmsd"] < 2.0)
        print("%s: success (top pose < 2 A) %d/%d = %.1f%%, mean top score %.3f,"
              " mean time %.1f s" % (kind, succ, len(ok), 100.0 * succ / len(ok),
              sum(r["score"] for r in ok) / len(ok),
              sum(r["time"] for r in rs) / len(rs)))


if __name__ == "__main__":
    main()
