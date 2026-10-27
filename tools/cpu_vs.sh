#!/bin/bash
# CPU throughput baseline: 32 single-threaded smina processes over a split library
# usage: cpu_vs.sh smina_binary split_dir receptor autobox_ligand outdir
BIN=$1; SPLIT=$2; REC=$3; BOX=$4; OUT=$5
mkdir -p $OUT
start=$(date +%s.%N)
ls $SPLIT/*.sdf | xargs -P 32 -I{} sh -c "$BIN -r $REC -l {} --autobox_ligand $BOX --seed 1 --cpu 1 -q -o $OUT/\$(basename {}) > /dev/null 2>&1"
end=$(date +%s.%N)
echo "CPU wall time: $(echo "$end - $start" | bc) s"
