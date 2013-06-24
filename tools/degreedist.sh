#!/bin/bash
set -e 

dataset=$1
nodes=$2

data="data/${dataset}"
in_file="${data}.dat"
dist="${data}/dist"
degrees_file="${dist}_degrees.dat"
out_file="${dist}_powerlaw.out"

mkdir -p $dist
cat $in_file | awk -F"\t" -v col=$nodes '{print $col}' | sort -n | uniq -c | awk '{print $1}' > $degrees_file
tools/degreedist.py $degrees_file 1>$out_file 2>/dev/null
rm -f $degrees_file
cat $out_file