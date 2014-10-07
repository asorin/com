#!/bin/bash
ds=$1
steps=$2
for i in `seq 1 $steps`; do 
  dspath=twitter_$ds/timesteps/ts$i/$ds
  lines=`wc -l data/${dspath}.dat | cut -d ' ' -f1`
  init=`expr $lines / 2`
  echo "real-time|$i|$dspath"
  tools/run_svd.sh real-time $dspath "2 10" $init 1
  echo "svd-batch|$i|$dspath"
  tools/run_svd.sh svd $dspath "2 10"
  echo
done
