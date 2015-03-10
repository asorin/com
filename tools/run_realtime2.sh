#!/bin/bash
#set -e

ds=$1
nc1=`echo $2 | awk -F"-" '{print $1}'`
nc2=`echo $2 | awk -F"-" '{print $2}'`
dimensions=$3
init=$4
timestep=$5
current_time=`date +%Y%m%d%H%M%S`

categ=`echo $ds | rev | cut -d'/' -f2- | rev`
data=`echo $ds | rev | cut -d'/' -f1 | rev`
root=~/work/com/data/$categ
dir=$root/svd-static

log=~/work/com/log/output.log

mutual=~/work/mutual3/mutual

in_file=$root/${data}.dat
out_file=$dir/${data}.communities
mutual_file=$dir/${data}.mutual

#rm -f ${mutual_file}
edgestotal=`wc -l ${in_file} | cut -d ' ' -f 1`
edges=`expr $edgestotal - $init`
steps=`expr 2 + $edges / $timestep`

mkdir -p $dir
bin/dcom -l ${in_file} -o ${out_file} -a partition-real-time -nc1 $nc1 -nc2 $nc2 -ndim $dimensions -nt 0 -oi $init -ts $timestep > $log 2>&1

for k in `seq $nc1 $nc2`; do
  # calculate mutual information with ground trutuh
  for s in `seq $steps`; do
    score=`$mutual ${out_file}.k${k}.rt.$s ${out_file}.k${k}.svd.$s | awk '{print $2}'`
    echo "$current_time|$k|$s|$score" >> ${mutual_file}
    tail -1 ${mutual_file}
  done
done
echo "Resulted communities written in ${out_file}" 
echo "Resulted NMI written in ${mutual_file}"

