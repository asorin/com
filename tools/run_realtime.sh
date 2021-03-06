#!/bin/bash
#set -e

ds=$1
clusters=`echo $2 | awk -F"-" '{print $1,$2}'`
dimensions=$3
init=$4
timestep=$5

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
for k in `seq $clusters`; do
#  if [ ! -f $out_file.${k} ]; then
    if [ "$#" -ge 4 ]; then
        extra="-oi $init -ts $timestep"
    fi
    bin/dcom -l ${in_file} -o ${out_file}.k${k} -a partition-real-time -nc $k -ndim $dimensions -nt 0 $extra >> $log 2>&1
#  fi
  # calculate mutual information with ground trutuh
  for s in `seq $steps`; do
    score=`$mutual ${out_file}.k${k}.rt.$s ${out_file}.k${k}.svd.$s | awk '{print $2}'`
    echo "$k|$s|$score" >> ${mutual_file}
    echo "$out_file : $k|$s|$score"
  done
done
echo "Results written in ${mutual_file}"

