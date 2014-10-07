#!/bin/bash
#set -e

partition=$1
ds=$2
clusters=`echo $3 | awk -F"-" '{print $1,$2}'`
onlineinit=$4
onlinestep=$5

categ=`echo $ds | rev | cut -d'/' -f2- | rev`
data=`echo $ds | rev | cut -d'/' -f1 | rev`
root=~/work/com/data/$categ
dir=$root/svd-static
ground_file=${root}/${data}.communities.ground

log=~/work/com/log/output.log

mutual=~/work/mutual3/mutual

in_file=$root/${data}.dat
out_file=$dir/${data}.communities
mutual_file=$dir/${data}.mutual

#rm -f ${mutual_file}
mkdir -p $dir
for k in `seq $clusters`; do
#  if [ ! -f $out_file.${k} ]; then
    if [ "$#" -ge 5 ]; then
        extra="-oi $onlineinit -os $onlinestep"
    fi
    bin/dcom -l ${in_file} -o ${out_file}.${k} -a partition-$partition -nc $k -nt 0 $extra >> $log 2>&1
#  fi
  # calculate mutual information with ground trutuh
  score=`$mutual ${ground_file} ${out_file}.${k} | awk '{print $2}'`
  echo "$k|$score" >> ${mutual_file}
  echo "$out_file : $k|$score"
done
echo "Results written in ${mutual_file}"

