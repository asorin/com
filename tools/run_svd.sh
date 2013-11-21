#!/bin/bash
#set -e

categ=$1
data=$2
clusters=`echo $3 | awk -F"-" '{print $1,$2}'`

root=~/work/com/data/$categ
dir=$root/svd-static
ground_file=${root}/${data}.communities.ground

log=~/work/com/log/output.log

mutual=~/work/mutual3/mutual

in_file=$root/${data}.dat
out_file=$dir/${data}.communities
mutual_file=$dir/${data}.mutual

rm -f ${mutual_file}
mkdir -p $dir
for k in `seq $clusters`; do
#  if [ ! -f $out_file.${k} ]; then
    bin/dcom -l ${in_file} -o ${out_file}.${k} -a partition-svd -nc $k -nt 0 >> $log 2>&1
#  fi
  # calculate mutual information with ground trutuh
  score=`$mutual ${ground_file} ${out_file}.${k} | awk '{print $2}'`
  echo "$k|$score" >> ${mutual_file}
  echo "$out_file : $k|$score"
done
echo "Results written in ${mutual_file}"

