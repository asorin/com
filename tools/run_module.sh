#!/bin/bash
set -e

output_info()
{
lines=`cat $1 | wc -l`
if [ $lines -gt 1 ]; then
echo "$1: " `tail -1 $1 | awk -F"\t" '{print $2,$3}'`
fi
}

module=$1
categ=$2
data=$3

in_data=data/$categ/${data}
file=${in_data}.dat

dir_output=data/$categ/output
file_output=output_${data}_${module}.csv

dir_tfidf=data/$categ/tfidf
data_tfidf=$dir_tfidf/${data}_tfidf
file_tfidf=${data_tfidf}.dat

conf="-c conf/${module}.yml"

mkdir -p $dir_output

bin/dcom $conf -l $file -o $file_output > /dev/null
output_info $file_output

thresholds="0.05 0.1 0.5 0.7 1 2 3"
for thr in $thresholds
do
  thr_file=${data_tfidf}_thr_${thr}.dat
  thr_file_output=${dir_output}/output_${data}_tfidf_thr_${thr}_${module}.csv
  
  bin/dcom $conf -l $thr_file -o $thr_file_output > /dev/null
  output_info $thr_file_output
done

