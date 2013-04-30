#!/bin/bash
set -e

output_info()
{
lines=`cat $1 | wc -l`
if [ $lines -gt 1 ]; then
echo "$1: " `tail -1 $1 | awk -F"\t" '{for (i=1;i<=NF;i++) print $i}'`
fi
}

output_info_general()
{
lines=`cat $1 | wc -l`
if [ $lines -gt 1 ]; then
echo "$1: " `tail -1 $1 | awk -F"\t" '{print " prj_users="$8", prj_objects="$9}'`
fi
}

output_info_modularity()
{
lines=`cat $1 | wc -l`
if [ $lines -gt 1 ]; then
echo "$1: " `tail -1 $1 | awk -F"\t" '{print " mod_users="$2", mod_objects="$3}'`
fi
}

mod=$1
categ=$2
data=$3
thresholds=$4

if [ "$mod" = "all" ]; then
  modules="modularity general"
else
  modules=$mod
fi

in_data=data/$categ/${data}
file=${in_data}.dat

dir_output=data/$categ/output

dir_tfidf=data/$categ/tfidf
data_tfidf=$dir_tfidf/${data}_tfidf
file_tfidf=${data_tfidf}.dat

log=log/output.log

mkdir -p $dir_output

echo Processing $file > $log 
for module in $modules; do
  file_output=$dir_output/output_${data}_${module}.csv
  bin/dcom -c conf/${module}.yml -l $file -o $file_output >> $log
  output_info_$module $file_output
done

linksDataFile=`wc -l $file | cut -d' ' -f1`

if [ -z "$thresholds" ]; then
  thresholds="0.05 0.1 0.5 0.7 1 2 3"
fi

for thr in $thresholds
do
  thr_file=${data_tfidf}_thr_${thr}.dat
  linksThrFile=`wc -l $thr_file | cut -d' ' -f1`

  if [[ $linksThrFile -ge $linksDataFile ]]; then
    echo "Skipping $thr_file: $linksThrFile" >> $log
    continue
  fi

  echo Processing $thr_file >> $log
  for module in $modules; do
    thr_file_output=${dir_output}/output_${data}_tfidf_thr_${thr}_${module}.csv
    bin/dcom -c conf/${module}.yml -l $thr_file -o $thr_file_output >> $log
    output_info_$module $thr_file_output
  done
done

