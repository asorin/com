#!/bin/bash
set -e 

network_info()
{
users=`cat $1 | awk '{print $1}' | sort -n | uniq | wc -l`
objects=`cat $1 | awk '{print $2}' | sort -n | uniq | wc -l`
links=`cat $1 | wc -l`
echo "$1 users=$users objects=$objects links=$links"

}

output_info()
{
lines=`cat $1 | wc -l`
if [ $lines -gt 1 ]; then
echo "$1: " `tail -1 $1 | awk -F"\t" '{for (i=1;i<=NF;i++) print $i}'`
fi
}

output_info_general_random()
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

idx=$1
mod=$2
categ=$3
data=$4
thresholds=$5

if [ "$mod" = "all" ]; then
  modules="modularity general_random"
else
  modules=$mod
fi

in_data=data/$categ/${data}
file=${in_data}.dat

dir_random=data/$categ/random/$idx
dir_output=data/$categ/random/$idx/output
mkdir -p $dir_output

dir_tfidf=data/$categ/tfidf
data_tfidf=$dir_tfidf/${data}_tfidf

linksDataFile=`wc -l $file | cut -d' ' -f1`

log=log/output.log

if [ -z "$thresholds" ]; then
  thresholds="0.05 0.1 0.5 0.7 1 2 3"
fi

for thr in $thresholds
do
  thr_file=${data_tfidf}_thr_${thr}.dat
  linksThrFile=`wc -l $thr_file | cut -d' ' -f1`
  linksToRm=`calc $linksDataFile - $linksThrFile`
  random_file=${dir_random}/random_${data}_tfidf_thr_${thr}.dat
  if [[ $linksToRm -le 0 || $linksToRm -ge $linksDataFile ]]; then
    echo "Skipping $thr_file: $linksThrFile" >> $log
    continue
  fi
  # generate random files
  tools/randfilter.py $file $linksToRm $random_file >> $log
  network_info $random_file
  
  # calculate stuff
  for module in $modules; do
    random_file_output=${dir_output}/output_random_${data}_tfidf_thr_${thr}_${module}.csv
    bin/dcom -c conf/${module}.yml -l $random_file -o $random_file_output >> $log
    output_info_$module $random_file_output
  done
done