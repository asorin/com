#!/bin/bash
set -e

categ=$1
data=$2
label=$3
thresholds=$4

root=~/work/com/data/$categ
dir=$root/$label
dir_tfidf=${dir}/tfidf
dir_colisted=${dir}/colisted
dir_tfidf_simple=${dir_tfidf}/simple
dir_tfidf_colisted=${dir_tfidf}/colisted
dirs="$dir_colisted $dir_tfidf_simple $dir_tfidf_colisted"
ground_file=${root}/${data}.communities.ground

log=~/work/com/log/output.log

oslom=~/work/OSLOM2/oslom_undir
mutual=~/work/mutual3/mutual

cd $dir_colisted
in_file=prj_usr_${data}.dat
out_file=prj_usr_${data}.communities
mutual_file=prj_usr_${data}.mutual

if [ ! -f $out_file ]; then
  $oslom -w -f $in_file >> $log 2>&1
  cat tp | grep -v "^#" > $out_file
  rm -fr tp prj_usr_${data}.dat_oslo_files
  # calculate mutual information with ground trutuh
  $mutual $ground_file $out_file | awk '{print $2}' > $mutual_file
fi
echo "$dir/$out_file : `cat $mutual_file`"

for dir in $dirs
do
  cd $dir
  for thr in $thresholds
  do
    in_file=prj_usr_${data}_tfidf_thr_${thr}.dat
    out_file=prj_usr_${data}_tfidf_thr_${thr}.communities
    mutual_file=prj_usr_${data}_tfidf_thr_${thr}.mutual

    if [ ! -f $dir/$in_file ]; then
      continue
    fi
    # detect communities using OSLOM method
    echo "Finding communities in $dir/$in_file" >> $log
    if [ ! -f $out_file ]; then
      $oslom -w -f $in_file >> $log 2>&1
      cat tp | grep -v "^#" > $out_file
      rm -fr tp prj_usr_${data}_tfidf_thr_${thr}.dat_oslo_files
      # calculate mutual information with ground trutuh
      $mutual $ground_file $out_file | awk '{print $2}' > $mutual_file
    fi
    echo "$dir/$out_file : `cat $mutual_file`"
  done
done
