#!/bin/bash
set -e

network_info()
{
users=`cat $1 | awk '{print $1}' | sort -n | uniq | wc -l`
objects=`cat $1 | awk '{print $2}' | sort -n | uniq | wc -l`
links=`cat $1 | wc -l`
echo "$1 users=$users objects=$objects links=$links"

}

categ=$1
data=$2
label=$3
thresholds=$4

in_data=data/$categ/${data}
file=${in_data}.dat

dir_tfidf=data/$categ/$label/tfidf
data_tfidf=$dir_tfidf/${data}_tfidf
file_tfidf=${data_tfidf}.dat

mkdir -p $dir_tfidf

if [ ! -f $file_tfidf ]; then
  bin/dcom -l $file -o $file_tfidf -a transform
fi

network_info $file_tfidf

if [ -z "$thresholds" ]; then
  thresholds="0.05 0.1 0.5 0.7 1 2 3"
fi

for thr in $thresholds
do
  thr_file=${data_tfidf}_thr_${thr}.dat
  if [ ! -f $thr_file ]; then
    cat $file_tfidf | awk -v thr=$thr '($4>thr){print $0}' > $thr_file
  fi
  network_info $thr_file
done
