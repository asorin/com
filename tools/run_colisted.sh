#!/bin/bash
set -e

output_info()
{
lines=`cat $1 | wc -l`
nodes=`cat $1 | awk -F"\t" '{print $1; print $2}' | sort | uniq | wc -l`
if [ $lines -gt 1 ]; then
echo "$1: $nodes nodes, $lines edges"
fi
}

categ=$1
data=$2
label=$3
tfidf_thresholds=$4

in_data=data/$categ/${data}
file=${in_data}.dat

dir_output=data/$categ/$label
dir_output_colisted=${dir_output}/colisted

dir_tfidf=data/$categ/$label/tfidf
data_tfidf=$dir_tfidf/${data}_tfidf
file_tfidf=${data_tfidf}.dat

dir_output_tfidf_simple=${dir_tfidf}/simple
dir_output_tfidf_colisted=${dir_tfidf}/colisted
mkdir -p $dir_output_tfidf_simple
mkdir -p $dir_output_tfidf_colisted

log=log/output.log

# convert original bipartite to co-listed projection
mkdir -p $dir_output_colisted
prj_colisted_file=${dir_output_colisted}/prj_usr_${data}.dat
if [ ! -f $prj_colisted_file ]; then
  bin/dcom -l $file -o $prj_colisted_file -nt 0 -a save_prj_colisted >> $log
fi
output_info $prj_colisted_file

linksDataFile=`wc -l ${file_tfidf} | cut -d' ' -f1`

# convert tf-idf filtered bipartite to simple and colisted projections
for thr in $tfidf_thresholds
do
  thr_file=${data_tfidf}_thr_${thr}.dat
  linksThrFile=`wc -l $thr_file | cut -d' ' -f1`

  if [[ $linksThrFile -ge $linksDataFile ]]; then
    echo "Skipping $thr_file: $linksThrFile" >> $log
    continue
  fi

  echo Processing $thr_file >> $log

  # convert tf-idf filtered bipartite to simple projection
  prj_simple_tfidf_file=${dir_output_tfidf_simple}/prj_usr_${data}_tfidf_thr_${thr}.dat
  if [ ! -f $prj_simple_tfidf_file ]; then
    bin/dcom -l $thr_file -o $prj_simple_tfidf_file -nt 0 -a save_prj >> $log
  fi
  output_info $prj_simple_tfidf_file

  # convert tf-idf filtered bipartite to co-listed projection
  prj_colisted_tfidf_file=${dir_output_tfidf_colisted}/prj_usr_${data}_tfidf_thr_${thr}.dat
  if [ ! -f $prj_colisted_tfidf_file ]; then
    bin/dcom -l $thr_file -o $prj_colisted_tfidf_file -nt 0 -a save_prj_colisted >> $log
  fi
  output_info $prj_colisted_tfidf_file

  # build colisted network from original biparite with same no of edges
  linksPrjThrFile=`wc -l $prj_simple_tfidf_file | cut -d' ' -f1`
  prj_colisted_thr_file=${dir_output_colisted}/prj_usr_${data}_tfidf_thr_${thr}.dat
  if [ ! -f $prj_colisted_thr_file ]; then
    cat $prj_colisted_file | sort -rn --key 3 | head -${linksPrjThrFile} | sort -n > $prj_colisted_thr_file
  fi
  output_info $prj_colisted_thr_file
done
