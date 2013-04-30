#!/bin/bash
set -e

datasets="lastfm/artists_tags"
#"lastfm/users_artists lastfm/users_tags"
#"southern/southern twitter_elections/users_domains_50k lastfm/users_tags"


thresholds="0.05 0.1 0.5 0.7 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6"
startrun=$1
endrun=$2

for ds in $datasets;
do
  categ=`echo $ds | cut -d'/' -f1`
  src=`echo $ds | cut -d'/' -f2`
  
  echo Processing $ds
#  rm -fr data/$categ/tfidf data/$categ/output data/$categ/random
#  tools/run_tfidf.sh $categ $src "$thresholds"
#  tools/run_module.sh all $categ $src "$thresholds"
  for i in `seq $startrun $endrun`; do
    tools/run_random.sh $i all $categ $src "$thresholds"
  done
  echo Done $ds
done