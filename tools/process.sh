#!/bin/bash
set -e

#lastfm/artists_tags
#lastfm/users_artists
#movielens/users_movies
#lastfm/users_tags
#twitter_elections/users_domains_50k
#southern/southern

thresholds="0.05 0.1 0.5 0.7 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6"

datasets=$1
label=$2
startrun=$3
endrun=$4
threads=$5

for ds in $datasets;
do
  categ=`echo $ds | cut -d'/' -f1`
  src=`echo $ds | cut -d'/' -f2`
  
  echo Processing $ds
  tools/run_tfidf.sh $categ $src $label "$thresholds" &
  tools/run_module.sh all $categ $src $label "$thresholds" &
  wait
  
  for i in `seq $startrun $threads $endrun`; do
	  maxth=`calc $threads-1`
	  echo "Processing random steps $i - `calc $i+$maxth`" 
	  for t in `seq 0 $maxth`; do
	    step=`calc $i + $t`
	    if [ $step -le $endrun ]; then
          tools/run_random.sh $step all $categ $src $label "$thresholds" &
	    fi
	  done
      wait
  done

  echo Done $ds
done