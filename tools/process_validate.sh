#!/bin/bash
set -e

#lastfm/artists_tags
#lastfm/users_artists
#movielens/users_movies
#lastfm/users_tags
#twitter_elections/users_domains_50k
#southern/southern

#norm2max
thresholds="0.05 0.1 0.5 0.7 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6"

#southern/movielens norm2total
#thresholds="0.02 0.05 0.07 0.1 0.15 0.2 0.25 0.3"

#norm2total
#thresholds="0.1 0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 1.8 2"

datasets=$1
label="validate"

for ds in $datasets;
do
  categ=`echo $ds | cut -d'/' -f1`
  src=`echo $ds | cut -d'/' -f2`
  
  echo Processing $ds
  tools/run_tfidf.sh $categ $src $label "$thresholds"
  
  tools/run_colisted.sh $categ $src $label "$thresholds"
  
  tools/run_oslom.sh $categ $src $label "$thresholds"
  
  echo Done $ds
done
