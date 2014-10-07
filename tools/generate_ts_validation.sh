#!/bin/bash
dataset=$1
steps=$2
dsdir=data/twitter_$dataset
outdir=$dsdir/timesteps
infile=$dsdir/${dataset}.dat
groundfile=$dsdir/${dataset}.communities.ground
total_lines=`wc -l $infile | cut -d ' ' -f 1`
echo $total_lines
iterations=`expr $total_lines / $steps + 1`
echo "Creating target dir $outdir"
mkdir -p $outdir

for i in `seq 1 $steps`; do 
  tsoutdir=$outdir/ts${i}
  mkdir -p $tsoutdir
  dstfile=$tsoutdir/${dataset}.dat
  dstgroundfile=$tsoutdir/${dataset}.communities.ground
  headlines=`expr $i \* $iterations`
  echo "Step $i - $headlines lines in $dstfile"
  head -$headlines $infile > $dstfile
  nodesfile=$tsoutdir/${dataset}_nodes.json
  cat $dstfile| awk '{print $1}' | sort -n| uniq | awk '{str=str","$1}END{print "["substr(str,2)"]"}' > $nodesfile
  echo "Filter ground truth nodes from $nodesfile to file $dstgroundfile"
  tools/filter_ground.py $nodesfile $groundfile $dstgroundfile
done
