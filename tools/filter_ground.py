#!/usr/bin/python
import sys
import os
import time
import json

if len(sys.argv) < 4:
    print "usage: filter_ground.py <nodes file> <src ground truth file> <dst ground truth file>"
    quit()

nodesFn=sys.argv[1]
srcGroundFn=sys.argv[2]
dstGroundFn=sys.argv[3]

nodesList = json.load(open(nodesFn,"rt"))
dstGroundF = open(dstGroundFn,"wt")

for line in open(srcGroundFn,"rt"):
    srcNodes = line.strip().split(" ")
    dstNodes = [node for node in srcNodes if int(node) in nodesList]
    dstGroundF.write("%s\n" % " ".join(dstNodes))

