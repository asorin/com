#!/usr/bin/python
import sys
import random

def get_nodes(fname):
    nodesMap = {}
    with open(fname) as f:
        for line in f:
            row = line.split('\t')
            nodesMap[row[1].strip()] = row[0].strip()
    return nodesMap


if len(sys.argv) < 3:
    print "usage: groundconvert.py <src ground file> <dst ground file> <nodes map file>"
    quit()

# expecting ground truth file in format:
#    node1.1,node1.2,node1.3,...
#    node2.1,node2.2,...
# one line for each community

delimiter=','
srcGround=sys.argv[1]
dstGround=sys.argv[2]

nodesMap=get_nodes(sys.argv[3])

notFoundNodes = 0
totalNodes = 0
srcGroundF = open(srcGround, "r")
dstGroundF = open(dstGround, "w")
for line in srcGroundF:
    if ":" in line:
        line=line.split(":")[1]
    line=line.strip("\r\n ")
    row=line.split(delimiter)
    nodelist = []
    for nodeStr in row:
        nodeStr = nodeStr.strip()
        totalNodes += 1
        if nodeStr in nodesMap:
            nodeId = nodesMap[nodeStr]
            nodelist.append(nodeId)
        else:
            notFoundNodes += 1
    dstGroundF.write(" ".join(nodelist) + "\n")
dstGroundF.close()
print "Couldn't find %d nodes out of %d, nodes map size of %d" % (notFoundNodes, totalNodes, len(nodesMap))
