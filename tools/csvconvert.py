#!/usr/bin/python
import sys
import os
import time
from datetime import datetime

def getNodeId(nodesMap, name):
    if name in nodesMap:
        return nodesMap[name]
    else:
        nid = len(nodesMap)+1
        nodesMap[name] = nid
        return nid

def dumpNodes(nodesMap, fname, start):
    f=open(fname, "w")
    for key, value in sorted(nodesMap.iteritems(), key=lambda (k,v): (v,k)):
        f.write("%d\t%s\n" % (value+start, key))
    f.close()
    
if len(sys.argv) < 5:
    print "usage: csvconvert.py <src links file> <dst links file> <dst mode 1 file> <dst mode 2 file>"
    quit()

# expecting links file in format:
#    node1,node2,yyyy-mm-dd hh:mm:ss
#    node3,node4,yyyy-mm-dd hh:mm:ss
# timestamp is optional

delimiter=','
srcLinks=sys.argv[1]
dstLinks=sys.argv[2]
dstNodes1=sys.argv[3]
dstNodes2=sys.argv[4]

srcLinksF = open(srcLinks)
nodes1Map = dict()
nodes2Map = dict()
linksList = []
for line in srcLinksF:
        if line.startswith("#"):
            continue
        line=line.strip("\r\n")
        row=line.split(delimiter)
        if len(row) >= 3:
            ts = datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S")
            tsInt = 1000*int(time.mktime(ts.timetuple()))
        else:
            tsInt = 0
        node1Id = getNodeId(nodes1Map, row[0].strip())
        node2Id = getNodeId(nodes2Map, row[1].strip())
        linksList.append((node1Id, node2Id, tsInt))

dumpNodes(nodes1Map, dstNodes1, 0)
startNodes2 = len(nodes1Map)
dumpNodes(nodes2Map, dstNodes2, startNodes2)

dstLinksF = open(dstLinks, "w")
for link in linksList:
    dstLinksF.write("%d\t%d\t%s\n" % (link[0], startNodes2 + link[1], link[2]))
dstLinksF.close()

