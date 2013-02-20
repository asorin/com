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
    print "usage: csvconvert.py <src file> <dst links file> <dst mode 1 file> <dst mode 2 file>"
    quit()
   
delimiter=','
src=sys.argv[1]
dstLinks=sys.argv[2]
dstNodes1=sys.argv[3]
dstNodes2=sys.argv[4]


srcF = open(src)
nodes1Map = dict()
nodes2Map = dict()
linksList = []
for line in srcF:
        line=line.strip("\r\n")
        row=line.split(delimiter)
        if len(row) < 3:
            continue
        node1Id = getNodeId(nodes1Map, row[1])
        node2Id = getNodeId(nodes2Map, row[2])
        ts = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
        linksList.append((node1Id, node2Id, 1000*int(time.mktime(ts.timetuple()))))

dumpNodes(nodes1Map, dstNodes1, 0)
startNodes2 = len(nodes1Map)
dumpNodes(nodes2Map, dstNodes2, startNodes2)

dstF = open(dstLinks, "w")
for link in linksList:
    dstF.write("%d\t%d\t%s\n" % (link[0], startNodes2 + link[1], link[2]))
dstF.close()
