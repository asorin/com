#!/usr/bin/python
import sys
import random
from operator import itemgetter

if len(sys.argv) < 3:
    print "usage: randusers.py <src file> <out file>"
    quit()

def get_links(fname):
    listLinks = []
    with open(fname) as f:
        for line in f:
            rows = line.strip().split('\t')
            if len(rows)!=3:
                continue
            listLinks.append(rows)
    return listLinks

srcFn = sys.argv[1]
outFn = sys.argv[2]

linksList = get_links(srcFn)
rndTimess = random.sample(range(1, len(linksList)+1), len(linksList))
outF = open(outFn,"wt")
for i in range(0, len(linksList)):
    linksList[i][2] = rndTimess[i]

for link in sorted(linksList,key=itemgetter(2)):
    link[2] = str(link[2])
    outF.write("\t".join(link)+"\n")
