#!/usr/bin/python
import sys
import random

def file_len(fname):
    with open(fname) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1


if len(sys.argv) < 4:
    print "usage: randfilter.py <src file> <links to remove> <out file>"
    quit()

srcFile = sys.argv[1]
inLinksCnt = file_len(srcFile)
rmLinksCnt = int(sys.argv[2])
outFile = sys.argv[3]

if rmLinksCnt >= inLinksCnt:
    print "Number of links to remove too high, maximum " + int(inLinksCnt-1)
    quit()

print "Generating random network with %d links out of %d" % (inLinksCnt-rmLinksCnt, inLinksCnt)

rndRmLinks = random.sample(range(1, inLinksCnt+1), rmLinksCnt)
#print sorted(rndRmLinks)
lineNo = 0
fout = open(outFile, "w")
for line in open(srcFile):
    lineNo += 1
    if not lineNo in rndRmLinks:
        fout.write(line)

fout.close()

print "Done to %s" % (outFile)
