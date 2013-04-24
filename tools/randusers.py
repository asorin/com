#!/usr/bin/python
import sys
import random

def get_users(fname):
    usersMap = {}
    with open(fname) as f:
        for line in f:
            usrid = int(line.split('\t')[0])
            if not usrid in usersMap:
                listLinks = []
                usersMap[usrid] = listLinks
            else:
                listLinks = usersMap[usrid]
            listLinks.append(line)
    return usersMap

if len(sys.argv) < 4:
    print "usage: randusers.py <src file> <users to keep> <out file>"
    quit()

srcFile = sys.argv[1]
usersMap = get_users(srcFile)
usersList = list(usersMap.iterkeys())
toKeep = int(sys.argv[2])
outFile = sys.argv[3]

if toKeep >= len(usersList):
    print "Number of users to keep too high, maximum " + len(usersList)-1
    quit()

print "Generating %d random users" % (toKeep)

rndUsers = random.sample(range(0, len(usersList)), toKeep)
fout = open(outFile, "w")
for idx in rndUsers:
    user = usersList[idx]
    for line in usersMap[user]:
        fout.write(line)
fout.close()

print "Done to %s" % (outFile)
